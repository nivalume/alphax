"""The research protocol: propose -> approve -> run -> oos -> forward -> report, plus calibrate and import-legacy."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

from alphax_lab.protocol import cards as card_rules
from alphax_lab.protocol import experiment_type, report
from alphax_lab.protocol import policy as policies
from alphax_lab.protocol.ledger import Ledger
from alphax_lab.protocol.versions import pipeline_version


def spec_id(card: dict, spec: dict) -> str:
    return hashlib.sha256(json.dumps({"card": card, "spec": spec}, sort_keys=True).encode()).hexdigest()[:12]


def _dump(value) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def _summary(result: dict) -> dict:
    """What the ledger keeps from a run: enough for gates, forward z-scores and the report index."""
    daily = np.asarray(result["daily_returns_1x"], dtype=float)
    return {
        "net": {k: result["net"][k] for k in ("t_stat", "sharpe", "days", "annualized_return", "max_drawdown")},
        "sharpe_2x": result["sharpe_2x"],
        "daily_mean": float(daily.mean()) if daily.size else None,
        "daily_std": float(daily.std(ddof=1)) if daily.size > 1 else None,
    }


class Lab:
    def __init__(self, root: str | Path, *, commit: bool = True) -> None:
        self.ledger = Ledger(root, commit=commit)
        self.root = self.ledger.root
        self.policy_path = self.root / "policy.yaml"
        new_policy = not self.policy_path.exists()
        self.policy = policies.load(self.policy_path)
        if new_policy:
            self.ledger.commit("policy: defaults", [self.policy_path])
        self.policy_version = policies.version(self.policy)
        self.pipeline_version = pipeline_version(self.policy_version)
        ignore = self.root / ".gitignore"
        if not ignore.exists():
            ignore.write_text("pending/\n")

    # ------------------------------------------------------------ helpers

    def _versions(self) -> dict:
        return {"policy_version": self.policy_version, "pipeline_version": self.pipeline_version}

    def is_current(self, entry: dict) -> bool:
        return entry.get("policy_version") == self.policy_version and entry.get("pipeline_version") == self.pipeline_version

    def pending(self) -> dict[str, dict]:
        return {p.stem: yaml.safe_load(p.read_text()) for p in sorted((self.root / "pending").glob("*.yaml"))}

    def tried(self) -> dict[str, str]:
        return self.ledger.tried_structures()

    def _write_report(self, name: str, text: str, data: dict) -> list[Path]:
        md = self.root / "reports" / f"{name}.md"
        js = self.root / "reports" / f"{name}.json"
        md.write_text(text)
        js.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float))
        return [md, js]

    # ------------------------------------------------------------ propose / review / approve

    def propose(
        self,
        llm,
        *,
        symbols: list[str],
        n: int,
        prompt: str | None = None,
        pdf: str | None = None,
        pages: tuple[int, int] | None = None,
    ) -> list[dict]:
        from alphax_lab import pdf as pdf_text
        from alphax_lab import propose

        notes = propose.pdf_mechanisms(llm, pdf_text.chunks(pdf, pages), pdf) if pdf else None
        items = propose.generate(
            llm, symbols=symbols, n=n, tried=self.tried(), policy=self.policy, prompt=prompt, pdf_notes=notes, pdf_path=pdf
        )
        for item in items:
            (self.root / "pending" / f"{item['id']}.yaml").write_text(_dump(item))
        log = self.root / "pending" / f"llm-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl"
        log.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in llm.transcript) + "\n")
        return items

    def approve(self, item_id: str) -> dict:
        """Freezes a (possibly hand-edited) pending card+spec into the registry."""
        path = self.root / "pending" / f"{item_id}.yaml"
        item = yaml.safe_load(path.read_text())
        if not isinstance(item, dict):
            raise ValueError(f"{path} must contain a mapping")
        card, kind = item["card"], experiment_type.get(item["card"].get("experiment_type", "factor_ts"))
        problems = card_rules.regulate(card, self.policy["card"]["forbidden_terms"])
        spec, spec_problems = kind.regulate(item["spec"], card, self.tried(), self.policy) if not problems else (None, [])
        problems += spec_problems
        if not problems:
            assert spec is not None
            error = kind.contract_check(spec, card, self.policy)
            if error:
                problems.append(f"contract check: {error} (run `alphax-lab repair {item_id}`)")
        if problems:
            item["problems"] = problems
            path.write_text(_dump(item))
            raise ValueError(f"{item_id} cannot be registered:\n- " + "\n- ".join(problems))
        assert spec is not None
        sid = spec_id(card, spec)
        if sid in self.ledger.registered():
            raise ValueError(f"{item_id} is identical to registered spec {sid}")
        registry = self.root / "registry" / f"{sid}.yaml"
        registry.write_text(_dump({"card": card, "spec": spec}))
        entry = self.ledger.append(
            "registered", {"spec_id": sid, "card": card, "spec": spec, **self._versions()}, f"register {sid} {card['title']}",
            [registry],
        )
        path.unlink()
        return entry

    def repair(self, llm, item_id: str) -> dict:
        from alphax_lab import propose

        path = self.root / "pending" / f"{item_id}.yaml"
        item = yaml.safe_load(path.read_text())
        if not isinstance(item, dict):
            raise ValueError(f"{path} must contain a mapping")
        kind = experiment_type.get(item["card"].get("experiment_type", "factor_ts"))
        spec, problems = kind.regulate(item["spec"], item["card"], self.tried(), self.policy)
        if problems:
            error = "; ".join(problems)
        else:
            assert spec is not None
            error = kind.contract_check(spec, item["card"], self.policy)
        if not error:
            return item
        item = propose.repair(llm, item, error)
        path.write_text(_dump(item))
        return item

    # ------------------------------------------------------------ in-sample

    def run(self) -> list[dict]:
        """Runs every registered spec that has not been run, once, on in-sample data."""
        done = self.ledger.runs()
        out = []
        for sid, reg in self.ledger.registered().items():
            if sid in done:
                continue
            out.append(self._run_one(reg))
        return out

    def _run_one(
        self,
        reg: dict,
        control: str | None = None,
        *,
        synthetic: str | None = None,
        real_data: bool = True,
        tests: int | None = None,
    ) -> dict:
        """One in-sample test. `control` labels a calibration run; `synthetic` swaps the factor
        for a negative control inside the harness."""
        card, spec = reg["card"], reg["spec"]
        kind = experiment_type.get(card.get("experiment_type", "factor_ts"))
        tests = tests if tests is not None else self.ledger.in_sample_tests() + 1
        try:
            result = kind.run(spec, card, self.policy, "in_sample", control=synthetic)
        except Exception as error:  # the data was read even if the run failed: it still counts
            payload = {"spec_id": reg["spec_id"], "window": "in_sample", "tests_at_run": tests, "passed": False,
                       "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), **self._versions()}
            if control:
                return self.ledger.append("control", {**payload, "control": control, "real_data": real_data})
            return self.ledger.append("run", payload)
        gate = kind.gate(result, self.policy, tests)
        passed = all(check["passed"] for check in gate)
        payload = {
            "spec_id": reg["spec_id"],
            "window": "in_sample",
            "tests_at_run": tests,
            "passed": passed,
            "gate": gate,
            "summary": _summary(result),
            **self._versions(),
        }
        name = f"{reg['spec_id']}.{control or 'in_sample'}"
        text = report.render(payload | {"seq": len(self.ledger.entries)}, card, spec, gate, kind.report_sections(result))
        files = self._write_report(name, text, result)
        if control:
            return self.ledger.append("control", {**payload, "control": control, "real_data": real_data}, f"control {control}", files)
        return self.ledger.append("run", payload, f"run {reg['spec_id']} {'PASS' if passed else 'reject'}", files)

    # ------------------------------------------------------------ out of sample

    def oos_candidates(self) -> list[str]:
        tested = set(self.ledger.oos_results())
        return [sid for sid, run in self.ledger.runs().items() if run["passed"] and self.is_current(run) and sid not in tested]

    def oos(self, *, force: bool = False) -> dict:
        o = self.policy["oos"]
        if self.ledger.oos_batches_used() >= o["max_batches"]:
            raise RuntimeError("the out-of-sample period is used up; survivors can only be judged by forward tracking")
        batches = self.ledger.oos_batches()
        if batches and not force:
            last = datetime.fromisoformat(batches[-1]["time"])
            due = last + timedelta(days=o["min_days_between_batches"])
            if datetime.now(timezone.utc) < due:
                raise RuntimeError(f"next OOS batch is due {due:%Y-%m-%d}; --force runs it early and is recorded")
        candidates = self.oos_candidates()
        if not candidates:
            raise RuntimeError("no current in-sample pass is waiting for out-of-sample testing")
        tests = self.ledger.oos_tests() + len(candidates)
        registered = self.ledger.registered()
        results, files = {}, []
        for sid in candidates:
            card, spec = registered[sid]["card"], registered[sid]["spec"]
            kind = experiment_type.get(card.get("experiment_type", "factor_ts"))
            result = kind.run(spec, card, self.policy, "out_of_sample")
            gate = kind.oos_gate(result, self.policy, tests)
            passed = all(c["passed"] for c in gate)
            results[sid] = {"passed": passed, "gate": gate, "summary": _summary(result)}
            entry = {"spec_id": sid, "window": "out_of_sample", **self._versions()}
            files += self._write_report(f"{sid}.out_of_sample", report.render(entry, card, spec, gate, kind.report_sections(result)), result)
        return self.ledger.append(
            "oos_batch",
            {"spec_ids": candidates, "tests_at_batch": tests, "forced": force, "results": results, **self._versions()},
            f"oos batch of {len(candidates)}",
            files,
        )

    # ------------------------------------------------------------ forward

    def forward(self) -> list[dict]:
        """Shadow-tracks OOS survivors on bars that did not exist when they were frozen."""
        f = self.policy["forward"]
        oos_end = self.policy["periods"]["out_of_sample"][1]
        registered, runs = self.ledger.registered(), self.ledger.runs()
        retired = {sid for sid, cps in self.ledger.forward_checkpoints().items() if cps[-1]["status"] in ("retired", "final_fail", "final_pass")}
        out = []
        for sid, oos in self.ledger.oos_results().items():
            if not oos["passed"] or sid in retired:
                continue
            reg = registered[sid]
            start = max(reg["time"][:10], oos_end)
            kind = experiment_type.get(reg["card"].get("experiment_type", "factor_ts"))
            try:
                result = kind.run(reg["spec"], reg["card"], self.policy, f"forward:{start}")
            except ValueError as error:
                out.append({"spec_id": sid, "status": "waiting", "reason": str(error)})
                continue
            daily = np.asarray(result["daily_returns_1x"], dtype=float)
            base = runs[sid]["summary"]
            z = (daily.mean() - base["daily_mean"]) / (base["daily_std"] / np.sqrt(daily.size)) if daily.size else 0.0
            days = int(daily.size)
            if z < f["retire_z"]:
                status = "retired"
            elif days >= f["final_days"]:
                status = "final_pass" if result["net"]["sharpe"] > 0 else "final_fail"
            elif days >= f["interim_days"]:
                status = "interim"
            else:
                status = "tracking"
            out.append(
                self.ledger.append(
                    "forward",
                    {"spec_id": sid, "start": start, "days": days, "z_vs_in_sample": float(z), "status": status,
                     "sharpe": result["net"]["sharpe"], **self._versions()},
                    f"forward {sid} {status}",
                )
            )
        return out

    # ------------------------------------------------------------ report

    def status(self) -> list[dict]:
        runs, oos, fwd = self.ledger.runs(), self.ledger.oos_results(), self.ledger.forward_checkpoints()
        rows = []
        for sid, reg in self.ledger.registered().items():
            run = runs.get(sid)
            if run is None:
                stage = "registered"
            elif not self.is_current(run):
                stage = "stale (policy or pipeline changed; re-run counts as a new test)"
            elif not run["passed"]:
                stage = "rejected in-sample" if "error" not in run else "error in-sample"
            elif sid not in oos:
                stage = "passed in-sample, awaiting OOS batch"
            elif not oos[sid]["passed"]:
                stage = "rejected out-of-sample"
            elif sid in fwd:
                stage = f"forward: {fwd[sid][-1]['status']} ({fwd[sid][-1]['days']} days)"
            else:
                stage = "passed OOS, awaiting forward data"
            rows.append({"spec_id": sid, "title": reg["card"]["title"], "stage": stage})
        return rows

    def summary(self) -> str:
        rows = self.status()
        runs = [r for r in self.ledger.runs().values()]
        passed = sum(r["passed"] for r in runs)
        oos = self.ledger.oos_results()
        n = self.ledger.in_sample_tests()
        lines = [
            "# alphax-lab ledger summary",
            "",
            f"policy `{self.policy_version}` · pipeline `{self.pipeline_version}` · {len(self.ledger.entries)} ledger entries",
            "",
            f"- In-sample tests counted (N): **{n}** → next test needs t ≥ {policies.t_required(n + 1, self.policy['gate']['alpha']):.2f}",
            f"- Out-of-sample tests counted (M): **{self.ledger.oos_tests()}**; batches used {self.ledger.oos_batches_used()}"
            f" of {self.policy['oos']['max_batches']}",
            f"- In-sample: {len(runs)} run, {passed} passed ({(1 - passed / len(runs)) if runs else 0:.0%} rejected)",
            f"- Out-of-sample: {len(oos)} tested, {sum(v['passed'] for v in oos.values())} passed",
            "",
            "A healthy pipeline rejects almost everything. A high pass rate means the feedback edge leaked.",
            "",
            "| spec | title | stage |",
            "|---|---|---|",
            *[f"| `{r['spec_id']}` | {r['title']} | {r['stage']} |" for r in rows],
        ]
        survivors = [r for r in rows if r["stage"].startswith("forward: final_pass")]
        lines += ["", f"Equal-weight combination members (forward final pass): {[r['title'] for r in survivors] or 'none'}"]
        text = "\n".join(lines) + "\n"
        path = self.root / "reports" / "summary.md"
        path.write_text(text)
        self.ledger.commit("summary", [path])
        return text

    # ------------------------------------------------------------ calibration and legacy

    def import_legacy(self, manifest: str | Path) -> int:
        done = {e["source"] for e in self.ledger.entries if e["kind"] == "legacy"}
        added = 0
        for item in yaml.safe_load(Path(manifest).read_text())["entries"]:
            if item["source"] in done:
                continue
            self.ledger.append("legacy", item, f"legacy {item['source']}")
            added += 1
        return added

    def calibrate(self, symbols: list[str]) -> dict:
        """Negative controls must be rejected even at the loosest threshold; positives are recorded as found."""
        from alphax_lab.types.factor_ts import calibration

        kind = experiment_type.get("factor_ts")
        outcome = {}
        leak_spec, leak_problems = kind.regulate(calibration.LEAK_SPEC, calibration.card(symbols, "leak"), {}, self.policy)
        outcome["leak_rejected_by_regulator"] = leak_spec is None and any("future" in p for p in leak_problems)
        for name, control, real in (("random", "random", False), ("shuffle", "shuffle", False), ("leak", "leak", False)):
            card, spec = calibration.card(symbols, name), calibration.NEGATIVE_SPEC
            reg = {"spec_id": f"control-{name}", "card": card, "spec": spec}
            entry = self._run_one(reg, name, synthetic=control, real_data=real, tests=1)
            outcome[f"{name}_rejected"] = not entry["passed"]
            if name == "leak":
                gate = {c["name"]: c for c in entry.get("gate", [])}
                outcome["leak_flagged_suspicious"] = "not_suspicious" in gate and not gate["not_suspicious"]["passed"]
        for name, (card, spec) in calibration.positives(symbols).items():
            entry = self._run_one({"spec_id": f"control-{name}", "card": card, "spec": spec}, f"positive-{name}")
            outcome[f"positive_{name}"] = "passed" if entry["passed"] else "rejected"
        outcome["calibrated"] = all(
            outcome[k] for k in ("leak_rejected_by_regulator", "random_rejected", "shuffle_rejected", "leak_rejected", "leak_flagged_suspicious")
        )
        return outcome
