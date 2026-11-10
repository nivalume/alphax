"""Fixed report template. Every number is either a pre-registered gate check or marked reference only."""

from __future__ import annotations

import yaml

REFERENCE = "*Reference only — not part of the decision. Reading it and adjusting the next card is fitting.*"


def render(entry: dict, card: dict, spec: dict, gate: list[dict], sections: list[tuple[str, str, bool]]) -> str:
    verdict = "PASSED" if all(check["passed"] for check in gate) else "REJECTED"
    lines = [
        f"# {card['title']} — {verdict}",
        "",
        f"spec `{entry['spec_id']}` · window `{entry['window']}` · policy `{entry['policy_version']}` · "
        f"pipeline `{entry['pipeline_version']}` · ledger seq {entry.get('seq', '?')}",
        "",
        "## Gate (pre-registered)",
        "",
        "| check | value | threshold | passed |",
        "|---|---|---|---|",
        *[f"| {c['name']} | {c['value']} | {c['threshold']} | {'yes' if c['passed'] else 'NO'} |" for c in gate],
        "",
    ]
    for title, body, gated in sections:
        lines += [f"## {title}{'' if gated else ' (reference)'}", "", body, ""]
        if not gated:
            lines += [REFERENCE, ""]
    lines += [
        "## Hypothesis card",
        "",
        "```yaml",
        yaml.safe_dump(card, sort_keys=False, allow_unicode=True).strip(),
        "```",
        "",
        "## Spec",
        "",
        "```yaml",
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True).strip(),
        "```",
    ]
    return "\n".join(lines) + "\n"
