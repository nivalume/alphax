import json
import subprocess

import pytest
from alphax_lab.protocol.ledger import Ledger, LedgerError


def test_append_counts_and_commits(tmp_path):
    ledger = Ledger(tmp_path / "ledger")
    ledger.append("legacy", {"source": "old", "tests": 5, "oos_tests": 2})
    ledger.append("run", {"spec_id": "a", "passed": False})
    ledger.append("control", {"spec_id": "c", "real_data": False})
    ledger.append("control", {"spec_id": "d", "real_data": True})
    ledger.append("oos_batch", {"spec_ids": ["a", "b"], "results": {}})
    assert ledger.in_sample_tests() == 5 + 1 + 1
    assert ledger.oos_tests() == 2 + 2
    assert ledger.oos_batches_used() == 2  # one real batch + the legacy look
    log = subprocess.run(["git", "log", "--oneline"], cwd=tmp_path / "ledger", capture_output=True, text=True).stdout
    assert len(log.splitlines()) == 5
    assert Ledger(tmp_path / "ledger").in_sample_tests() == 7  # reloads and verifies


@pytest.mark.parametrize("tamper", ["edit", "delete"])
def test_tampering_breaks_the_chain(tmp_path, tamper):
    ledger = Ledger(tmp_path, commit=False)
    for i in range(3):
        ledger.append("run", {"spec_id": str(i), "passed": False})
    lines = ledger.file.read_text().splitlines()
    if tamper == "edit":
        entry = json.loads(lines[1])
        entry["passed"] = True
        lines[1] = json.dumps(entry, sort_keys=True)
    else:
        del lines[1]
    ledger.file.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError, match="chain broken"):
        Ledger(tmp_path, commit=False)


def test_commit_is_a_no_op_when_nothing_changed(tmp_path):
    ledger = Ledger(tmp_path)
    report = tmp_path / "reports" / "summary.md"
    report.write_text("same text")
    ledger.commit("summary", [report])
    ledger.commit("summary", [report])  # re-running `report` must not fail
    log = subprocess.run(["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert len(log.splitlines()) == 1


def test_git_errors_are_reported(tmp_path):
    ledger = Ledger(tmp_path)
    with pytest.raises(LedgerError, match="git add"):
        ledger.commit("bad", [tmp_path / "missing.md"])
