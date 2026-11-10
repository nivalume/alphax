"""alphax.processors against Qlib's own processors.

The fixture (tests/generators/generate_qlib_processors.py) runs Qlib's
``qlib.data.dataset.processor`` classes over a (datetime, instrument) frame;
here the same data is a (rows, symbols) panel.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from alphax import processors as P

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "qlib_processors.tsv"
INPUTS = ("f0", "f1", "f2", "f3", "f4", "LABEL0")
START, END = "2020-01-09", "2020-02-05"


def fixture():
    lines = [line for line in FIXTURE.read_text().splitlines() if not line.startswith("#")]
    rows = list(csv.reader(lines, delimiter="\t"))
    header, rows = rows[0], rows[1:]
    dates = list(dict.fromkeys(row[0] for row in rows))
    n = len(rows) // len(dates)

    def panel(k: int) -> np.ndarray:
        return np.array([float(row[k]) for row in rows]).reshape(len(dates), n)

    data = {name: panel(header.index(name)) for name in INPUTS}
    cases: dict[str, dict[str, np.ndarray]] = {}
    for k, name in enumerate(header):
        case, sep, field = name.partition(":")
        if sep:
            cases.setdefault(case, {})[field] = panel(k)
    return data, np.array(dates, dtype="datetime64[D]"), cases


def processors(case: str) -> list[P.Processor]:
    fit = {"fit_start_time": START, "fit_end_time": END}
    return {
        "inf": lambda: [P.ProcessInf()],
        "fillna": lambda: [P.Fillna()],
        "fillna_feature": lambda: [P.Fillna(fields_group="feature", fill_value=-1)],
        "zscore": lambda: [P.ZScoreNorm(**fit)],
        "robust": lambda: [P.RobustZScoreNorm(**fit, fields_group="feature")],
        "robust_noclip": lambda: [P.RobustZScoreNorm(**fit, clip_outlier=False)],
        "minmax": lambda: [P.MinMaxNorm(**fit)],
        "cs_zscore": lambda: [P.CSZScoreNorm()],
        "cs_zscore_label": lambda: [P.CSZScoreNorm(fields_group="label")],
        "cs_robust": lambda: [P.CSZScoreNorm(method="robust")],
        "cs_rank": lambda: [P.CSRankNorm()],
        "cs_rank_label": lambda: [P.CSRankNorm(fields_group="label")],
        "cs_fillna": lambda: [P.CSZFillna()],
        "tanh": lambda: [P.TanhProcess()],
        "dropna_label": lambda: [P.DropnaLabel()],
        "dropna_feature": lambda: [P.DropnaProcessor(fields_group="feature")],
        "drop_col": lambda: [P.DropCol(col_list=["f1", "f3"])],
        "filter_col": lambda: [P.FilterCol(fields_group="feature", col_list=["f0", "f2"])],
        "default_infer": lambda: [P.ProcessInf(), P.ZScoreNorm(**fit), P.Fillna()],
        "default_learn": lambda: [
            P.ProcessInf(),
            P.ZScoreNorm(**fit),
            P.Fillna(),
            P.DropnaLabel(),
            P.CSZScoreNorm(fields_group="label"),
        ],
        "workflow": lambda: [
            P.RobustZScoreNorm(**fit, fields_group="feature", clip_outlier=True),
            P.Fillna(fields_group="feature"),
            P.DropnaLabel(),
            P.CSRankNorm(fields_group="label"),
        ],
    }[case]()


def check(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    nan = np.isnan(expected)
    assert np.array_equal(np.isnan(actual), nan), label
    infinite = np.isinf(expected)
    assert np.array_equal(actual[infinite], expected[infinite]), label
    finite = np.isfinite(expected)
    tolerance = 1e-9 * np.maximum(np.abs(expected[finite]), 1.0)
    assert np.all(np.abs(actual[finite] - expected[finite]) <= tolerance), label


def test_every_case_matches_qlib():
    data, dates, cases = fixture()
    assert len(cases) == 21
    before = {name: x.copy() for name, x in data.items()}
    for case, expected in cases.items():
        out = P.process(data, processors(case), index=dates)
        assert list(out) == list(expected), case
        for name, want in expected.items():
            check(out[name], want, f"{case}:{name}")
    for name, x in data.items():
        np.testing.assert_array_equal(x, before[name], err_msg=f"{name} was changed in place")


def test_the_fixture_exercises_its_edge_cases():
    data, dates, cases = fixture()
    # A row of all infinities: ProcessInf has no finite mean to put there.
    assert np.isinf(data["f1"][15]).all() and np.isnan(cases["inf"]["f1"][15]).all()
    # A field constant over the fit range is left alone by ZScoreNorm.
    np.testing.assert_array_equal(cases["zscore"]["f2"], data["f2"])
    # A field without values in the fit range normalises to NaN.
    assert np.isnan(cases["zscore"]["f4"]).all()
    # The robust clip binds.
    assert (np.abs(cases["robust"]["f3"]) == 3).any()
    # DropnaLabel drops whole rows.
    dropped = np.isnan(data["LABEL0"])
    assert dropped.any() and all(np.isnan(x[dropped]).all() for x in cases["dropna_label"].values())


def test_fitted_processors_apply_their_statistics_to_new_rows():
    data, dates, _ = fixture()
    infer = processors("default_infer")
    whole = P.process(data, infer, index=dates)
    later = P.process({k: v[30:] for k, v in data.items()}, infer, index=dates[30:], fit=False)
    for name in whole:
        np.testing.assert_array_equal(later[name], whole[name][30:], err_msg=name)


def test_a_date_alone_covers_its_whole_day():
    times = np.array(
        ["2020-01-02T09:30", "2020-01-02T14:30", "2020-01-03T09:30", "2020-01-03T14:30"],
        dtype="datetime64[ns]",
    )
    x = np.array([[1.0], [3.0], [100.0], [200.0]])
    norm = P.ZScoreNorm("2020-01-02", "2020-01-02").fit({"x": x}, index=times)
    np.testing.assert_allclose(norm({"x": x})["x"][:2, 0], [-1.0, 1.0])
    # An exact time is an inclusive bound too.
    exact = P.ZScoreNorm(times[0], times[1]).fit({"x": x}, index=times)
    np.testing.assert_allclose(exact({"x": x})["x"][:2, 0], [-1.0, 1.0])


def test_misuse_is_reported():
    x = {"x": np.ones((3, 2))}
    with pytest.raises(ValueError, match="index"):
        P.ZScoreNorm(START, END).fit(x)
    with pytest.raises(RuntimeError, match="fitted"):
        P.MinMaxNorm(START, END)(x)
    with pytest.raises(ValueError, match="fields_group"):
        P.Fillna(fields_group="features")(x)
    with pytest.raises(KeyError, match="nope"):
        P.CSRankNorm(fields_group=["nope"])(x)
    with pytest.raises(ValueError, match="method"):
        P.CSZScoreNorm(method="rank")
    assert not P.DropnaLabel().is_for_infer() and P.Fillna().is_for_infer()
