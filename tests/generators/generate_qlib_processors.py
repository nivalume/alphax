"""Generate the Qlib oracle for ``alphax.processors``.

Runs Qlib's own data processors (``qlib.data.dataset.processor``) over a
deterministic panel in Qlib's layout: rows ``(datetime, instrument)``,
columns ``(feature | label, name)``. A list of processors is applied the way
``DataHandlerLP`` applies it: each one is fitted on the output of the one
before, then applied.

Writes ``qlib_processors.tsv``: one row per ``(datetime, instrument)`` of the
full grid, the inputs, then ``<case>:<name>`` for every column each case
keeps. A row a case drops (``DropnaLabel``, ``DropnaProcessor``) is ``nan``
in every one of its columns; a column a case drops is absent.

    conda run -n rdagent4qlib python tests/generators/generate_qlib_processors.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.data.dataset.processor import (
    CSRankNorm,
    CSZFillna,
    CSZScoreNorm,
    DropCol,
    DropnaLabel,
    DropnaProcessor,
    Fillna,
    FilterCol,
    MinMaxNorm,
    ProcessInf,
    RobustZScoreNorm,
    TanhProcess,
    ZScoreNorm,
)

OUT = Path(__file__).parents[1] / "fixtures" / "qlib_processors.tsv"
DATES = pd.bdate_range("2020-01-02", periods=40)
INSTRUMENTS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
FEATURES = ("f0", "f1", "f2", "f3", "f4")
FIT = {"fit_start_time": DATES[5], "fit_end_time": DATES[24]}


def frame() -> pd.DataFrame:
    """Awkward on purpose: NaN cells and whole NaN dates, infinities (one
    date where every value is infinite), a constant feature, outliers for the
    robust clip, a feature with no value in the fit range, NaN labels."""
    t, n = len(DATES), len(INSTRUMENTS)
    row = np.arange(t, dtype=np.float64)[:, None]
    col = np.arange(n, dtype=np.float64)[None, :]
    f0 = np.sin(row * 0.7 + col * 1.3) * 2.0 + col * 0.25
    f1 = np.cos(row * 0.45 - col) * 3.0 + 1.0
    f2 = np.full((t, n), 5.0)
    f3 = ((row * 7 + col * 11) % 13 - 6.0) * 0.4
    f4 = row * 0.1 + col
    label = np.sin(row * 0.31 + col * 0.77) * 0.02
    f0[[3, 17, 30], [1, 4, 0]] = np.nan
    f0[12, :] = np.nan
    f1[[2, 9, 21, 33], [0, 3, 5, 2]] = [np.inf, -np.inf, np.inf, -np.inf]
    f1[15, :] = [np.inf, -np.inf, np.inf, np.inf, -np.inf, np.inf]
    f1[26, [1, 2]] = np.nan
    f2[[8, 28], [2, 5]] = np.nan
    f3[[6, 18], [3, 1]] = [40.0, -55.0]
    f3[20, :] = np.nan
    f4[:27, :] = np.nan
    f4[30, 3] = np.nan
    label[[4, 19, 36], [2, 0, 5]] = np.nan
    label[39, :] = np.nan
    columns = pd.MultiIndex.from_tuples([("feature", f) for f in FEATURES] + [("label", "LABEL0")])
    index = pd.MultiIndex.from_product([DATES, INSTRUMENTS], names=["datetime", "instrument"])
    values = np.stack([f0, f1, f2, f3, f4, label], axis=-1).reshape(t * n, len(columns))
    return pd.DataFrame(values, index=index, columns=columns)


def cases() -> list[tuple[str, list]]:
    return [
        ("inf", [ProcessInf()]),
        ("fillna", [Fillna()]),
        ("fillna_feature", [Fillna(fields_group="feature", fill_value=-1)]),
        ("zscore", [ZScoreNorm(**FIT)]),
        ("robust", [RobustZScoreNorm(**FIT, fields_group="feature")]),
        ("robust_noclip", [RobustZScoreNorm(**FIT, clip_outlier=False)]),
        ("minmax", [MinMaxNorm(**FIT)]),
        ("cs_zscore", [CSZScoreNorm()]),
        ("cs_zscore_label", [CSZScoreNorm(fields_group="label")]),
        ("cs_robust", [CSZScoreNorm(method="robust")]),
        ("cs_rank", [CSRankNorm()]),
        ("cs_rank_label", [CSRankNorm(fields_group="label")]),
        ("cs_fillna", [CSZFillna()]),
        ("tanh", [TanhProcess()]),
        ("dropna_label", [DropnaLabel()]),
        ("dropna_feature", [DropnaProcessor(fields_group="feature")]),
        ("drop_col", [DropCol(col_list=["f1", "f3"])]),
        ("filter_col", [FilterCol(fields_group="feature", col_list=["f0", "f2"])]),
        # Qlib's handler defaults, then a typical Alpha158 workflow config.
        ("default_infer", [ProcessInf(), ZScoreNorm(**FIT), Fillna()]),
        (
            "default_learn",
            [ProcessInf(), ZScoreNorm(**FIT), Fillna(), DropnaLabel(), CSZScoreNorm(fields_group="label")],
        ),
        (
            "workflow",
            [
                RobustZScoreNorm(**FIT, fields_group="feature", clip_outlier=True),
                Fillna(fields_group="feature"),
                DropnaLabel(),
                CSRankNorm(fields_group="label"),
            ],
        ),
    ]


def run(processors: list, df: pd.DataFrame) -> pd.DataFrame:
    """As ``DataHandlerLP``: fit each processor on the previous output."""
    df = df.copy()
    for processor in processors:
        processor.fit(df)
        df = processor(df)
    return df


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, ".17g")


def main() -> None:
    qlib.init()
    df = frame()
    header = ["datetime", "instrument", *[name for _, name in df.columns]]
    columns = [df[col].to_numpy() for col in df.columns]
    specs = []
    for case, processors in cases():
        specs.append(f"# case {case} = " + " -> ".join(type(p).__name__ for p in processors))
        out = run(processors, df).reindex(df.index)
        for group, name in out.columns:
            header.append(f"{case}:{name}")
            columns.append(out[(group, name)].to_numpy(dtype=np.float64))

    with OUT.open("w", newline="") as handle:
        handle.write(f"# qlib={qlib.__version__}\n")
        handle.write(f"# numpy={np.__version__} pandas={pd.__version__}\n")
        handle.write(f"# fit range {FIT['fit_start_time'].date()} .. {FIT['fit_end_time'].date()} (inclusive)\n")
        handle.write("# rows a case drops are nan in all its columns; columns it drops are absent\n")
        for spec in specs:
            handle.write(spec + "\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for k, (date, instrument) in enumerate(df.index):
            writer.writerow([date.date().isoformat(), instrument, *(format_float(float(c[k])) for c in columns)])
    print(f"wrote {OUT} ({len(df)} rows, {len(header) - 2} columns)")


if __name__ == "__main__":
    main()
