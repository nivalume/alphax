"""One path resolves to the files a key reads: scan rules, content sniffing,
and the guarantee that a file name never carries data."""
import gzip
from pathlib import Path

import numpy as np
import alphax
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.csv as pyarrow_csv
import pyarrow.parquet as parquet

KEY = "ADAUSDC.BINANCE-1-DAY-LAST-EXTERNAL"
OTHER = "ETHUSDC.BINANCE-1-DAY-LAST-EXTERNAL"
BARS = ("open", "high", "low", "close", "volume")


def schema():
    fields = [("event_time", pa.uint64()), ("available_time", pa.uint64())] + [(name, pa.float64()) for name in BARS]
    return alphax.data.DataSchema("custom", pa.schema(fields), pa.schema(fields), lambda batch, context: batch)


def table_of(times, values=None):
    values = np.asarray(times, dtype=float) if values is None else np.asarray(values, dtype=float)
    return pa.table({
        "event_time": pa.array(times, pa.uint64()),
        "available_time": pa.array(times, pa.uint64()),
        **{name: pa.array(values, pa.float64()) for name in BARS},
    })


def write(path, times, *, values=None):
    table = table_of(times, values)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wb") as handle:
            pyarrow_csv.write_csv(table, handle)
    elif path.name.endswith(".csv"):
        pyarrow_csv.write_csv(table, path)
    else:
        parquet.write_table(table, path)
    return path


def rows(dataset):
    """The (key, available times, closes) of every decoded batch."""
    return [(context.key, part.column("available_time").to_pylist(), part.column("close").to_pylist())
            for context, part in dataset.batches]


def test_a_scan_reads_data_files_in_filename_order_and_skips_the_rest(tmp_path):
    directory = tmp_path / KEY
    write(directory / "b.parquet", [20])
    write(directory / "a.csv", [10])
    write(directory / "c.csv.gz", [30])
    write(directory / ".hidden.csv", [1])
    write(directory / "notes.txt", [1])
    (directory / "nested.csv").mkdir()
    source = alphax.data.Source(directory, schema(), KEY)
    assert [Path(one).name for one in source.items[0][1]] == ["a.csv", "b.parquet", "c.csv.gz"]
    assert [one[1] for one in rows(source.load())] == [[10], [20], [30]]


def test_one_file_holds_exactly_one_key(tmp_path):
    file = write(tmp_path / "a.csv", [10])
    assert alphax.data.Source(file, schema(), KEY).prepare().shape == (1, 1)
    with pytest.raises(ValueError, match="single file"):
        alphax.data.Source(file, schema(), [KEY, OTHER])


def test_a_truncated_gzip_stream_fails_instead_of_returning_rows(tmp_path):
    path = tmp_path / "broken.csv.gz"
    path.write_bytes(b"\x1f\x8b\x08\x00\x00")
    with pytest.raises(ValueError):
        alphax.data.Source(path, schema(), KEY).load()


def test_gzipped_parquet_is_refused_with_a_reason(tmp_path):
    packed = tmp_path / "rows.parquet.gz"
    write(tmp_path / "rows.parquet", [10])
    with gzip.open(packed, "wb") as handle:
        handle.write((tmp_path / "rows.parquet").read_bytes())
    with pytest.raises(ValueError, match="gzipped Parquet"):
        alphax.data.Source(packed, schema(), KEY).load()


def test_parquet_content_under_a_csv_name_is_still_parquet(tmp_path):
    path = tmp_path / "rows.csv"
    parquet.write_table(table_of([10], [3.0]), path)
    panel = alphax.data.Source(path, schema(), KEY).prepare()
    assert panel.close[0, 0] == 3.0


def test_a_key_that_names_no_instrument_is_rejected_by_the_source_binding(tmp_path):
    file = write(tmp_path / "a.csv", [10])
    source = alphax.data.Source(file, schema(), "not-a-valid-key")
    with pytest.raises(ValueError, match="invalid instrument"):
        source.load()


def test_every_file_of_one_key_is_read_and_the_range_filters_by_availability(tmp_path):
    directory = tmp_path / "bars" / KEY
    write(directory / "2026-01-01_2026-02-01.parquet", [0, 5], values=[1.0, 2.0])
    write(directory / "2026-02-01_2026-03-01.parquet", [10, 15], values=[3.0, 4.0])
    source = alphax.data.Source(tmp_path / "bars", schema(), KEY)
    assert [one[1] for one in rows(source.load())] == [[0, 5], [10, 15]]
    # [start, end) on the decoded availability column, never on a file name. A
    # file that contributes no rows still reports its own batch.
    assert [one[1] for one in rows(source.load(start=5, end=15))] == [[5], [10]]
    assert [one[1] for one in rows(source.load(start=10))] == [[], [10, 15]]
    assert source.load(end=5).prepare().time.tolist() == [0]
