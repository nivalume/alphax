"""The data of the scenario: the jfinance datasets the agents are told about, and the panels strategies run on.

The jfinance pipeline writes one JSON schema per dataset (``binance.vision.market.crypto.bar.raw.v1.json``)
and the dataset's rows as Parquet files, one directory per stream (instrument and bar interval):
``<data>/<market>/<slug>.v<version>/<INSTRUMENT>-1-DAY-LAST-EXTERNAL/*.parquet``. The schema files are the
fact; this module reads and renders them, and exports the configured datasets into ``(bars, instruments)``
panels that the strategy code and the backtest read.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment as JinjaEnvironment
from jinja2 import StrictUndefined

from alphax_agent.core.exceptions import EnvironmentUnavailableError
from alphax_agent.env.alphax import AlphaXEnvironment, AlphaXEnvSettings

logger = logging.getLogger(__name__)

NO_DATA_SCHEMA = "No data schema is available; the data directory or the schema list may not be configured."

#: Columns every bar dataset has that are not values: the stream and its clock.
_KEYS = frozenset({"instrument_id", "bar_type", "ts_open", "ts_close", "ts_event", "ts_init"})
_OHLCV = ("open", "high", "low", "close", "volume")
_NUMERIC = ("decimal", "float32", "float64", "int8", "int16", "int32", "int64")


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str
    spec: Mapping[str, Any]
    """The column as the schema file declares it, for the export script."""

    @property
    def numeric(self) -> bool:
        return str(self.spec.get("type")) in _NUMERIC


@dataclass(frozen=True)
class DataSchema:
    slug: str
    version: int
    name: str
    description: str
    columns: tuple[Column, ...]
    stream_field: str
    """The field a stream is keyed by: ``bar_type`` for bars, ``instrument_id`` for events."""

    @property
    def directory(self) -> str:
        """The dataset directory, relative to the market directory."""
        return f"{self.slug}.v{self.version}"

    @property
    def is_bar(self) -> bool:
        names = {column.name for column in self.columns}
        return self.stream_field == "bar_type" and {"ts_open", "ts_close", *_OHLCV} <= names

    @property
    def values(self) -> tuple[Column, ...]:
        """The numeric columns a strategy can read."""
        return tuple(column for column in self.columns if column.name not in _KEYS and column.numeric)


def load_schemas(schema_dir: Path) -> dict[str, DataSchema]:
    """The latest version of each schema in schema_dir, by slug.

    A file that does not parse or lacks a slug, a version or columns is skipped with a warning: the
    data pipeline may be writing the directory while the scenario renders.
    """
    found: dict[str, DataSchema] = {}
    if not schema_dir.is_dir():
        return found
    for path in sorted(schema_dir.glob("*.json")):
        try:
            schema = _parse(path)
        except (OSError, KeyError, TypeError, ValueError) as error:
            logger.warning("Skipping the data schema %s: %s", path, error)
            continue
        if schema.slug not in found or schema.version > found[schema.slug].version:
            found[schema.slug] = schema
    return found


class DataLayout:
    """Where the rows of each dataset are, for one interval."""

    def __init__(self, settings: AlphaXEnvSettings) -> None:
        self._settings = settings

    def dataset_dir(self, schema: DataSchema) -> Path:
        return self._settings.market_path / schema.directory

    def instruments(self, schema: DataSchema) -> list[str]:
        """The instruments with a stream of the configured interval, sorted."""
        root = self.dataset_dir(schema)
        if not root.is_dir():
            return []
        suffix = f"-{self._settings.stream_suffix}"
        return sorted(
            path.name[: -len(suffix)] for path in root.iterdir() if path.is_dir() and path.name.endswith(suffix)
        )

    def has_rows(self, schema: DataSchema) -> bool:
        return bool(self.instruments(schema))


@dataclass(frozen=True)
class DatasetSelection:
    """The configured datasets that can be used, and why the others cannot."""

    schemas: tuple[DataSchema, ...]
    problems: tuple[str, ...]


def select_datasets(settings: AlphaXEnvSettings) -> DatasetSelection:
    """The datasets of ``settings.schemas`` that are bar datasets with rows on disk, in the configured order."""
    available = load_schemas(settings.schema_path)
    layout = DataLayout(settings)
    chosen: list[DataSchema] = []
    problems: list[str] = []
    for slug in settings.schemas:
        schema = available.get(slug)
        if schema is None:
            problems.append(f"{slug}: no schema in {settings.schema_path}")
        elif not schema.is_bar:
            problems.append(f"{slug}: not a bar dataset (only bar datasets can be put into a panel yet)")
        elif not layout.has_rows(schema):
            problems.append(f"{slug}: no {settings.interval} stream under {layout.dataset_dir(schema)}")
        else:
            chosen.append(schema)
    return DatasetSelection(schemas=tuple(chosen), problems=tuple(problems))


def usable_datasets(settings: AlphaXEnvSettings) -> list[str]:
    """Slugs of every bar dataset with rows on disk, to suggest when a configured one is not usable."""
    layout = DataLayout(settings)
    return [
        slug
        for slug, schema in sorted(load_schemas(settings.schema_path).items())
        if schema.is_bar and layout.has_rows(schema)
    ]


def render_data_schemas(settings: AlphaXEnvSettings) -> str:
    """The prompt section of every configured dataset that can be used, one block each."""
    if not settings.schema_path.is_dir():
        # The prompt falls back to "no data schema is available", which reads like a missing setting;
        # say which directory was looked at so the two causes are distinguishable.
        logger.warning("No data schemas under %s: the prompt will not describe any dataset.", settings.schema_path)
    selection = select_datasets(settings)
    for problem in selection.problems:
        logger.warning("Leaving a dataset out of the prompt: %s", problem)
    return "\n\n".join(_render(schema, settings.interval) for schema in selection.schemas)


def _render(schema: DataSchema, interval: str) -> str:
    lines = [f"### `{schema.slug}` ({schema.name}, {interval} bars)", ""]
    if schema.description:
        lines += [schema.description, ""]
    lines += ["| Column | Type | Description |", "|---|---|---|"]
    lines += [f"| `{column.name}` | {column.type} | {column.description} |" for column in schema.values]
    return "\n".join(lines)


def _parse(path: Path) -> DataSchema:
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = tuple(
        Column(
            name=str(column["name"]),
            type=_type_of(column),
            description=_described(payload, str(column["name"])),
            spec=dict(column),
        )
        for column in payload["columns"]
    )
    return DataSchema(
        slug=str(payload["slug"]),
        version=int(payload["version"]),
        name=str(payload.get("name") or payload["slug"]),
        description=str(payload.get("description") or ""),
        columns=columns,
        stream_field=_stream_field(payload),
    )


def _stream_field(payload: Mapping[str, Any]) -> str:
    fields = payload.get("stream_fields")
    required = fields.get("required") if isinstance(fields, Mapping) else None
    return str(required[0]) if isinstance(required, list) and required else ""


def _described(payload: Mapping[str, object], name: str) -> str:
    annotations = payload.get("annotations") or {}
    fields = annotations.get("fields") if isinstance(annotations, Mapping) else None
    annotation = fields.get(name) if isinstance(fields, Mapping) else None
    if isinstance(annotation, Mapping):
        return str(annotation.get("description") or "")
    return ""


def _type_of(column: Mapping[str, object]) -> str:
    if column.get("type") == "decimal" and column.get("precision") is not None:
        return f"decimal({column['precision']},{column.get('scale', 0)})"
    return str(column.get("type") or "")


class StrategyDataCatalog:
    """Three panels of the same data, each in ``<root>/<variant>/panel.npz``:

    * ``full`` - every instrument of the universe over the warm-up and the whole period, for the backtest,
    * ``debug`` - a few instruments over the last in-sample bars, for coding,
    * ``debug_cut`` - ``debug`` truncated at ``cut_index``, for the look-ahead check.

    A panel holds ``time`` (the open of each bar, int64 UTC nanoseconds), ``symbols`` and one
    ``(bars, symbols)`` float64 array per field; ``meta.json`` says what was exported.
    """

    FILE_NAME = "panel.npz"
    VARIANTS = ("full", "debug", "debug_cut")
    RESERVED = ("time", "symbols")

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._panels: dict[str, dict[str, np.ndarray]] = {}
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def path(self, variant: str) -> Path:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown data variant {variant!r}; known: {self.VARIANTS}")
        return self._root / variant / self.FILE_NAME

    @property
    def meta_path(self) -> Path:
        return self._root / "meta.json"

    def ready(self) -> bool:
        return all(self.path(variant).exists() for variant in self.VARIANTS) and self.meta_path.exists()

    @property
    def meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            raise EnvironmentUnavailableError(f"Strategy data in {self._root} has not been exported yet")
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    @property
    def cut_index(self) -> int:
        return int(self.meta["cut_index"])

    def panel(self, variant: str = "debug") -> dict[str, np.ndarray]:
        with self._lock:
            if variant not in self._panels:
                with np.load(self.path(variant), allow_pickle=False) as npz:
                    self._panels[variant] = {name: npz[name] for name in npz.files}
            return self._panels[variant]

    def fields(self, variant: str = "debug") -> dict[str, np.ndarray]:
        return {name: values for name, values in self.panel(variant).items() if name not in self.RESERVED}

    def shape(self, variant: str = "debug") -> tuple[int, int]:
        rows, columns = self.panel(variant)["close"].shape
        return int(rows), int(columns)

    def describe(self) -> str:
        """Description of the panel shown to the agents that write and review code."""
        meta = self.meta
        fields = self.fields("debug")
        names = ", ".join(f"`{name}`" for name in fields)
        aliases = "".join(
            f"\n- `{alias}` is the `{column}` column." for alias, column in meta.get("aliases", {}).items()
        )
        debug = meta["debug"]
        full = meta["full"]
        return (
            f"`data` is a dict of NumPy float64 arrays of shape (bars, instruments), one per field: {names}.{aliases}\n"
            f"Row t is the {meta['interval']} bar that opens at `time[t]` (UTC) and closes one bar later; its values "
            "are known at its close. Rows are consecutive bars, oldest first; columns are instruments, always in the "
            "same order. NaN marks a bar an instrument did not trade (not yet listed, delisted, or a gap).\n"
            f"During development the panel holds {len(debug['symbols'])} instruments over {debug['bars']} bars "
            f"({debug['start']} to {debug['end']}); in the backtest it holds {len(full['symbols'])} instruments over "
            f"{full['bars']} bars ({full['start']} to {full['end']}), the first {full['warmup_bars']} of which only "
            "fill the windows. Code must work for both without change and must not depend on the number of rows "
            "or columns.\n"
            f"Last rows of the development panel (first instrument):\n{self._head(fields)}"
        )

    def profile(self, field: str | None = None, instrument: str | None = None) -> str:
        """Summary statistics of the development panel, for the data inspection tool."""
        panel = self.panel("debug")
        symbols = [str(symbol) for symbol in panel["symbols"]]
        fields = self.fields("debug")
        if field is not None and field not in fields:
            return f"Field {field!r} does not exist; fields: {list(fields)}"
        if instrument is not None and instrument not in symbols:
            return f"Instrument {instrument!r} is not in the development panel; instruments: {symbols}"
        columns = [symbols.index(instrument)] if instrument is not None else list(range(len(symbols)))
        lines = [
            f"bars={len(panel['time'])}, instruments={len(columns)} "
            f"({', '.join(symbols[column] for column in columns)}), "
            f"{_day(int(panel['time'][0]))}..{_day(int(panel['time'][-1]))}",
            "field | mean | std | p01 | p50 | p99 | nan_ratio",
        ]
        for name, values in fields.items():
            if field is not None and name != field:
                continue
            block = values[:, columns]
            finite = block[np.isfinite(block)]
            if finite.size:
                p01, p50, p99 = np.quantile(finite, [0.01, 0.5, 0.99])
                stats = f"{finite.mean():.6g} | {finite.std():.6g} | {p01:.6g} | {p50:.6g} | {p99:.6g}"
            else:
                stats = "n/a | n/a | n/a | n/a | n/a"
            lines.append(f"{name} | {stats} | {1 - finite.size / max(block.size, 1):.3f}")
        return "\n".join(lines)

    def _head(self, fields: Mapping[str, np.ndarray]) -> str:
        time = self.panel("debug")["time"]
        rows = range(max(0, len(time) - 3), len(time))
        header = "time | " + " | ".join(fields)
        body = [
            f"{_day(int(time[row]))} | " + " | ".join(f"{values[row, 0]:.6g}" for values in fields.values())
            for row in rows
        ]
        return "\n".join([header, *body])


def _day(nanoseconds: int) -> str:
    return str(np.datetime64(nanoseconds, "ns").astype("datetime64[D]"))


class StrategyDataExporter:
    """Writes the catalog's panels from the jfinance store, inside the alphax environment."""

    SCRIPT = "export_panel.py"
    CONFIG = "export_config.json"
    TEMPLATE = "templates/export_panel.py.j2"

    def __init__(
        self,
        environment: AlphaXEnvironment,
        settings: Any,
        catalog: StrategyDataCatalog,
    ) -> None:
        # settings is a CryptoStrategySettings; not imported here to keep data.py free of the scenario's options.
        self._env = environment
        self._settings = settings
        self._catalog = catalog
        self._templates = JinjaEnvironment(undefined=StrictUndefined, keep_trailing_newline=True)

    def export(self, *, force: bool = False) -> None:
        config = self.config()
        if self._catalog.ready() and not force and self._catalog.meta.get("config") == config:
            return
        root = self._catalog.root
        root.mkdir(parents=True, exist_ok=True)
        (root / self.CONFIG).write_text(json.dumps(config, indent=2), encoding="utf-8")
        (root / self.SCRIPT).write_text(self._script(), encoding="utf-8")
        result = self._env.run_python(self.SCRIPT, root, args=[self.CONFIG], timeout=self._env.settings.timeout)
        if not result.ok or not self._catalog.ready():
            raise EnvironmentUnavailableError(f"Exporting the strategy panels failed:\n{result.tail()}")

    def config(self) -> dict[str, Any]:
        """Everything the export depends on; a change in it exports again."""
        settings = self._settings
        env: AlphaXEnvSettings = settings.alphax
        selection = select_datasets(env)
        if selection.problems or not selection.schemas:
            usable = usable_datasets(env)
            raise EnvironmentUnavailableError(
                "These configured datasets cannot be used:\n"
                + "\n".join(f"- {problem}" for problem in selection.problems or ["(no dataset configured)"])
                + f"\nBar datasets with {env.interval} rows on disk: {', '.join(usable) or 'none'}"
            )
        primary = selection.schemas[0]
        period = settings.period
        return {
            "datasets": [
                {
                    "slug": schema.slug,
                    "root": str(DataLayout(env).dataset_dir(schema)),
                    "columns": [dict(column.spec) for column in schema.columns],
                    "values": [column.name for column in schema.values],
                }
                for schema in selection.schemas
            ],
            "interval": env.interval,
            "suffix": env.stream_suffix,
            "bar_ns": env.bar_nanoseconds,
            "start": period.start.isoformat(),
            "oos_start": period.oos_start.isoformat(),
            "end": period.end.isoformat(),
            "warmup_bars": settings.backtest.warmup_bars,
            "symbols": settings.universe.symbols,
            "max_instruments": settings.universe.max_instruments,
            "min_bars": settings.universe.min_bars,
            "debug_instruments": settings.debug.instruments,
            "debug_bars": settings.debug.bars,
            "cut_fraction": settings.debug.cut_fraction,
            "aliases": {"amount": "quote_volume"} if "quote_volume" in {c.name for c in primary.values} else {},
        }

    def _script(self) -> str:
        source = resources.files("alphax_agent.scenarios.crypto_strategy").joinpath(self.TEMPLATE)
        return self._templates.from_string(source.read_text(encoding="utf-8")).render(
            file_name=StrategyDataCatalog.FILE_NAME, variants=list(StrategyDataCatalog.VARIANTS)
        )


def describe_symbols(symbols: Sequence[str], limit: int = 12) -> str:
    shown = ", ".join(symbols[:limit])
    return shown + (f" and {len(symbols) - limit} more" if len(symbols) > limit else "")
