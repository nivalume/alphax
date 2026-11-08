"""The price-volume data that factor code reads, and its export from qlib."""

from __future__ import annotations

import json
import threading
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment as JinjaEnvironment
from jinja2 import StrictUndefined

from alphax.core.exceptions import EnvironmentUnavailableError
from alphax.env.qlib import QlibEnvironment
from alphax.scenarios.qlib_factor.settings import QlibFactorSettings


class FactorDataCatalog:
    """Three copies of the same panel, each in ``<root>/<variant>/daily_pv.h5``:

    * ``full`` - the whole market and period, used for the backtest,
    * ``debug`` - a few instruments over a short window, used while coding,
    * ``debug_cut`` - ``debug`` truncated at ``cut_date``, used by the look-ahead check.
    """

    FILE_NAME = "daily_pv.h5"
    KEY = "data"
    VARIANTS = ("full", "debug", "debug_cut")

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._frames: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def path(self, variant: str) -> Path:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown data variant {variant!r}; known: {self.VARIANTS}")
        return self._root / variant / self.FILE_NAME

    def ready(self) -> bool:
        return all(self.path(variant).exists() for variant in self.VARIANTS) and self._meta_path.exists()

    @property
    def meta(self) -> dict[str, Any]:
        if not self._meta_path.exists():
            raise EnvironmentUnavailableError(f"Factor data in {self._root} has not been exported yet")
        return json.loads(self._meta_path.read_text(encoding="utf-8"))

    @property
    def cut_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.meta["cut_date"])

    def frame(self, variant: str = "debug") -> pd.DataFrame:
        with self._lock:
            if variant not in self._frames:
                self._frames[variant] = pd.read_hdf(self.path(variant), key=self.KEY)
            return self._frames[variant]

    def describe(self) -> str:
        """Description of the data shown to every agent."""
        meta = self.meta
        sample = self.frame("debug")
        columns = "\n".join(f"  - {column}: {dtype}" for column, dtype in sample.dtypes.items())
        return (
            f"File `{self.FILE_NAME}` in the working directory, HDF5 key `{self.KEY}`; load it with "
            f'`pd.read_hdf("{self.FILE_NAME}", key="{self.KEY}")`.\n'
            "A pandas DataFrame of daily bars with a MultiIndex (`datetime`, `instrument`), sorted, "
            "one row per trading day and instrument.\n"
            f"Columns (qlib field names, keep the `$`):\n{columns}\n"
            "Prices are adjusted for splits and dividends; `$factor` is the adjustment factor "
            "(raw price = adjusted price / $factor). `$volume` is the adjusted traded volume.\n"
            f"During development the file holds {meta['debug_instruments']} instruments from {meta['debug_start']} "
            f"to {meta['debug_end']}; in the backtest it holds the whole {meta['market']} universe "
            f"({meta['full_instruments']} instruments, {meta['full_start']} to {meta['full_end']}). "
            "Code must work for both without change.\n"
            f"First rows:\n{sample.head(5).to_string()}"
        )

    def profile(self, column: str | None = None, instrument: str | None = None) -> str:
        """Summary statistics of the debug sample, for the data inspection tool."""
        frame = self.frame("debug")
        if instrument is not None:
            if instrument not in frame.index.get_level_values("instrument"):
                return f"Instrument {instrument!r} is not in the development sample."
            frame = frame.xs(instrument, level="instrument", drop_level=False)
        if column is not None:
            if column not in frame.columns:
                return f"Column {column!r} does not exist; columns: {list(frame.columns)}"
            frame = frame[[column]]
        stats = frame.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
        stats["nan_ratio"] = frame.isna().mean()
        dates = frame.index.get_level_values("datetime")
        return (
            f"rows={len(frame)}, instruments={frame.index.get_level_values('instrument').nunique()}, "
            f"dates={dates.min().date()}..{dates.max().date()}\n{stats.to_string()}\n\n"
            f"Last rows:\n{frame.tail(5).to_string()}"
        )

    @property
    def _meta_path(self) -> Path:
        return self._root / "meta.json"


class FactorDataExporter:
    """Writes the catalog files from the qlib data provider, inside the qlib environment."""

    SCRIPT = "export_data.py"

    def __init__(self, environment: QlibEnvironment, settings: QlibFactorSettings, catalog: FactorDataCatalog) -> None:
        self._env = environment
        self._settings = settings
        self._catalog = catalog
        self._templates = JinjaEnvironment(undefined=StrictUndefined, keep_trailing_newline=True)

    def export(self, *, force: bool = False) -> None:
        if self._catalog.ready() and not force:
            return
        root = self._catalog.root
        root.mkdir(parents=True, exist_ok=True)
        (root / self.SCRIPT).write_text(self._script(), encoding="utf-8")
        result = self._env.run_python(self.SCRIPT, root, timeout=self._env.settings.timeout)
        if not result.ok or not self._catalog.ready():
            raise EnvironmentUnavailableError(f"Exporting factor data from qlib failed:\n{result.tail()}")

    def _script(self) -> str:
        settings = self._settings
        source = resources.files("alphax.scenarios.qlib_factor").joinpath("templates/export_data.py.j2")
        return self._templates.from_string(source.read_text(encoding="utf-8")).render(
            provider_uri=str(settings.qlib.provider_path),
            region=settings.qlib.region,
            market=settings.market,
            fields=settings.fields,
            start=settings.data_start,
            end=settings.end_date,
            debug_start=settings.debug.start,
            debug_end=settings.debug.end,
            debug_instruments=settings.debug.instruments,
            cut_fraction=settings.debug.cut_fraction,
            file_name=FactorDataCatalog.FILE_NAME,
            key=FactorDataCatalog.KEY,
        )
