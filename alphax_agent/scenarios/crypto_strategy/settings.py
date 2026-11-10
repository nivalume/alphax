"""Options of the crypto strategy scenario."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class CryptoStrategySettings(BaseModel):
    data_root_dir: Path = Field(
        default=Path("~/jarvis-data"),
        description="Where the data lives: the schemas under schemas/, the rows under crypto/",
    )
    max_strategies_per_experiment: int = Field(
        default=2, ge=1, description="One mechanism and at most this many variants of it per experiment"
    )

    @property
    def data_path(self) -> Path:
        """``data_root_dir`` with ``$VAR`` and ``~`` expanded (a literal ``~`` would point below the cwd)."""
        return Path(os.path.expandvars(str(self.data_root_dir))).expanduser().resolve()
