from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from support import ScriptedBackend, SyntheticMarket

from alphax_agent.config.settings import LLMSettings
from alphax_agent.llm.client import LLMClient


@pytest.fixture
def backend() -> ScriptedBackend:
    return ScriptedBackend()


@pytest.fixture
def llm(backend: ScriptedBackend) -> LLMClient:
    return LLMClient(backend, LLMSettings(model="scripted", max_retries=0), sleeper=lambda _: None)


@pytest.fixture(scope="session")
def market() -> SyntheticMarket:
    return SyntheticMarket()


@pytest.fixture(scope="session")
def catalog_root(tmp_path_factory: pytest.TempPathFactory, market: SyntheticMarket) -> Path:
    return market.write_catalog(tmp_path_factory.mktemp("catalog"))
