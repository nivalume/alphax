"""Crypto strategy research: economic hypotheses, the strategies that test them, coded and backtested with alphax.

The data is a jfinance store (``~/.jfinance``); strategies run and are backtested in an interpreter with the
alphax engine. There is no baseline.
"""

from alphax_agent.scenarios.crypto_strategy.scenario import CryptoStrategyScenario
from alphax_agent.scenarios.crypto_strategy.settings import CryptoStrategySettings

__all__ = ["CryptoStrategyScenario", "CryptoStrategySettings"]
