"""Built-in scenarios. Importing this package registers them with the scenario registry."""

from alphax_agent.scenarios.crypto_strategy import CryptoStrategyScenario
from alphax_agent.scenarios.qlib_factor import QlibFactorScenario

__all__ = ["CryptoStrategyScenario", "QlibFactorScenario"]
