"""The research roles this scenario does not implement yet.

Only the hypothesis and the experiment design exist. The environment, the coder, the runner and the
summarizer are left empty: each slot below fails with a clear message when the workflow reaches it, so a run
stops at the first missing step instead of producing something that looks like a result. The runner does not
override ``establish_baseline``, so the scenario has no baseline either.

Stop a run after the design with ``alphax-agent run --until design``.
"""

from __future__ import annotations

from alphax_agent.core.exceptions import ConfigurationError
from alphax_agent.core.experiment import Experiment
from alphax_agent.core.feedback import ExperimentFeedback
from alphax_agent.core.roles import Developer, Runner, Summarizer
from alphax_agent.core.trace import Trace


def _missing(role: str) -> ConfigurationError:
    return ConfigurationError(
        f"The crypto_strategy scenario has no {role} yet; stop the run after the design stage with --until design"
    )


class PendingDeveloper(Developer):
    def develop(self, experiment: Experiment, trace: Trace) -> Experiment:
        raise _missing("developer")


class PendingRunner(Runner):
    def run(self, experiment: Experiment, trace: Trace) -> Experiment:
        raise _missing("runner")


class PendingSummarizer(Summarizer):
    def summarize(self, experiment: Experiment, trace: Trace) -> ExperimentFeedback:
        raise _missing("summarizer")
