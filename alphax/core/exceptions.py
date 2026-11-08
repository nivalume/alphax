"""Exception hierarchy.

Every failure the platform knows how to recover from derives from :class:`AlphaXError`.
The workflow catches these per research step, records them in the trace and moves on
to the next loop; anything else is treated as a defect.
"""

from __future__ import annotations


class AlphaXError(Exception):
    """Base class of all recoverable platform errors."""


class ConfigurationError(AlphaXError):
    """The settings or the plugin wiring are invalid."""


class LLMError(AlphaXError):
    """The language model call failed permanently."""


class LLMTransientError(LLMError):
    """The language model call failed in a way that is worth retrying."""


class OutputParsingError(AlphaXError):
    """A model reply could not be parsed into the expected structure."""


class AgentError(AlphaXError):
    """An agent could not produce a valid answer within its budget."""


class EnvironmentUnavailableError(AlphaXError):
    """The execution environment is missing a dependency or data."""


class ExecutionError(AlphaXError):
    """A command in the execution environment failed."""


class ProposalError(AlphaXError):
    """No usable hypothesis or experiment design was produced."""


class CoderError(AlphaXError):
    """No task of the experiment could be implemented."""


class RunnerError(AlphaXError):
    """The experiment could not be run or produced no result."""
