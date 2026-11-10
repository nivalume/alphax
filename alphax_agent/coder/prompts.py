"""The generic prompts of the coder and of the reflection agent.

They describe no domain. A scenario subclasses them (or copies their template keys and inputs) to add
the sections its code depends on, and overrides ``coder.system`` / ``reflector.system`` in its own
prompt file to place them.
"""

from __future__ import annotations

from alphax_agent.agents.prompt import AgentPrompt


class CoderPrompt(AgentPrompt):
    base_package = "alphax_agent.coder"
    system_template = "coder.system"
    user_template = "coder.user"
    inputs = (
        "task",
        "operator",
        "required_files",
        "parent_code",
        "parent_feedback",
        "lineage",
        "lessons",
        "peers",
        "tools",
    )


class ReflectorPrompt(AgentPrompt):
    base_package = "alphax_agent.coder"
    system_template = "reflector.system"
    user_template = "reflector.user"
    inputs = ("transitions", "lessons", "max_lessons")
