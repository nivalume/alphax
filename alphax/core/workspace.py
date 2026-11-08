"""File-based workspaces holding the code of an implementation."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from alphax.core.task import Task


class Workspace(BaseModel):
    """A directory plus the source files that define its content.

    ``files`` is the source of truth: the directory can be deleted and rebuilt from it at any
    time, which keeps experiments reproducible and lets a workspace be forked cheaply.
    """

    path: Path
    files: dict[str, str] = Field(default_factory=dict)

    def resolve(self, name: str) -> Path:
        """Return the absolute path of ``name`` and refuse paths that leave the workspace."""
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Workspace file names must be relative and stay inside the workspace: {name!r}")
        return self.path / relative

    def inject(self, files: Mapping[str, str]) -> None:
        """Add or replace source files and write them to disk."""
        for name, content in files.items():
            target = self.resolve(name)
            self.files[name] = content
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def materialize(self) -> Path:
        """Create the directory and write every source file into it."""
        self.path.mkdir(parents=True, exist_ok=True)
        self.inject(dict(self.files))
        self.prepare()
        return self.path

    def prepare(self) -> None:
        """Hook for subclasses that need data or assets next to the code."""

    def link(self, source: Path, name: str | None = None) -> Path:
        """Symlink an external file (typically read-only data) into the workspace."""
        target = self.resolve(name or source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(Path(source).resolve())
        return target

    def read(self, name: str) -> str | None:
        return self.files.get(name)

    def fork(self, path: Path) -> Self:
        """Copy of this workspace rooted at another directory; nothing is written yet."""
        return self.model_copy(update={"path": Path(path), "files": dict(self.files)}, deep=True)

    def clear(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def render_code(self, suffixes: Iterable[str] = (".py",)) -> str:
        """All source files with the given suffixes, formatted for a prompt."""
        wanted = tuple(suffixes)
        blocks = [
            f"File: {name}\n```\n{content}\n```"
            for name, content in sorted(self.files.items())
            if name.endswith(wanted)
        ]
        return "\n\n".join(blocks) if blocks else "(no source files)"


class WorkspaceFactory(ABC):
    """Creates the workspace in which one implementation of a task lives."""

    @abstractmethod
    def create(self, task: Task, path: Path) -> Workspace: ...
