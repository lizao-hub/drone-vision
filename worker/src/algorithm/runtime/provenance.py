"""Which code produced a run: one revision string for headers and logs.

A batch log that says "frames=854 fps=20.4" is only useful next to the revision
that produced it; after a refactor the same numbers mean something different.
``git rev-parse`` is used when the checkout is a repository (with a ``-dirty``
suffix for uncommitted changes), otherwise the sources are hashed so a plain
copy of the tree still identifies itself.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# .../worker/src/algorithm/runtime/provenance.py -> worker/
_WORKER_ROOT = Path(__file__).resolve().parents[3]


def code_revision(root: Path | None = None) -> str:
    """Short Git revision, or ``source-<hash>``, or ``unknown``."""
    root = (root or _WORKER_ROOT).resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        revision = completed.stdout.strip()
        if revision:
            return f"{revision}-dirty" if _is_dirty(root) else revision
    except (OSError, subprocess.SubprocessError):
        pass
    digest = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py"))
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return f"source-{digest.hexdigest()[:12]}" if files else "unknown"


def _is_dirty(root: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(completed.stdout.strip())
