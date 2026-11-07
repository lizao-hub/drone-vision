"""Entry point: ``python -m drone_worker [batch] [args]``.

``batch`` is the only mode left, so the prefix is accepted and dropped; it stays
supported because existing commands spell it out.
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["batch"]:
        arguments = arguments[1:]
    from drone_worker.batch import main_batch

    return main_batch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
