"""Entry point: ``python -m drone_worker [batch|pipeline|relay] [args]``.

No subcommand means ``batch``, which is also still accepted spelled out because
existing commands do.  ``relay`` is the live path: RTSP in, annotated RTMP out,
over the same serial pipeline.
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["pipeline"]:
        from drone_worker.parallel import main_parallel

        return main_parallel(arguments[1:])
    if arguments[:1] == ["relay"]:
        from drone_worker.relay import main_relay

        return main_relay(arguments[1:])
    if arguments[:1] == ["batch"]:
        arguments = arguments[1:]
    from drone_worker.batch import main_batch

    return main_batch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
