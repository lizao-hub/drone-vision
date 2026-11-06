"""Worker entry point: dispatch to the offline batch runner or the live relay.

``python -m drone_worker [args]``              offline batch (default)
``python -m drone_worker batch [args]``        the same, spelled out
``python -m drone_worker relay --source ...``  live relay: RTSP in, RTMP out

Both runners drive the same multi-process, multi-GPU engine; they differ only in
how they pace frames.
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else ""
    rest = arguments[1:]

    if command == "relay":
        from drone_worker.relay import main_relay

        return main_relay(rest)
    if command == "batch":
        from drone_worker.batch import main_batch

        return main_batch(rest)

    from drone_worker.batch import main_batch

    return main_batch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
