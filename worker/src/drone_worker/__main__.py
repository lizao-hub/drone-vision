"""Entry point: ``python -m drone_worker [batch|pipeline|relay] [args]``.

The three entries share one flag vocabulary (``--source``, ``--device``,
``--stride``, ``--log-every``, the model overrides) and differ only in topology:

* ``batch``    -- one device, file in, JSONL out (mp4 with ``--save-video``);
* ``pipeline`` -- ``--device 0,1`` splits segmentation and detection;
* ``relay``    -- live in, annotated RTMP out.

No subcommand means ``batch``; ``batch`` spelled out is still accepted.
"""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["pipeline"]:
        from drone_worker.pipeline import main_pipeline

        return main_pipeline(arguments[1:])
    if arguments[:1] == ["relay"]:
        from drone_worker.relay import main_relay

        return main_relay(arguments[1:])
    if arguments[:1] == ["batch"]:
        arguments = arguments[1:]
    from drone_worker.batch import main_batch

    return main_batch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
