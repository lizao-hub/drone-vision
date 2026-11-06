"""The layering, asserted: which module is allowed to import which.

Each check runs in a fresh interpreter, because "module X is not in
``sys.modules``" only means anything before something else imported it.  These
are the arrows the layout establishes, so they are worth a test rather than a
comment:

    domain     -/->  perception, engine, runtime     (the vocabulary is the bottom)
    perception -/->  engine                          (adapters never import an engine)
    perception.assembly -/-> engine                  (the policy is not the orchestration)
    engine     -/->  runtime                         (the engine imports no I/O, no root)
    engine.pipeline -/-> torch                       (the parent stays CUDA-free so it forks)

The reverse arrows are the real ones: ``engine`` imports ``perception`` (workers
drive the adapters; the parent hands a ``StageResult`` to the assembler) and
``runtime`` imports ``engine`` (the sources produce ``FramePacket``; the factory
builds the engine).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _fresh(code: str) -> str:
    """Run ``code`` in a new interpreter with ``src`` importable."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _pulled(*prefixes: str) -> str:
    """``True``/``False``: does any loaded module start with one of ``prefixes``?"""
    return f"print(any(n.startswith({prefixes!r}) for n in sys.modules))\n"


def test_the_engine_does_not_import_torch_or_the_composition_root():
    # The parent must stay CUDA-free so it can fork; importing the factory would
    # pull in the tracker stack and therefore torch.
    assert _fresh(
        "import sys\n"
        "import algorithm.engine.pipeline\n"
        "print('algorithm.runtime.factory' in sys.modules)\n"
        "print('torch' in sys.modules)\n"
    ).split() == ["False", "False"]


def test_the_engine_does_not_import_the_runtime():
    # Frame I/O and the composition root sit above the engine, not below it.
    assert _fresh(
        "import sys\n"
        "import algorithm.engine.pipeline\n" + _pulled("algorithm.runtime")
    ) == "False"


def test_the_assembly_policy_does_not_import_the_engine():
    # The merge-and-decide step is perception; it can be built and exercised
    # without spawning anything.
    assert _fresh(
        "import sys\n"
        "import algorithm.perception.assembly\n" + _pulled("algorithm.engine")
    ) == "False"


def test_perception_imports_no_engine():
    # The model adapters, the masking and the deck tracker sit below the engine.
    assert _fresh(
        "import sys\n"
        "import algorithm.perception.masking\n"
        "import algorithm.perception.viaduct_tracker\n" + _pulled("algorithm.engine")
    ) == "False"


def test_the_domain_imports_nothing_above_it():
    # ``domain`` is the vocabulary every layer speaks; it may use numpy, but it
    # may not reach for an adapter, an engine or a frame source.
    assert _fresh(
        "import sys\n"
        "import algorithm.domain.observations\n"
        + _pulled(
            "algorithm.perception",
            "algorithm.engine",
            "algorithm.runtime",
            "torch",
            "cv2",
        )
    ) == "False"
