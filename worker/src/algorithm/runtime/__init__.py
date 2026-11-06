"""The execution shell around the algorithm: where frames come from, and wiring.

Two kinds of thing live here, both outside ``perception``:

* I/O adapters -- :mod:`algorithm.runtime.sources`, :mod:`algorithm.runtime.live`,
  :mod:`algorithm.runtime.rtmp_sink`, :mod:`algorithm.runtime.snapshots`.
* The composition root, :mod:`algorithm.runtime.factory`: settings in, wired
  engine out, so no entry point repeats the wiring.

The engine lives in :mod:`algorithm.engine`; this package produces the
:class:`~algorithm.engine.packet.FramePacket` it consumes and builds it, so the
dependency runs ``runtime -> engine -> perception``.
"""
