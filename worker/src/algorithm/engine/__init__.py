"""The execution shell: three model processes and the parent that drives them.

One frame enters, one :class:`~algorithm.domain.observations.Observation` leaves::

    packet    FramePacket: one frame plus the metadata the engine needs
    shared    the shared-memory ring and the messages that cross process edges
    workers   one model per process, one GPU each (segment, vehicle, line)
    pipeline  the parent-side engine: submit / result / run, ordered assembly

The parent loads no model and never touches CUDA, which is what lets it fork;
each worker pins ``CUDA_VISIBLE_DEVICES`` before its first ``import torch``.

Nothing here decides what a frame *means*.  The parent decodes the workers'
messages into a :class:`~algorithm.perception.assembly.StageResult` and hands it
to :class:`~algorithm.perception.assembly.ObservationAssembler`, which owns the
trackers and the validity policy.
"""
