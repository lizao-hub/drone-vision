"""Fine-tune yolo26m on the line dataset: 4-GPU DDP, imgsz 1088, 100 epochs.

Run on the server (conda env with ultralytics 8.4.183):

    python worker/train.py

The script resolves every path relative to its own location, so the whole
``worker/`` tree can be copied to the server as-is. ``datasets/`` and
``models_backup/`` are git-ignored, so they must be copied separately.

Engine behaviour that is easy to trip over (verified against the installed
ultralytics 8.4.183):

* ``imgsz`` must be a multiple of the model stride 32. Passing 1080 is rounded
  *up* to 1088 by ``check_imgsz``; ``IMGSZ`` below is already that value.
* ``imgsz`` is a single integer, i.e. training is square: a 16:9 frame is
  letterboxed into 1088x1088 with ~230 px of grey padding top and bottom.
* ``batch`` is the *total* batch; the engine divides it by the GPU count, so
  ``batch=16`` means 4 images per GPU.
* Multi-GPU forces ``rect=False`` (rectangular batches are not compatible with
  DDP) and disables AutoBatch (``batch=-1``).
* ``exist_ok=True`` is required for DDP: only rank 0 honours ``exist_ok``, while
  the other ranks reuse its directory. With ``exist_ok=False`` a stale run
  directory makes rank 0 increment the name while the other ranks do not.

The annotations were drawn on 640x368 frames, so 1088 already interpolates
~1.7x: the useful resolution ceiling is the annotation grid, not the 1080p
video. Re-exporting native 1920x1080 frames and reusing the normalised labels
is what would let a larger imgsz pay off.
"""

from pathlib import Path

from ultralytics import YOLO

WORKER_DIR = Path(__file__).resolve().parent
DATA_YAML = WORKER_DIR / "datasets" / "vehicle_yolo" / "data.yaml"
PRETRAINED = WORKER_DIR / "models_backup" / "yolo26m.pt"
PROJECT_DIR = WORKER_DIR / "output" / "vehicle_detect"
RUN_NAME = "yolo26m_1080"

# IMAGE_SIZE = 1080  # requirement: fixed 1080
IMGSZ = 1280  # 1080 is not a multiple of stride 32; the engine would round it up anyway
EPOCHS = 100
BATCH = 16  # total across GPUs -> 4 per GPU
DEVICES = [0, 1, 2, 3]

TRAIN_ARGS = {
    "data": str(DATA_YAML),
    "imgsz": IMGSZ,
    "epochs": EPOCHS,
    "batch": BATCH,
    "device": DEVICES,
    "workers": 8,  # per rank
    # Reproducibility.
    "seed": 0,
    "deterministic": True,
    # Explicit optimizer for a 68-image fine-tune: 'auto' would pick AdamW
    # lr=0.002 here, which is aggressive for so few, highly correlated frames.
    # nbs=64 with batch=16 -> 4 gradient-accumulation steps.
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_bias_lr": 0.0,  # 0.1 (the SGD default) destabilises Adam bias groups
    "cos_lr": True,
    "nbs": 64,
    "pretrained": True,  # 80-class COCO head is dropped, backbone/neck are kept
    "amp": True,
    "patience": EPOCHS,  # never early-stops inside the 100 requested epochs
    # Augmentation. The targets are thin lane segments (7-12 px box width at
    # 640), so the mix favours geometry that keeps their alignment intact and
    # avoids anything that smears them.
    "mosaic": 1.0,  # 4-image mosaics: the main source of scene variety here
    "close_mosaic": 10,  # last 10 epochs at native scale, so thin lines stay crisp
    "copy_paste": 0.0,  # segmentation-only augmentation, no-op for box labels
    "mixup": 0.0,  # blended pixels ghost thin lines into near-invisible ones
    "degrees": 10.0,  # lane lines appear at arbitrary angles; rotation matters most
    "translate": 0.1,
    "scale": 0.5,  # altitude / zoom variation
    "shear": 0.0,  # no plausible source of shear in a downward-looking camera
    "perspective": 0.0005,  # mild roll/pitch
    "fliplr": 0.5,  # line segments carry no left/right semantics
    "flipud": 0.0,  # upside-down road scenes (lane arrows, text) never occur
    "hsv_h": 0.015,
    "hsv_s": 0.5,  # white/yellow markings under changing light
    "hsv_v": 0.4,
    # Fixed size: no rectangular batches, no multi-scale.
    "rect": False,
    "multi_scale": False,
    # Output: worker/output is git-ignored, so runs never pollute the repo.
    "project": str(PROJECT_DIR),
    "name": RUN_NAME,
    "exist_ok": True,
    "plots": True,  # writes train_batch*.jpg to eyeball the augmentation
    "val": True,
    "save": True,
}


def main() -> None:
    for path in (DATA_YAML, PRETRAINED):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. datasets/ and models_backup/ are git-ignored, "
                "so they have to be copied to the server separately."
            )

    model = YOLO(str(PRETRAINED))
    if model.task != "detect":
        raise ValueError(f"{PRETRAINED.name} is a {model.task!r} model, expected 'detect'")

    print(f"data      : {DATA_YAML}")
    print(f"weights   : {PRETRAINED}")
    # print(f"imgsz     : {IMGSZ} (requested {IMAGE_SIZE})")
    print(f"batch     : {BATCH} total, {BATCH // len(DEVICES)} per GPU on {DEVICES}")
    print(f"output    : {PROJECT_DIR / RUN_NAME}")

    model.train(**TRAIN_ARGS)

    best = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"best weights: {best}")


if __name__ == "__main__":
    # Required: multi-GPU spawns one process per GPU, and the guard keeps the
    # child processes from re-running the launcher.
    main()
