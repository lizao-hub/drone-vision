"""FastAPI control plane for the drone demo.

The web console only manages the AI worker.  The video source is simulated on
Windows with ffmpeg and is deliberately outside the platform:

* ``/api/algorithm/*`` starts/stops the worker container over the Docker socket
* ``/api/status``      reports worker and MediaMTX path state

Video never flows through this service; the browser plays MediaMTX over WHEP.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .controllers import path_online, worker


def _status() -> dict:
    drone1 = path_online("drone1")
    drone1_yolo = path_online("drone1_yolo")
    return {
        "algorithm": {"running": worker.running(), "online": drone1_yolo},
        "paths": {"drone1": drone1, "drone1_yolo": drone1_yolo},
    }


app = FastAPI(title="Drone Demo Control", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/status")
def status() -> dict:
    return _status()


@app.post("/api/algorithm/start")
def algorithm_start() -> dict:
    if not path_online("drone1"):
        raise HTTPException(status_code=409, detail="信号源未上线，请先在 Windows 启动推流")
    try:
        worker.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _status()


@app.post("/api/algorithm/stop")
def algorithm_stop() -> dict:
    try:
        worker.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _status()
