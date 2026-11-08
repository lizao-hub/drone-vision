"""Worker control for the demo.

Only the AI worker is managed from here.  The video source is external: the
Windows simulator pushes ``drone1`` with ffmpeg, just like a real dock/drone
would publish before the platform ever gets involved.

Two backends, selected by ``WORKER_MODE``:

* ``container`` (default): start/stop ``drone-worker`` through the Docker socket.
* ``process``: start/stop a local ``python -m drone_worker relay ...`` process,
  for running the platform on a workstation without building the image.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import docker
from docker.errors import DockerException, NotFound

MEDIAMTX_API = os.environ.get("MEDIAMTX_API", "http://127.0.0.1:9997")
WORKER_CONTAINER = os.environ.get("WORKER_CONTAINER", "drone-worker")
WORKER_MODE = os.environ.get("WORKER_MODE", "container").strip().lower()


class DockerController:
    """Start/stop the worker container through the Docker socket."""

    def __init__(self, name: str = WORKER_CONTAINER) -> None:
        self.name = name
        self._client: docker.DockerClient | None = None
        self._lock = threading.Lock()
        self._cache_at = 0.0
        self._cache_value = False

    def _container(self):
        if self._client is None:
            self._client = docker.from_env()
        return self._client.containers.get(self.name)

    def running(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if now - self._cache_at < 1.5:
                return self._cache_value
            try:
                value = self._container().status == "running"
            except (NotFound, DockerException, OSError):
                value = False
            self._cache_at = now
            self._cache_value = value
            return value

    def start(self) -> None:
        try:
            self._container().start()
        except NotFound as exc:
            raise RuntimeError(
                f"container {self.name} not found; run `docker compose up -d` first"
            ) from exc
        except DockerException as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            self._cache_at = 0.0

    def stop(self) -> None:
        try:
            self._container().stop(timeout=2)
        except NotFound as exc:
            raise RuntimeError(
                f"container {self.name} not found; run `docker compose up -d` first"
            ) from exc
        except DockerException as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            self._cache_at = 0.0


class ProcessController:
    """Start/stop the worker as a local subprocess (no Docker, local dev).

    ``WORKER_PYTHON`` (interpreter), ``WORKER_ARGS`` (shell-split arguments,
    default ``relay``), ``WORKER_WORKDIR`` and ``WORKER_PYTHONPATH`` configure it.
    """

    def __init__(self) -> None:
        self.python = os.environ.get("WORKER_PYTHON") or sys.executable
        self.args = shlex.split(os.environ.get("WORKER_ARGS", "relay"))
        self.workdir = os.environ.get("WORKER_WORKDIR") or None
        self.pythonpath = os.environ.get("WORKER_PYTHONPATH") or None
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            if not self.args:
                raise RuntimeError("WORKER_ARGS must not be empty")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if self.pythonpath:
                existing = env.get("PYTHONPATH")
                env["PYTHONPATH"] = (
                    f"{self.pythonpath}{os.pathsep}{existing}"
                    if existing
                    else self.pythonpath
                )
            try:
                self._process = subprocess.Popen(
                    [self.python, *self.args], cwd=self.workdir, env=env
                )
            except OSError as exc:
                raise RuntimeError(f"cannot start worker process: {exc}") from exc

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def path_online(name: str) -> bool:
    """Ask MediaMTX whether a path currently has a publisher."""
    url = f"{MEDIAMTX_API}/v3/paths/get/{name}"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            data = json.load(response)
        return bool(data.get("online"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False


worker = ProcessController() if WORKER_MODE == "process" else DockerController()
