"""Container control for the demo.

Only the AI worker is managed from here.  The video source is external: the
Windows simulator pushes ``drone1`` with ffmpeg, just like a real dock/drone
would publish before the platform ever gets involved.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

import docker
from docker.errors import DockerException, NotFound

MEDIAMTX_API = os.environ.get("MEDIAMTX_API", "http://127.0.0.1:9997")
WORKER_CONTAINER = os.environ.get("WORKER_CONTAINER", "drone-worker")


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


def path_online(name: str) -> bool:
    """Ask MediaMTX whether a path currently has a publisher."""
    url = f"{MEDIAMTX_API}/v3/paths/get/{name}"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            data = json.load(response)
        return bool(data.get("online"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False


worker = DockerController()
