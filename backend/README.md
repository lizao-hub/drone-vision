# backend — worker 控制 API

浏览器不能直接启停算法进程/容器，所以这个 FastAPI 服务做“桥”：

| 接口 | 作用 |
| --- | --- |
| `POST /api/algorithm/start` | 启动 worker（容器或本地进程，见 `WORKER_MODE`） |
| `POST /api/algorithm/stop` | 停止 worker |
| `GET /api/status` | worker 状态 + MediaMTX 两条流是否在线 |

推流不归平台管：Windows 侧的 `scripts/push-windows.ps1` 模拟真实设备，
先推 `drone1`，页面只负责观看和控制 AI。

视频本身不经过这个服务：浏览器直接用 WHEP 连 MediaMTX 的 8889。

## 运行方式

由 compose 启动，不单独运行：

```bash
docker compose up -d --build backend
```

容器需要两样东西（compose 已配好）：

1. `network_mode: host`：用 `127.0.0.1` 访问 MediaMTX 的 9997；
2. `/var/run/docker.sock`：启停 `drone-worker`。

## 本地进程模式（不构建镜像）

`WORKER_MODE=process` 时不再走 Docker socket，而是直接 `subprocess` 启动
`python -m drone_worker relay ...`，用于在开发机上跑通「前端 + 后端 + 算法」：

```powershell
$env:WORKER_MODE="process"
$env:WORKER_PYTHON="D:\Software\Anaconda3\envs\pytorch\python.exe"
$env:WORKER_WORKDIR="E:\Pycharm\drone-vision\worker"
$env:WORKER_PYTHONPATH="E:\Pycharm\drone-vision\worker\src"
$env:WORKER_ARGS="relay --source rtsp://127.0.0.1:8554/drone1 --dest rtmp://127.0.0.1:1935/drone1_yolo"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`/api/status` 的 `algorithm.online` 依然取自 MediaMTX 的 `drone1_yolo` 是否在线，
所以进程模式下的界面行为和容器模式一致。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MEDIAMTX_API` | `http://127.0.0.1:9997` | MediaMTX 控制 API |
| `WORKER_MODE` | `container` | `container`：Docker socket；`process`：本地子进程 |
| `WORKER_CONTAINER` | `drone-worker` | 容器模式下要控制的容器名 |
| `WORKER_PYTHON` | 当前解释器 | 进程模式下的 Python |
| `WORKER_ARGS` | `relay` | 进程模式下的参数（shell 风格拆分） |
| `WORKER_WORKDIR` | 当前目录 | 进程模式的工作目录 |
| `WORKER_PYTHONPATH` | 不设置 | 会拼到子进程的 `PYTHONPATH` 前面 |

## 依赖

- `fastapi` / `uvicorn`
- `docker`：通过 socket 控制 worker 容器

`requirements.txt` 由 `uv.lock` 导出：

```powershell
cd E:\Pycharm\drone-vision\backend
uv export --frozen --no-hashes --no-dev --no-emit-project -o requirements.txt
```

## 注意

- 挂 `docker.sock` 等于给容器宿主机 Docker 控制权，仅限本机演示。
- `docker compose down` 会删掉 worker 容器，之后“启动算法”会报容器不存在，重新 `docker compose up -d` 即可。
- 信号源（`drone1`）未上线时，启动算法会返回 409。
