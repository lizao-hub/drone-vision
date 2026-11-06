# backend — 容器内的控制 API

浏览器不能直接启停容器，所以这个 FastAPI 服务做“桥”：

| 接口 | 作用 |
| --- | --- |
| `POST /api/algorithm/start` | 通过 Docker socket 启动 `drone-worker` |
| `POST /api/algorithm/stop` | 通过 Docker socket 停止 `drone-worker` |
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

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MEDIAMTX_API` | `http://127.0.0.1:9997` | MediaMTX 控制 API |
| `WORKER_CONTAINER` | `drone-worker` | 要控制的容器名 |

## 依赖

- `fastapi` / `uvicorn`
- `docker`：通过 socket 控制 worker 容器

`requirements.txt` 由 `uv.lock` 导出：

```powershell
cd E:\Pycharm\docker-demo2\backend
uv export --frozen --no-hashes --no-dev --no-emit-project -o requirements.txt
```

## 注意

- 挂 `docker.sock` 等于给容器宿主机 Docker 控制权，仅限本机演示。
- `docker compose down` 会删掉 worker 容器，之后“启动算法”会报容器不存在，重新 `docker compose up -d` 即可。
- 信号源（`drone1`）未上线时，启动算法会返回 409。
