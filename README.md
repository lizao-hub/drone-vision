# drone-vision

无人机俯视视角下的高架车流分析：分割高架路面 → 其余区域涂黑 → 在涂黑图上检测车辆与车道线 → ByteTrack 追踪 → 按每个高架实体统计**密度、平均车速、拥挤度**。仓库同时包含一套本地演示平台（MediaMTX + Vue + FastAPI + worker 容器）。

## 效果

![一帧检测结果](worker/img.png)

- 蓝色半透明：高架**外**区域；高架内部不涂色。
- 绿色 `#1` / 橙色 `#2`：两块高架实体，轮廓、`#N` 标签与左下角统计三者同色。
- 车框里的 `km/h`：只标给测出速度的车，没有速度的车不画。
- 左下角每块板一行：密度（像素覆盖率）、车数 `n`、平均车速、拥挤度（`clear` / `moderate` / `severe`）。
## 目录

```text
worker/                算法主体（uv 项目 + Dockerfile）
  src/algorithm/domain/      契约 + 状态机（车辆地速、每板密度/均速/拥挤度）
  src/algorithm/perception/  模型适配、ByteTrack、deck 追踪、pipeline
  src/algorithm/runtime/     帧源、JSONL 快照、组合根
  src/drone_worker/          CLI 入口 + 渲染
backend/               控制 API（FastAPI + Docker SDK）
frontend/              Vue 3 + Vite + nginx
deploy/mediamtx/       MediaMTX 配置（推流演示用）
docker-compose.yml     本地演示平台编排
docs/                  需求文档
```

## 本地跑 worker

需要先放权重（不入库）：

- `worker/models/viaduct/weights/best.pt`：segment，单类 `viaduct`
- `worker/models/vehicle/weights/best.pt`：detect，单类 `car`
- `worker/models/line/weights/best.pt`：detect，单类 `line`

```powershell
cd worker
$env:PYTHONPATH="E:\Pycharm\drone-vision\worker\src"
"D:\Software\Anaconda3\envs\pytorch\python.exe" -m drone_worker --video videos\test_Video1.mp4
```

输出 `worker/output/<视频名>_pipeline.mp4`；加 `--json output/<视频名>.jsonl` 可另存每帧一条 Observation。参数、数据结构与各 runtime 的判定规则见 [worker/README.md](worker/README.md)。

## 测试

```powershell
cd worker
$env:PYTHONPATH="E:\Pycharm\drone-vision\worker\src"
"D:\Software\Anaconda3\envs\pytorch\python.exe" -m pytest tests -q
```

73 个测试，纯 CPU，不需要权重和视频。

## 容器平台（本地演示）

前置：WSL2 里的 Docker + Compose、Windows 侧的 FFmpeg（推流用）、`assets/drone.mp4`，以及 `worker/models/` 下的三个权重。

`docker-compose.yml` 编排四个服务：`mediamtx`（推流 / 播放）、`frontend`（:8080）、`backend`（:8000，经 Docker socket 启停 worker）、`worker`（算法容器）。

worker 当前是**批处理**入口：读挂载的视频、写标注 mp4。权重、视频、输出都通过卷挂载，不烤进镜像：

| 挂载 | 容器路径 |
| --- | --- |
| `worker/models` | `/app/models`（只读） |
| `worker/videos` | `/app/videos`（只读） |
| `worker/output` | `/app/output` |

```bash
docker compose up -d --build
docker compose ps
```

### 推流与播放

```powershell
# Windows：把 assets\drone.mp4 循环推到 MediaMTX 的 drone1
powershell -ExecutionPolicy Bypass -File scripts\push-windows.ps1
```

- 控制台：<http://localhost:8080>
- 原画：<http://localhost:8889/drone1>
- 检测流：<http://localhost:8889/drone1_yolo>（待接入，见下）

> 实时链路尚未实现：worker 暂时不会从 `rtsp://…/drone1` 拉流，也不会把结果回推到 `drone1_yolo`。`backend` 的「启动算法 / 停止算法」目前只是启停 `drone-worker` 容器（该容器跑完批处理就退出）。

## 当前状态

- 已完成：分割 / 检测 / 追踪、每板密度 + 平均车速 + 拥挤度、JSONL、标注视频、73 个测试、worker / backend / frontend 镜像与 compose。
- 未完成：实时 RTSP→RTMP 中继（worker 的 `relay` 入口）、平台侧实时展示 worker 输出。

## 安全说明

`backend` 挂载了 `/var/run/docker.sock`，等于拥有宿主机 Docker 控制权。本项目只用于本机演示，不要把 compose 直接暴露到公网。
