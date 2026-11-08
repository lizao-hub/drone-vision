# drone-vision

无人机俯视视角下的高架车流分析：分割高架路面 → 其余区域涂黑 → 在涂黑图上检测车辆与车道线 → ByteTrack 追踪 → 按每个高架实体统计**密度、平均车速、拥挤度**；配套实时视频平台（推流 → 拉流推理 → 浏览器低延迟播放）与 Docker 一键部署。

![效果演示：高架路面分割与实例跟踪（#1/#2）、车辆跟踪与地速标注、左下角每板密度/均速/拥挤度](worker/img.png)

## 端到端链路

```text
Windows FFmpeg ──RTMP──▶ drone1 ─┐
                                 ├─▶ MediaMTX ──WebRTC(WHEP)──▶ 浏览器
worker: drone1 ──RTSP──▶ 算法 ──RTMP──▶ drone1_yolo ─┘
frontend(nginx :8080) ──/api──▶ backend ──Docker socket──▶ worker 容器（启停）
```

- `http://localhost:8080`：控制台（Vue3 + nginx，启停算法、查看状态）
- `http://localhost:8889/drone1`：原画（WebRTC 低延迟播放）
- `http://localhost:8889/drone1_yolo`：检测画

## Docker 一键部署

前置：Docker + Docker Compose；GPU 推理需要 NVIDIA Container Toolkit（取消 `docker-compose.yml` 里 worker `deploy` 段的注释）。
权重不入库，放到 `worker/models/` 下的三个 `best.pt`（路径见 [worker/README.md](worker/README.md)）。

```bash
docker compose up -d --build
```

一次拉起 4 个容器：

| 容器 | 内容 |
| --- | --- |
| `drone-mediamtx` | 流媒体服务（MediaMTX）：RTMP 入、RTSP/HLS/WebRTC 出 |
| `drone-backend` | FastAPI 控制面：经 Docker socket 启停 worker、查 MediaMTX 状态 |
| `drone-frontend` | Vue3 + nginx 控制台，WHEP(WebRTC) 播放 |
| `drone-worker` | 算法 worker：RTSP 拉流 → 推理 → RTMP 回推 `drone1_yolo` |

Windows 侧用 ffmpeg 模拟无人机/机巢推流（默认推 `assets/drone.mp4`，`-Video` 可指定）：

```powershell
.\scripts\push-windows.ps1
```

拿不到 GPU 时 worker 自动退回 CPU：画面能出，但轨迹/测速跟不上，仅作兜底。

## 本地运行（免 Docker）

三种模式，参数与数据结构详见 [worker/README.md](worker/README.md)：

```powershell
cd worker
$env:PYTHONPATH="$PWD\src"
# 离线 batch：视频进，标注 mp4 / JSONL 出
python -m drone_worker --source videos\video1_1080P.mp4
# 两卡流水线：分割在卡0、检测在卡1，多进程 + 共享内存（离线吞吐优化）
python -m drone_worker pipeline --source videos\video1_1080P.mp4 --devices 0,1
# 实时 relay：RTSP 拉流推理、RTMP 回推（串行链路）
python -m drone_worker relay --source rtsp://127.0.0.1:8554/drone1 --dest rtmp://127.0.0.1:1935/drone1_yolo --device 0
```

后端 `WORKER_MODE=process` 时直接在本机拉起 relay 进程，不构建镜像也能跑通前端 + 后端 + 算法。

## 算法

- **三模型协同**（imgsz 1280）：YOLO26m-seg（27.0M）分割高架路面 ROI → 涂黑非路面区域 → 两个 YOLO26m（21.8M）在遮罩图上双路检测车辆与车道线 → ByteTrack 轨迹 + 掩码 IoU 的高架实例跟踪。
- **精度**（自建数据集 val，imgsz 1280）：车辆 box mAP50 **0.983** / 车道线 box mAP50 **0.970** / 高架分割 mask mAP50 **0.975**。
- **车速估计**：车道线是静止的，其轨迹速度即无人机自身运动，相减补偿；像素-世界尺度用已知车宽（1.95m）/ 滚动车宽中位数自标定（俯视单尺度假设），不需要标定板或逆透视。
- **统计**：按高架实例输出覆盖率密度、平均车速、三级拥挤度（clear / moderate / severe：密度闸门 8% + 速度阈值 30 / 12 km/h）。

## 性能（1080P 视频、`--stride 3`、RTX 4080、PyTorch FP32）

| 项 | 数值 |
| --- | --- |
| 单卡串行 | 48.7ms/帧（三模型合计 43.8ms） |
| 两卡流水线（分割卡 + 检测卡） | 推理吞吐 **1.59×**（16.2 → 25.8 帧/s），854 帧输出与串行基线逐字段一致 |
| 6MB 帧跨进程交接 | `Queue`+pickle 5.99ms → 共享内存 **0.81ms** |
| ONNX 固定 shape FP16 导出 | 三模型单帧 43.8 → 35.6ms（**−19%**），精度代价 viaduct mask mAP50-95 −0.016 |

## 目录

```text
backend/       FastAPI 控制面（Docker socket 启停 worker、MediaMTX 状态）
frontend/      Vue3 + nginx 控制台，WHEP(WebRTC) 播放
deploy/        MediaMTX 配置
scripts/       Windows 端 ffmpeg 推流脚本
worker/        算法主体（uv 项目）
  src/algorithm/domain/      契约 + 状态机（车辆地速、每板密度/均速/拥挤度）
  src/algorithm/perception/  模型适配、ByteTrack、deck 追踪、pipeline/stages
  src/algorithm/runtime/     帧源、共享内存流水线、协调进程、RTMP sink、快照
  src/drone_worker/          batch / pipeline / relay 入口与渲染
```

## 测试

```powershell
cd worker
$env:PYTHONPATH="$PWD\src"
python -m pytest tests -q
```

108 个测试，纯 CPU，不需要权重和视频。依赖以 `worker/requirements.txt` 为准（`ultralytics==8.4.183`；旧版 ultralytics 的 track id 是全局计数，会让 `test_each_label_owns_its_identity_namespace` 失败）。
