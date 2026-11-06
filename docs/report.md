# drone-vision 技术报告

> 生成时间：2026-11-06 · 验证环境：Windows 11 + WSL2 / RTX 4050 Laptop 6GB / Ryzen 7 7735H (8C16T) / 16GB

## 1. 项目简介

无人机俯视视角下的高架车流分析：先分割高架路面并将其余区域涂黑，再在涂黑图上检测车辆与车道线，用 ByteTrack 做多目标跟踪，最后按**每个高架实体**统计密度、平均车速与拥挤度，输出标注视频、JSONL 快照或实时 RTMP 流。仓库同时包含一套本地演示平台（MediaMTX + FastAPI + Vue 3），浏览器经 WebRTC 观看实时检测画面。

## 2. 技术栈

### 2.1 算法 / worker

| 类别 | 选型 |
| --- | --- |
| 语言 / 依赖管理 | Python 3.12（`>=3.12,<3.14`），uv（`uv.lock`，`uv export` 生成 requirements） |
| 深度学习 | PyTorch 2.14.0+cu126（CUDA 12.6）、Ultralytics 8.4.183 |
| 模型 | YOLO-seg（`viaduct`，imgsz 960）+ 2 × YOLO-detect（`car` / `line`，imgsz 640） |
| 跟踪 | ByteTrack（`ultralytics.trackers` + `lap`），每个标签独立 tracker |
| 图像 / 视频 | OpenCV 5.0（文件、RTSP 拉流、mp4 写出、绘制）、NumPy 2.5 |
| 推流 | ffmpeg 子进程 + `rawvideo` 管道 → H.264/FLV/RTMP |

### 2.2 后端

- FastAPI（>=0.115）+ Uvicorn，Python 3.12，uv 管理
- Docker SDK for Python（`docker>=7`）：经 `/var/run/docker.sock` 启停 worker 容器
- 双模式 worker 控制：`WORKER_MODE=container`（Docker）/ `process`（本地子进程，便于不构建镜像直接联调）
- REST：`GET /api/status`、`POST /api/algorithm/start|stop`、`GET /api/health`

### 2.3 前端

- Vue 3.5（Composition API / `<script setup>`）+ Vite 8 + `@vitejs/plugin-vue`
- 原生 WebRTC 播放器（WHEP 协议，零三方依赖：`fetch` + `RTCPeerConnection`）
- nginx 1.27（容器：静态文件 + `/api` 反向代理）
- Node 22（构建阶段）

### 2.4 基础设施

- Docker Compose：`mediamtx` / `backend` / `frontend` / `worker`，全部 `network_mode: host`
- MediaMTX 1.21.3：RTMP :1935、RTSP :8554、WebRTC/WHEP :8889、HLS :8888、API :9997
- WSL2 运行平台，Windows 端用 FFmpeg 推流（模拟真实无人机）
- 协议链路：RTMP 推入 → RTSP 拉取 → 推理 → RTMP 回推 → 浏览器 WHEP 播放

## 3. 架构

```text
worker/src/
  algorithm/
    domain/        不可变契约 + 状态机（无 I/O）
      observations.py / geometry.py
      vehicle_speed_runtime.py   # 每车地速
      viaduct_runtime.py         # 每板密度 + 均速 + 拥挤度
    perception/    模型适配、ByteTrack、deck 追踪、pipeline 编排
    runtime/       帧源（文件 / RTSP）、RTMP sink、JSONL 快照、组合根
  drone_worker/
    cli.py         入口分发（默认/`batch` → batch，`relay` → relay）
    batch.py       离线：视频文件 → mp4 / JSONL
    relay.py       实时：RTSP → 标注 RTMP
    render.py      渲染
```

- **分层**：`domain` 只依赖 NumPy 与自身契约；`perception` 隔离第三方 SDK（ultralytics 对象不出适配器）；`runtime` 承担 I/O 与装配；`drone_worker` 只做入口与展示。
- **数据契约**：`Observation` 一帧内的全部事实（deck 实例与轨迹、车辆检测与轨迹、车道线轨迹），不含任何决策。
- **状态机**：`VehicleSpeedRuntime` 同时维护车辆流与地面流（车道线静止 → 图像流即相机流），`ViaductRuntime` 组合它并按 deck 聚合；两者都遵循"缺失不写 0"的策略。
- **组合优先于继承**：`ViaductRuntime` 注入 `VehicleSpeedRuntime`，每帧只调用一次（时间戳不前进会被视为新轨迹）。

## 4. 代码规模

| 区域 | 文件数 | 行数 |
| --- | --- | --- |
| `worker/src`（算法 + 入口） | 29 | 2742 |
| `worker/tests` | 8 | 1162 |
| `backend/app` | 3 | 186 |
| `frontend/src` | 4 | 676 |

## 5. 测试

**86 个 pytest 用例**，纯 CPU、不加载模型、不需要视频，全量运行 **约 2.5 s**。

| 文件 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| `test_viaduct_runtime.py` | 23 | 每板密度/均速归属、重叠掩码、拥挤四档边界与可配置阈值、冻结帧、数据类不变量 |
| `test_pipeline.py` | 13 | 单帧编排、模型异常 → 整帧无效、多实例 deck、降级分支 |
| `test_vehicle_speed_runtime.py` | 11 | 相对流、停驶车为 0、方向、冷启动、冻结帧、尺度中位数 |
| `test_viaduct_tracker.py` | 11 | 掩码 IoU 匹配（含大掩码的降采样路径）、实例顺序交换、漏检保持、新板新 id |
| `test_relay.py` | 9 | 最新帧队列丢旧、ffmpeg 参数、分辨率校验、relay 参数校验 |
| `test_masking.py` | 8 | 掩码并集 / 涂黑、复用缓冲与尺寸校验 |
| `test_tracking.py` | 6 | 分标签 id、时间戳约束 |
| `test_geometry.py` | 5 | `Box` / `Mask` 值对象校验 |

## 6. 串行 pipeline 性能

### 6.1 方法

- 输入：`worker/videos/test_Video1.mp4`（640×368 @24fps）；另将同一段上采样到 **1920×1080**（与推流端 `push-windows.ps1` 的 1080p 输出一致）作为实时场景。
- 参数：`stride=1`、`imgsz=960/640/640`、默认阈值；GPU 为 RTX 4050 Laptop 6GB。
- 指标：每帧 `Observation.diagnostics.processing_ms`（分割 + 涂黑 + 车辆检测 + 车道线检测 + 三个 tracker + 组装），统计 **p50/p95**；剔除前 3 帧预热（首帧含模型加载 ≈ 3.6–4.1 s）。
- 串行吞吐 = `1000 / p50`；批处理 wall fps = CLI 输出的 `frames / wall`（含解码、runtime 聚合、JSON 写入）。

### 6.2 结果（优化前基线）

| 场景 | p50 | p95 | 串行吞吐 | 批处理 wall fps |
| --- | --- | --- | --- | --- |
| GPU，640×368 | **115.8 ms** | 134.6 ms | **8.6 fps** | 6.5 fps |
| GPU，1920×1080 | **212.2 ms** | 232.0 ms | **4.7 fps** | 3.6 fps |
| CPU，640×368 | **1168.5 ms** | 1187.2 ms | **0.86 fps** | 0.7 fps |

> 640×368 GPU 若同时写标注 mp4（渲染 + 编码），wall 从 6.5 fps 降到 6.0 fps，约 +12 ms/帧。优化后的数字见 6.6。

### 6.3 分阶段拆解（优化前，同一进程内实测，p50，各 40 帧）

| 步骤（p50 ms） | 640×368 | 1920×1080 | 倍数 |
| --- | --- | --- | --- |
| `segment_predict`（YOLO-seg + retina mask 上采样到原图） | 54.03 | 58.44 | 1.08 |
| `deck_tracker`（多实例掩码 IoU 匹配） | 0.18 | 4.44 | 25.0 |
| `union_mask`（实例并集 + `Mask` 拷贝） | 0.03 | 0.99 | 30.0 |
| `mask_any`（空掩码检查） | 0.01 | 0.01 | 1.4 |
| `apply_mask`（复制整帧 + 掩码外清零） | 4.51 | 32.74 | 7.3 |
| `detect_predict`（车辆检测） | 17.44 | 18.11 | 1.04 |
| `vehicle_tracker`（ByteTrack） | 1.41 | 1.39 | 1.0 |
| `line_predict`（车道线检测） | 17.02 | 18.04 | 1.06 |
| `line_tracker`（ByteTrack） | 0.78 | 0.77 | 1.0 |
| **合计** | **95.4** | **134.9** | **1.41** |

> 绝对毫秒数在同机不同次运行间有约 ±20% 的波动（GPU 频率/温度），6.2 与 6.3 是两次独立测量，比较倍数比比较绝对值更可靠；6.2 的 `processing_ms` 还包含置信度过滤与 `Observation` 组装（<1 ms）。表中"掩码类"= `deck_tracker` + `union_mask` + `mask_any` + `apply_mask`，640×368 约 4.7 ms（5%），1080p 约 38.2 ms（28%）。

### 6.4 结论（优化前）

- 串行 pipeline 在 **640×368 约 8.6 fps**、**1080p 约 4.7 fps**（RTX 4050 Laptop）；CPU 仅 0.86 fps，轨迹无法跨帧关联，因此实时链路要求 GPU。
- 模型部分基本与分辨率无关（YOLO 按固定 `imgsz` letterbox 推理）；变贵的是"按像素计"的掩码操作，从约 4.7 ms 涨到 38.2 ms，其中 `apply_mask` 的整帧复制 + 布尔索引清零（4.5 → 32.7 ms）和 `deck_tracker` 的全尺寸掩码 IoU（0.18 → 4.44 ms）是主体，1080p 下合计占 28%。
- 实时 relay 的发送节奏按 10 fps 配速，实际吞吐受处理速度限制（1080p 约 4–5 fps 输出），读线程只保留最新帧以保证低延迟而非高帧率。

### 6.5 优化空间（1080p 实测，同机 p50）

| 变体（单次调用） | p50 | 说明 |
| --- | --- | --- |
| `apply_mask`：`copy()` + 布尔索引（优化前） | 23–35 ms | 通用花式索引，内存带宽受限 |
| `fill(0)` + `cv2.copyTo(frame, u8, dst)`（**已采用**） | **0.51 ms** | 复用 dst；`copyTo` 不会清空掩码外，必须先 fill |
| `fill(0)` + `cv2.bitwise_and(..., mask=)` | 1.07 ms | 同样正确，略慢 |
| `cv2.bitwise_and(..., mask=)` 不清空 dst | 0.85 ms | **错误**：掩码外保留旧内容（单测抓到） |
| 掩码 `bool → uint8` | 1.28 ms | 每帧一次，无法避免（除非 `Mask` 改存 uint8） |
| GPU `torch.where`（每帧 H2D/D2H） | 1.75 ms | 传输占主导，和 cv2 打平 |
| 掩码 IoU：整幅 1080p | 0.64 ms/对 | 优化前每个 deck 比较 3–4 对 |
| 掩码 IoU：1/4 降采样（**已采用**） | 0.08 ms/对 | 语义不变，阈值 0.3 对误差不敏感 |

### 6.6 优化后复测

改动：`apply_mask` 改走 OpenCV（复用缓冲 + `copyTo`），`_mask_iou` 在长边 ≥128 时按 1/4 采样比较。

分阶段（同方法、同机、各 40 帧，p50）：

| 步骤 | 640×368 前 → 后 | 1080p 前 → 后 |
| --- | --- | --- |
| `apply_mask` | 4.51 → **0.22 ms** | 32.74 → **2.07 ms** |
| `deck_tracker` | 0.18 → 0.21 ms | 4.44 → **1.40 ms** |
| `union_mask` / `mask_any` | 0.04 → 0.06 ms | 1.0 → 0.96 ms |
| 模型三件套 | 88.4 → 88.2 ms | 94.6 → 93.3 ms |
| **合计** | **95.4 → 90.9 ms（10.5 → 11.0 fps）** | **134.9 → 99.9 ms（7.4 → 10.0 fps）** |

端到端批处理（`stride=1`，同一命令，含解码/JSON，wall fps）：

| 场景 | 优化前 | 优化后 |
| --- | --- | --- |
| 640×368，100 帧 | p50 115.8 ms / 6.5 fps | p50 89.4 ms / **8.5 fps** |
| 1080p，60 帧 | p50 212.2 ms / 3.6 fps | p50 101.9 ms / **6.6 fps** |

> 端到端的提升幅度大于分阶段（两次运行间 GPU 频率有 ±20% 波动，检测/车道线耗时也有差异）；**按同方法的分阶段对比，代码改动带来 1080p 单帧 −35 ms（−26%）**，与 6.5 的预测一致。改完后三个模型合计 93.3 ms 成为硬下限（10.7 fps）。

复现命令（GPU）：

```powershell
cd worker
$env:PYTHONPATH="E:\Pycharm\drone-vision\worker\src"
"D:\Software\Anaconda3\envs\pytorch\python.exe" -m drone_worker `
  --video videos\test_Video1.mp4 --stride 1 --max-frames 100 --no-video `
  --json C:\Temp\bench.jsonl
```

逐阶段统计：读取 JSONL 的 `diagnostics`，对 `segment_ms / detect_ms / line_ms / processing_ms` 求均值、中位数与 p95（剔除前 3 帧）。

## 7. 简历可用条目

**中文**

- 独立设计并实现无人机俯视高架车流分析系统（Python / PyTorch / Ultralytics / OpenCV）：YOLO 分割 + 双检测 + ByteTrack 多目标跟踪，串行端到端 **11.0 fps @640×368、10.0 fps @1080p**（RTX 4050 Laptop，优化后；CPU 0.86 fps）。
- 设计 `domain / perception / runtime` 三层架构与不可变数据契约，领域层零 I/O；**86 个 pytest 单元测试**纯 CPU 2.5 s 跑完，覆盖 pipeline 容错、追踪、测速与聚合逻辑。
- 定位并优化 1080p 下的内存带宽瓶颈：把掩码涂黑从 numpy 布尔索引改为 OpenCV SIMD + 缓冲复用（32.7 → 2.1 ms）、掩码 IoU 降采样比较（4.4 → 1.4 ms），单帧 135 → 100 ms（−26%）。
- 自研单目测速：以静止车道线的图像流为地面参考、车辆框宽中位数标定像素尺度，输出每车地速（停驶车测得真实 0），并据此计算每个高架实体的平均车速。
- 实现每实体密度 + 平均车速 + 三档拥挤度（clear / moderate / severe），并在渲染层用"同色轮廓 + `#N` 标签"解决多实体与统计行的一一对应。
- 搭建 RTSP → 推理 → RTMP 实时链路：读线程仅保留最新帧控制延迟、ffmpeg 子进程推流、断流自动重连；配合 Docker Compose（MediaMTX + FastAPI + Vue 3/WHEP）实现浏览器实时观看与启停。

**English keywords**

Python · PyTorch · Ultralytics YOLO (seg/detect) · ByteTrack · OpenCV · FastAPI · Vue 3 · Vite · WebRTC/WHEP · RTMP/RTSP · Docker Compose · MediaMTX · pytest

## 8. 已知局限

- 实时链路需要 GPU：CPU 端到端约 1 fps，帧间隔超过追踪器寿命，轨迹无法关联（只会出画面、不出速度）。
- 密度是像素覆盖率，随飞行高度变化；跨航高比较需改用"车辆数 / 面积"口径（已有 `scale_px` 可换算）。
- 拥挤度是按密度闸门 + 车速阈值的启发式，不是交通流模型；逐帧判定在阈值附近会抖动（未做迟滞）。
- 单目测速依赖"车宽先验 + 车道线静止"两条假设；航向角变化或车道线缺失时速度不可用（`flow_unavailable`）。
- 分割实例偶发拆/并，会为该板签发新的 `deck_id`（基准中出现了 `deck-000003/000004`），跨帧统计需按 id 对齐。
