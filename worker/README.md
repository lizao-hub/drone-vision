# drone-worker

无人机视角：先分割高架路面，其余涂黑，再把涂黑图送检测，最后用 ByteTrack 追踪 **bus / car / truck**，同时在涂黑图上检测车道线并追踪。

## 环境

- conda 环境 `pytorch`（torch 2.14.0 + ultralytics 8.4.183 + opencv 5.0.0）
- 权重：
  - `models/viaduct/weights/best.pt`：segment，单类 `viaduct`，训练 imgsz 960
  - `models/vehicle/weights/best.pt`：detect，单类 `car`（按当前权重）；检测端全部输出，追踪端只给 `bus/car/truck` 发 id，其余类别仍留在 `Observation.vehicle`
  - `models/line/weights/best.pt`：detect，单类 `line`，训练 imgsz 640；车道线段检测

## 运行

```powershell
$env:PYTHONPATH="E:\Pycharm\drone-vision\worker\src"
cd E:\Pycharm\drone-vision\worker
"D:\Software\Anaconda3\envs\pytorch\python.exe" -m drone_worker --video videos\test_Video1.mp4
```

`batch` 子命令等价（`python -m drone_worker batch --video videos\test_Video1.mp4`）。

输出：

- `output/<视频名>_pipeline.mp4`：高架区域不涂色、其余区域蓝色半透明 + 只给有速度的车画框并在框内写速度 + 左下角每块板一行 `#1 15.8% n9 22km/h moderate`（颜色与板上轮廓、`#N` 标签一致）
- `--json output/<视频名>.jsonl`（可选）：每帧一条 Observation

常用参数：`--stride`（默认 **3**，即每 3 帧推理一次、每 3 帧产出一个 Observation / 一行 JSON）、`--max-frames`（按采样帧计数）、`--car-width-m`（车辆参考宽度，默认 1.95）、`--viaduct-conf`、`--vehicle-conf`、`--line-conf`、`--line-imgsz`（默认 640）、`--no-video`，以及多卡布局的 `--segment-gpus` / `--vehicle-gpu` / `--line-gpu` / `--window` / `--mask-capacity`（见下文）。

## 单帧数据流

```text
frame
  → 分割(viaduct)                      # retina_masks=True，FP16，直接得到原帧尺寸掩码
  → Observation.viaduct: tuple[DetectedObject, ...]   # 每个实例独立 box + mask
  → 贪心掩码 IoU 匹配上一帧              # 实例顺序不稳定，靠这个给 deck 稳定 id
  → Observation.viaduct_tracks: tuple[TrackedObject, ...]   # track_id + box + mask
  → 多实例并集                          # 两段高架必须并集，取单实例会丢一半路面
  → 涂黑掩码外区域                       # masked[~mask] = 0
  → 检测(模型全部类别，单一 confidence)    # 输入就是涂黑图
  → Observation.vehicle（检测结果原样保留）
  → 每类一个 ByteTrack（id 形如 car-bt-000001，只给 bus/car/truck 发 id）→ Observation.vehicle_tracks
  → 车道线检测（同一张涂黑图，模型单类 line，imgsz 640）→ ByteTrack → Observation.line_tracks
  → Observation
```

`DetectedObject` 是唯一的结果类型：`label / confidence / box / mask=None`。分割适配器给每个实例填 `mask`，检测适配器留 `None`。`Observation.viaduct` 是分割实例（每个带自己的 mask，两段高架就是两个对象），`Observation.viaduct_tracks` 是同一批实例配上跨帧稳定 id 的结果（`track_id` 形如 `deck-000001`，`TrackedObject` 也带 `mask`，因为密度要按像素算），`Observation.vehicle` 是检测结果（未赋身份），`Observation.vehicle_tracks` 是配上 track_id 的轨迹，`Observation.line_tracks` 是车道线的轨迹（`track_id` 形如 `line-bt-000001`，`label="line"`）。用于涂黑的并集掩码是派生量，不重复放进 observation。

车道线检测与车辆检测共用同一张涂黑图，原因：实测原图上会捎带检测到桥下/路口的路面标线，涂黑后只剩高架路面的线；`frame_45` 两者都检出 3 条持续存在的线（`line-bt-000001/2/3`），`line_ms` 约 17ms（首帧预热 146ms）。

高架的实例顺序不稳定（同一块板在两帧里的下标可能互换），[viaduct_tracker.py](worker/src/algorithm/perception/viaduct_tracker.py) 用**掩码 IoU 贪心匹配**上一帧的 deck：≥0.3 就复用 id，未匹配的实例发新 id，消失的 deck 保留 `max_gap_ms=750` 以防单帧漏检换 id。实测 16 帧里两块板的顺序交换了多次，`deck-000001/000002` 始终跟着各自的板。

## Runtime：车辆地速（MVP）

[vehicle_speed_runtime.py](worker/src/algorithm/domain/vehicle_speed_runtime.py) 是唯一的测速 runtime：自己同时维护车辆流和地面流，车辆流减地面流后的绝对值 × 尺度就是每辆车的地速。

```text
v_car_img = 一条车辆轨迹 box 中心速度的 EMA                # px/s
v_ground  = 当前帧所有车道线轨迹速度的均值                  # px/s（地面静止 → 相机流）
speed     = |v_car_img - v_ground| * car_width_m / 全图车宽中位数
```

- 尺度 = `car_width_m`（默认 1.95）/ 最近 2s 全图车宽中位数，俯视视角不分带。
- 车道线轨迹和车辆轨迹各自都要攒够 `min_track_age=3` 个速度样本，所以前 3 帧是 `valid=False`（`flow_unavailable`）；车辆轨迹在尺度/地面流缺失的帧里也继续更新参考位置，不会每次中断都重新预热。
- 尺度缺失 → `scale_unavailable`；地面流缺失 → `flow_unavailable`；`frame_valid=False` 或降级帧冻结，不更新状态。
- `valid=False` 时 `speeds=()`，**不会给 0**；停着的车会测出真的 0（相对流≈0，而且与尺度无关）。
- 轨迹丢失 750ms 后从状态里删除；`valid=True` 但 `speeds=()` 表示这一帧没有成熟到可报的车辆，而不是"没有车"。

CLI 每帧打印 `car_speed t=1875ms: 12 cars (min 12.2 max 46.4 km/h)`，每个车框里画 `35km/h`，JSON 里是 `vehicle_speeds`（由 `ViaductUpdate.speed` 直接给出）。实测 120 帧：1388 个车速样本，均值 19.2 km/h（min 0.6 / max 52.6），117/120 帧有输出。

## Runtime：高架实体（密度 + 平均车速）

[viaduct_runtime.py](worker/src/algorithm/domain/viaduct_runtime.py) 是最终聚合层：一次 `ViaductRuntime.update(observation)` 给出每个 viaduct 实体的密度、平均车速和拥挤标签。

```text
density(deck)    = |(该板名下车辆框的并集) ∩ deck 掩码| / |deck 掩码|
mean_speed(deck) = 该板上成熟车辆地速的算术平均（逐帧，无权重）
congestion(deck) = 密度先做闸门、再看车速的启发式标签
```

| 密度 \ 均速 | ≥ 30 km/h | 12–30 km/h | < 12 km/h |
| --- | --- | --- | --- |
| **< 8%** | clear | clear | clear（车少，慢不等于拥堵） |
| **≥ 8%** | clear（车多但跑得动） | **moderate** | **severe** |

- deck id 直接取 `Observation.viaduct_tracks`（pipeline 里的 `ViaductTracker` 已经做了掩码 IoU 匹配），这个 runtime 不再自己匹配、不持有 deck 状态。
- 车辆地速来自注入的 `VehicleSpeedRuntime`（组合，不继承）。它每帧只能被调用一次：同一个时间戳再调一次会被当成新轨迹、清空预热。
- 框归属只看**框中心**：中心落在哪块板的掩码里就属于哪块；掩码重叠时 `deck_id` 小的优先，保证一辆车不会进两次平均。覆盖率也只统计这块板名下的框，压在边界上的框不两边都算。
- 密度用全部检测框（`Observation.vehicle`），平均车速只用攒够 `min_track_age` 的轨迹，所以 `vehicles` 和参与平均的车数可以不同。
- 速度侧不可用（定不出尺度 / 没有地面流）→ **整帧 `valid=False`**，reason 透传 `scale_unavailable` / `flow_unavailable`，不产出 deck；没有 deck → `viaduct_unavailable`。
- `valid=True` 但某块板 `mean_speed_mps=None`：这块板这一帧没有可测的车（没车，或轨迹还没成熟）；停着的车是测出来的 `0.0`，不是缺失。
- 冻结策略沿用 `VehicleSpeedRuntime`：`frame_valid=False` 或 `unknown_reasons` 非空的帧不产出 deck。
- 拥挤判定是**启发式**，不是交通流模型：三个阈值是 `ViaductRuntime(clear_density=0.08, clear_speed_kmh=30.0, severe_speed_kmh=12.0)` 的构造参数；没有可测车速时直接判 `clear`（不拿密度硬凑一个等级）。
- 密度是像素覆盖率，会随无人机高度变化：固定航高下够用，跨高度要换成"车辆数/面积"（`VehicleSpeedRuntime` 的 `scale_px` 能给出米/像素）。
- 逐帧判定会在阈值附近抖动，目前没有做迟滞；`valid=True` 时 `congestion` 一定有值（`clear` / `moderate` / `severe`）。

CLI 每帧打印 `viaduct t=125ms: deck-000001 15.8% n9 22.4km/h moderate, deck-000002 12.7% n4 clear`；视频左下角每块板一行，颜色与板上轮廓、`#N` 标签一致：`#1 15.8% n9 22km/h moderate`、`#2 12.7% n4 -- km/h clear`；没有 deck 统计的帧（速度不可用）左下角留空。JSON 里是 `viaduct_stats[*].decks[*]`（含 `congestion`）。`ViaductUpdate.speed` 里仍带每辆车的 `speed_mps / speed_kmh`，渲染和 JSON 的 `vehicle_speeds` 直接用它。

## 实时 relay（RTSP → RTMP）

```powershell
cd worker
$env:PYTHONPATH="E:\Pycharm\drone-vision\worker\src"
"D:\Software\Anaconda3\envs\pytorch\python.exe" -m drone_worker relay `
  --source rtsp://127.0.0.1:8554/drone1 `
  --dest rtmp://127.0.0.1:1935/drone1_yolo
```

链路：`drone1 ──RTSP──▶ 读线程 ──最新帧──▶ pipeline + ViaductRuntime ──annotate──▶ ffmpeg ──RTMP──▶ drone1_yolo`。

- **输入**：RTSP，或本地视频文件（按自身帧率回放、结束后循环，方便没有推流时联调）。
- **不累积延迟**：读线程只保留最新一帧（`algorithm/runtime/live.py` 的 `LatestFrameQueue`），pipeline 跟不上就丢旧帧，`--log-every` 的输出里有丢帧数。
- **输出配速**：按 `--out-fps`（默认 10）发送；pipeline 更快时 sleep 到下一个时隙，更慢时不追赶。
- **推流**：ffmpeg 子进程吃 `bgr24` 原始帧，编 H.264 后以 FLV/RTMP 输出；断流抛 `BrokenPipe` 后自动重建。ffmpeg 解析顺序 `--ffmpeg` → `$FFMPEG_BIN` → `PATH` → `imageio-ffmpeg`。
- **重连**：RTSP 断开、文件结束都按 1s 间隔重开；时间戳来自单调时钟并保证严格递增（tracker 会拒绝回退的时间）。
- **分辨率变化**：帧尺寸变了会重建 `VehicleSpeedRuntime` / `ViaductRuntime` 和 ffmpeg sink（尺度是按像素算的，必须跟着变）。
- **必须 GPU**：三个 worker 都把 `CUDA_VISIBLE_DEVICES` 钉在各自的卡上并用 `cuda:0`，没有 CPU 回退。relay 是实时链路，pipeline 至少要跑到几 fps 才能让轨迹跨帧接上。
- **慢管线参数**：`--track-timeout-ms`（默认 3000，ByteTrack 丢失超时）、`--speed-track-timeout-ms`（默认 3000）、`--scale-window-ms`（默认 5000）、`--min-scale-samples`（默认 5）都比批处理更宽松，专为"处理帧间隔比源帧大"的情况准备；管线足够快时可以不动。

关键参数：`--fps`（给 tracker 的名义源帧率，默认 25）、`--out-fps`（默认 10）、`--bitrate`（默认 4M）、`--rtsp-transport tcp|udp`、`--max-frames`（0=常驻，配文件源可做一次性回放）、`--log-every`，以及下面那张多卡布局表。

## 多进程多卡

### 结论先行

整条流水线就是**三个 worker 进程 + 一个父进程**：

```text
分割 ──▶ 并集遮罩 ──┬──▶ 车辆检测   ┐
                    └──▶ 车道线检测 ┴──▶ 父进程按帧号组装 ──▶ 追踪 ──▶ Observation
```

三段模型各占一个进程、各钉一张卡，并按帧流水线化：分割跑第 i 帧时，车辆/车道线
已经在跑第 i-1 帧。三段模型本身都没有跨帧状态，跟踪器留在父进程、严格按帧号
顺序组装，所以每帧的有效性、reason 和 id 都由同一个对象决定
（`perception/assembly.py` 的 `ObservationAssembler`）。

| 配置（测 100 帧真实 1080p 高架视频） | 每帧 | 吞吐 |
| --- | --- | --- |
| 分割 1 卡（0 / 1 / 2） | 20.4 ms | 49.1 fps |
| 分割 2 卡（0,4 / 5 / 6） | 14.1 ms | 70.9 fps |

（同一台机器连续跑的波动约 ±10%。三段本身的耗时：`segment 19.3ms / detect 9.8ms
/ line 9.3ms`。1 张分割卡时瓶颈是分割；2 张时分割摊到 8-10ms、瓶颈换成检测——
检测被持续压满后 boost 时钟下降，从 9.8ms 变成 10-11ms。）

端到端 `batch`（90 帧 1080p，含解码、密度/速度统计、JSONL，`--no-video`）：
分割双卡 61.7 fps。

冷启动（`import ultralytics` 拉进 torch）约 25s：worker 的模型加载/前向预热与
父进程的 tracker 预热并行，这笔开销不落在第 0 帧上。

### 拓扑

```text
父进程（纯 CPU，不加载模型、不碰 CUDA）
  submit(frame, ts, seq)                 ┌─ 分割 worker × N  GPU a, b, …
    └─ 写共享内存槽 seq ────────────────▶│  predict → 遮罩 → 写回 masked/masks
                                         │  └─▶ 车辆检测 worker  GPU c
                                         │  └─▶ 车道线检测 worker GPU d
  按帧号顺序组装 ◀──────── 结果队列 ─────┘
    └─ ObservationAssembler：三个 tracker + 失效策略 → Observation
```

- **共享内存环形槽**：`window` 个槽，每槽三块缓冲（`frame` 原图、`masked`
  遮黑图、`masks` 每实例掩码）。1080p 一帧跨进程只做一次 `memcpy`，不走
  pickle 管道。
- **跟踪器留在父进程**：`ByteTracker` / `ViaductTracker` 只被父进程按
  `frame_sequence` 顺序调用一次，每帧只调一次，所以跨帧状态只有一份。三段模型
  本身无跨帧状态，乱序到达不影响结果（结果按 seq 对齐，不是按到达顺序）。
- **判定只有一份实现**：`perception/assembly.py` 的 `ObservationAssembler` 决定哪种
  失败让整帧失效、reason 谁优先、遮罩为空时跳过检测。它只依赖
  `perception/viaduct_tracker.py` 和 `domain`，不 import 编排代码。
- **钉卡**：worker 在 `import torch` **之前**设 `CUDA_VISIBLE_DEVICES`，之后
  一律用 `cuda:0`，模型加载也只会落在自己的卡上。父进程在 fork 之前必须没有
  CUDA 上下文；调用方已经初始化过 CUDA 时自动改用 `spawn`（Windows 只有
  `spawn`）。
- **预热**：worker 启动时先跑一次空帧前向（cuDNN autotune 约 2s）；父进程在
  fork 之后用后台线程 `import ultralytics.trackers.byte_tracker`（冷缓存 ~21s），
  与 worker 加载重叠，避免这笔开销落在第 0 帧上。

### 用法

```powershell
# 批处理：分割用 0、4 两张卡，车辆检测 5，车道线 6
python -m drone_worker --video videos\test_Video1.mp4 `
  --segment-gpus 0,4 --vehicle-gpu 5 --line-gpu 6 --json output\run.jsonl

# 实时 relay 同理
python -m drone_worker relay --source rtsp://127.0.0.1:8554/drone1 `
  --dest rtmp://127.0.0.1:1935/drone1_yolo `
  --segment-gpus 0,4 --vehicle-gpu 5 --line-gpu 6
```

`--segment-gpus` 有几个卡就起几个分割 worker；总卡数 = 分割卡数 + 2。
卡号是 `nvidia-smi` 的物理序号，允许重复（不推荐，会互相抢算力）。

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--segment-gpus` | `0` | 分割 worker 的显卡列表，逗号分隔 |
| `--vehicle-gpu` | `1` | 车辆检测卡 |
| `--line-gpu` | `2` | 车道线检测卡 |
| `--window` | `4` | 在途帧数 = 共享内存槽数；吞吐/延迟/内存的折中 |
| `--mask-capacity` | `32` | 每帧最多几个分割实例（每个实例 H×W 字节） |

内存（1080p）：每槽 `2×6.2MB`（原图 + 遮黑图）+ `32×2.07MB`（掩码）≈ 79MB，
`window=4` 约 315MB 共享内存，由父进程创建、退出时 `unlink`。容器里跑要先
放开 `/dev/shm`（`docker-compose.yml` 的 worker 已设 `shm_size: 512mb`；容器
默认 64MB 会在建槽时 `ENOSPC`）。

`--mask-capacity` 不能乱调小：用 8 时，这段素材里 9-13 个实例的帧会直接判成
`viaduct_adapter_error`。实测该视频每帧实例数分布：
2 个 41%、3 个 21%、最多 13 个（`scripts/count_instances.py`）。换权重或改阈值
后重测一遍，再定这个值。

### 语义与已知限制

- **判定与统计只有一份实现**：有效帧判定、`unknown_reasons` 优先级、deck id、
  ByteTrack id、密度/车速统计都只在一个地方算。
- **分辨率固定**：pipeline 启动时按第一帧分配共享内存，之后换分辨率会抛
  `FrameGeometryChanged`；relay 捕获后重建 pipeline，batch 直接报错。
- **帧必须是 `uint8` 的 H×W×3**：共享内存按这个布局分配，`submit` 会校验。
- **无效帧**：`submit` 直接抛 `ValueError`（batch/relay 的帧源不会产生这种帧）。
- **`processing_ms` 的语义**：它是"这一帧的关键路径工作量"
  （`分割 + max(车辆, 车道线) + 组装`），不是排队延迟，所以它可能大于
  1000/fps——流水线是并行的，两者本来就不是一回事。真实吞吐看 fps。
- **首帧 autotune**：`batch` 在计时之前显式 `pipeline.warm_up(height, width)`
  （只传几何，内容无关），worker 启动时各跑一次空帧前向（cuDNN autotune 约 2s），
  所以 `wall/fps` 和 `median_proc` 都不受影响。
- **启动 ~25s**：几乎全部是 `import ultralytics`（拉 torch）和三个 worker 的模型加载。
- **检测是 2 个分割 worker 后的瓶颈**：再往上加吞吐要让检测也拆开，或者改用
  批推理，目前没做。

### 基准脚本

```powershell
$env:PYTHONPATH="src"
# 分段耗时/吞吐：分割用两张卡
python scripts\bench_stages.py --video ..\..\1080p.mp4 --size native `
  --segment-gpus 0,4 --vehicle-gpu 5 --line-gpu 6 --window 6

# 分割实例数分布（决定 --mask-capacity）
python scripts\count_instances.py --video ..\..\1080p.mp4 --frames 440 --stride 3
```

## 目录

`algorithm` 是自洽的四层包：`domain`（契约 + 状态机）、`perception`（模型适配 + 掩码 + 追踪 + 失效判定）、`engine`（三个 worker + 父进程编排，纯执行机制）、`runtime`（帧源、快照、组合根）。依赖单向向下：`runtime → engine → perception → domain`。`drone_worker` 只剩入口和可视化，任何新入口（容器实时循环、批量回归）都复用 `algorithm.runtime.factory`，不依赖 CLI 包。

`batch` / `relay` 只依赖 `algorithm/engine/pipeline.py` 的 `MultiProcessPipeline`（`warm_up` / `process` / `run` / `close`），吃的是 `FramePacket`（帧 + 帧号 + 时间戳 + fps），没有裸元组。两个入口的参数、校验和卡布局共用 `drone_worker/args.py` 里同一份定义，不会各写各的。

| 路径 | 职责 |
| --- | --- |
| `src/algorithm/domain/geometry.py` | `Box`、`Mask` 两个不可变值对象，不做 I/O |
| `src/algorithm/domain/observations.py` | `DetectedObject`/`TrackedObject`/`Diagnostics`/`Observation` 契约 |
| `src/algorithm/domain/vehicle_speed_runtime.py` | `VehicleSpeedRuntime`：车辆流减地面流 × 尺度 → 每辆车的地速 |
| `src/algorithm/domain/viaduct_runtime.py` | `ViaductRuntime`：每个 viaduct 实体的密度 + 平均车速，组合 `VehicleSpeedRuntime`，id 来自 `viaduct_tracks` |
| `src/algorithm/config.py` | `PipelineSettings` / `MpPipelineSettings`：阈值、类别、掩码参数、进程/显卡布局，纯数据 |
| `src/algorithm/engine/__init__.py` | 包说明：三个 worker + 一个父进程，纯执行壳（不做任何"这意味着什么"的判定） |
| `src/algorithm/engine/packet.py` | `FramePacket`：一帧 + 帧号 + 时间戳 + fps，引擎的输入；`FrameGeometryChanged`：换了分辨率的帧（relay 据此重建） |
| `src/algorithm/engine/shared.py` | 共享内存槽（原图/遮黑图/掩码）与进程间消息 |
| `src/algorithm/engine/workers.py` | 分割/车辆/车道线 worker，各自钉一张卡 |
| `src/algorithm/engine/pipeline.py` | `MultiProcessPipeline`：submit/result/run、按帧号组装、背压与关停；tracker 由组合根注入（父进程不 import factory） |
| `src/algorithm/perception/raw.py` | SDK 无关的辅助函数（张量转换/权重校验） |
| `src/algorithm/perception/segment_model.py` | `ViaductSegmentAdapter`：YOLO(seg) → 原始掩码 |
| `src/algorithm/perception/masking.py` | 掩码并集 / 涂黑 |
| `src/algorithm/perception/detect_models.py` | `DetectionAdapter`：YOLO(detect) → `DetectedObject`（车辆/车道线共用） |
| `src/algorithm/perception/tracking.py` | `ByteTracker`：每类独立 BYTETracker，`VEHICLE_LABELS` / `LINE_LABELS` |
| `src/algorithm/perception/viaduct_tracker.py` | `ViaductTracker`：贪心掩码 IoU，给高架实例分配跨帧稳定 id |
| `src/algorithm/perception/assembly.py` | `StageResult`（一帧模型输出 + 各段成败/耗时，失效原因带异常对象）+ `ObservationAssembler`（三个 tracker + 失效策略）：模型输出 → Observation 的合流与判定 |
| `src/algorithm/runtime/sources.py` | `VideoFileSource`：本地视频 → `FramePacket`（时间戳严格递增） |
| `src/algorithm/runtime/live.py` | `LiveSource` + `LatestFrameQueue`：实时取流、只留最新帧 |
| `src/algorithm/runtime/rtmp_sink.py` | `FfmpegSink`：原始帧 → ffmpeg → FLV/RTMP |
| `src/algorithm/runtime/snapshots.py` | `record()`：`Observation` + `ViaductUpdate` → JSONL 一行 |
| `src/algorithm/runtime/factory.py` | 组合根：`PipelineSettings` → 已装配的 pipeline；默认权重/视频路径 |
| `src/drone_worker/render.py` | 可视化：高架外蓝色半透明、有速度的车框 + 速度、左下角每板一行统计（同色轮廓 + `#N` 锚点） |
| `src/drone_worker/args.py` | 两个入口共用的参数定义与校验、`MpPipelineSettings` 装配、`build_pipeline_from`（唯一装配点） |
| `src/drone_worker/batch.py` | `python -m drone_worker [batch]`：视频文件进、标注 mp4 / JSONL 出（默认入口） |
| `src/drone_worker/relay.py` | `python -m drone_worker relay`：RTSP 进、标注 RTMP 出（实时入口） |
| `src/drone_worker/cli.py` | 入口分发：默认 / `batch` → `batch.py`，`relay` → `relay.py` |
| `scripts/bench_stages.py` | 分段耗时与吞吐，可选是否缩放输入 |
| `scripts/count_instances.py` | 统计分割模型每帧实例数分布，用来定 `--mask-capacity` |

## 行为约定

- 任一模型抛异常 → 整帧 `frame_valid=False`，`unknown_reasons` 记录原因，不产出内容（分割 `viaduct_adapter_error`、deck 追踪 `viaduct_tracker_error`、车辆检测 `vehicle_adapter_error`、车辆追踪 `tracker_error`、车道线检测 `line_adapter_error`、车道线追踪 `line_tracker_error`）。
- 分割正常但没检到高架 → 帧有效、`viaduct=()`、跳过检测（全黑图无意义），reason `viaduct_not_detected`（车道线也一起跳过）。
- 掩码只做本帧多实例并集：不做跨帧保持、不做边界膨胀、不按框中心过滤；检测输入就是并集掩码涂黑后的图。
- 新轨迹在第二次匹配后才发布（ByteTrack 标准确认策略），首帧只有已确认轨迹。
- 检测端不按类别过滤：车辆模型所有类别共用 `vehicle_confidence`（默认 0.30），车道线模型用独立的 `line_confidence`（默认 0.30）。
- 追踪端只给 `bus/car/truck` 发身份，`bike/person` 的框被 tracker 忽略（仍保留在 `Observation.vehicle` 里）；车道线只给 `line` 发身份。
- 检测只有一个置信度，因此 ByteTrack 的低分关联阶段实际为空。
- 采样：只有第 0、3、6… 帧会喂给引擎一次，因此 Observation 数量、JSONL 行数都等于 `源帧数 / stride`；`timestamp_ms` 仍是源视频时间（间隔约 125ms @24fps），轨迹按这个时间过期。
- 采样间隔变大后车辆位移也大，ByteTrack 的 IoU/卡尔曼关联会变弱：`stride=5`（208ms）时快速车道的车位移约 30px，相邻采样 IoU 只有 0.1~0.2，低于 ByteTrack 未确认轨迹的 0.3 门限，会一直“新建未确认轨迹→丢弃”，表现为检测到了但没有轨迹；`stride=3`（125ms）可以正常确认。
- 三处 `predict` 都固定 `rect=True`：按步幅取最小矩形 letterbox（1080p 16:9 下分割是 960×544、检测是 640×384），相比方图少约 40% 的填充像素；框坐标仍归一化到原图，retina 掩码也由 SDK 映射回原帧。
- 分割模型固定 `quantize=16`（FP16）：它是三模型里最慢的一段，也是唯一能吃满 tensor core 的模型；两个 640 检测模型保持 FP32。

## 测试

```powershell
$env:PYTHONPATH="E:\Pycharm\drone-vision\worker\src"
"D:\Software\Anaconda3\envs\pytorch\python.exe" -m pytest tests -q
```

128 个测试覆盖：几何校验、掩码并集/涂黑（含复用缓冲与"空桥面"判定）、`StageResult` 携带失效原因而不是布尔标志、`ObservationAssembler` 的六条失效优先级（分割/车辆/车道线适配器错误、两条 tracker 错误、空桥面跳过检测）、deck 贪心匹配（顺序交换/漏检保持/新板新 id/大掩码降采样）、ByteTrack 分标签 id 与时间戳约束、车辆地速 runtime（相对流、静止车为 0、方向、冷启动、冻结帧、尺度中位数）、高架实体 runtime（每板密度与算术平均车速、中心归属、重叠掩码只算一次、速度不可用整帧无效、冻结帧、`valid` 时 `mean=None` 表示该板没车可测、拥挤判定四档边界与可配置阈值）、实时 relay（最新帧队列丢旧、ffmpeg 参数、分辨率校验、relay 参数校验、帧尺寸变化时重建引擎）、共享内存槽与消息就绪判定、参数校验与 GPU 列表解析、分层依赖（`engine` 不 import `runtime`/torch、`engine.pipeline` 不 import 组合根、`assembly` 不 import 编排、`perception` 不 import 引擎、`domain` 不 import 上层）、两个入口共用同一套参数与校验、`cli.py` 的 `batch`/`relay` 分发。
