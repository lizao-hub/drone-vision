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
"D:\Software\Anaconda3\envs\pytorch\python.exe" -m drone_worker --source videos\video1_1080P.mp4
```

`batch` 子命令等价（`python -m drone_worker batch --source videos\video1_1080P.mp4`）。

输出（目录由 `--project` 决定，默认 `worker/output`）：

- `<视频名>.jsonl`（`--save-json`，**默认开**）：每帧一条 Observation；`--no-save-json` 关闭
- `<视频名>_pipeline.mp4`（`--save-video`，**默认关**）：高架区域不涂色、其余区域蓝色半透明 + 只给有速度的车画框并在框内写速度 + 左下角每块板一行 `#1 15.8% n9 22km/h moderate`（颜色与板上轮廓、`#N` 标签一致）

常用参数：`--source`（输入视频，必填）、`--project`（输出目录）、`--device`（GPU 序号或 `cpu`）、`--stride`（默认 **3**，即每 3 帧推理一次、每 3 帧产出一个 Observation / 一行 JSON），以及运维开关 `--max-frames`、`--log-every`、`--save-video`、`--no-save-json`。

调参项不再走命令行，全部在 [config/pipeline.yaml](config/pipeline.yaml)：三个模型的权重路径、`imgsz`、置信度、`car_width_m`、ByteTrack 的 `max_lost_ms` / `match_threshold`。启动时用 pydantic 严格校验（未知字段、越界值直接报错），再经 `PipelineProfile.to_settings()` 变成 [config.py](src/algorithm/config.py) 的 `PipelineSettings`。

每行 JSONL 的 `diagnostics` 现在是 fps 归因表：`segment_ms` / `detect_ms` / `line_ms` / `processing_ms`（流水线内），`decode_ms`（读帧解码），`queue_wait_ms`（直播取帧等待，batch 为 0），`sink_ms`（画框 + 写 mp4）。batch 会打印 `run: revision=...` 头行和 `timing:` 汇总行。

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
  --dest rtmp://127.0.0.1:1935/drone1_yolo `
  --device 0
```

链路：`drone1 ──RTSP──▶ 读线程 ──最新帧──▶ pipeline + ViaductRuntime ──annotate──▶ ffmpeg ──RTMP──▶ drone1_yolo`。

- **输入**：RTSP，或本地视频文件（按自身帧率回放、结束后循环，方便没有推流时联调）。
- **不累积延迟**：读线程只保留最新一帧（`algorithm/runtime/live.py` 的 `LatestFrameQueue`），pipeline 跟不上就丢旧帧，`--log-every` 的输出里有丢帧数。
- **输出配速**：按 `--out-fps`（默认 10）发送；pipeline 更快时 sleep 到下一个时隙，更慢时不追赶。
- **推流**：ffmpeg 子进程吃 `bgr24` 原始帧，编 H.264 后以 FLV/RTMP 输出；断流抛 `BrokenPipe` 后自动重建。ffmpeg 解析顺序 `--ffmpeg` → `$FFMPEG_BIN` → `PATH` → `imageio-ffmpeg`。
- **重连**：RTSP 断开、文件结束都按 1s 间隔重开；时间戳来自单调时钟并保证严格递增（tracker 会拒绝回退的时间）。
- **分辨率变化**：帧尺寸变了会重建 `VehicleSpeedRuntime` / `ViaductRuntime` 和 ffmpeg sink（尺度是按像素算的，必须跟着变）。
- **需要 GPU**：relay 是实时链路，pipeline 至少要跑到几 fps 才能让轨迹跨帧接上；CPU 实测约 1 fps，轨迹每帧都断，只会出画面、没有速度（`flow_unavailable`）。`--device 0` 拿不到 GPU 时会自动退回 `cpu` 并在日志里说明。
- **慢管线参数**：`--track-timeout-ms`（默认 3000，ByteTrack 丢失超时）、`--speed-track-timeout-ms`（默认 3000）、`--scale-window-ms`（默认 5000）、`--min-scale-samples`（默认 5）都比批处理更宽松，专为"处理帧间隔比源帧大"的情况准备；管线足够快时可以不动。

关键参数：`--device cpu|0`（默认 `cpu`）、`--fps`（给 tracker 的名义源帧率，默认 25）、`--out-fps`（默认 10）、`--bitrate`（默认 4M）、`--rtsp-transport tcp|udp`、`--max-frames`（0=常驻，配文件源可做一次性回放）、`--log-every`。

## 目录

`algorithm` 是自洽的三层包：`domain`（契约 + 状态机）、`perception`（模型适配 + 追踪）、`runtime`（帧源、快照、组合根）。`drone_worker` 只剩入口和可视化，任何新入口（容器实时循环、批量回归）都复用 `algorithm.runtime.factory`，不依赖 CLI 包。

| 路径 | 职责 |
| --- | --- |
| `src/algorithm/domain/geometry.py` | `Box`、`Mask` 两个不可变值对象，不做 I/O |
| `src/algorithm/domain/observations.py` | `DetectedObject`/`TrackedObject`/`Diagnostics`/`Observation` 契约 |
| `src/algorithm/domain/vehicle_speed_runtime.py` | `VehicleSpeedRuntime`：车辆流减地面流 × 尺度 → 每辆车的地速 |
| `src/algorithm/domain/viaduct_runtime.py` | `ViaductRuntime`：每个 viaduct 实体的密度 + 平均车速，组合 `VehicleSpeedRuntime`，id 来自 `viaduct_tracks` |
| `src/algorithm/config.py` | `PipelineSettings`：阈值、类别、掩码参数，纯数据 |
| `src/algorithm/perception/raw.py` | SDK 无关的辅助函数（张量转换/权重校验） |
| `src/algorithm/perception/segment_model.py` | `ViaductSegmentAdapter`：YOLO(seg) → 原始掩码 |
| `src/algorithm/perception/masking.py` | 掩码并集 / 涂黑 |
| `src/algorithm/perception/detect_models.py` | `DetectionAdapter`：YOLO(detect) → `DetectedObject`（车辆/车道线共用） |
| `src/algorithm/perception/tracking.py` | `ByteTracker`：每类独立 BYTETracker，`VEHICLE_LABELS` / `LINE_LABELS` |
| `src/algorithm/perception/viaduct_tracker.py` | `ViaductTracker`：贪心掩码 IoU，给高架实例分配跨帧稳定 id |
| `src/algorithm/perception/pipeline.py` | 唯一编排点：一帧进，一个 `Observation` 出 |
| `src/algorithm/runtime/frame_packet.py` | `FramePacket`：一帧 + 帧号 + 时间戳 + fps |
| `src/algorithm/runtime/sources.py` | `VideoFileSource`：本地视频 → `FramePacket`（时间戳严格递增） |
| `src/algorithm/runtime/live.py` | `LiveSource` + `LatestFrameQueue`：实时取流、只留最新帧 |
| `src/algorithm/runtime/rtmp_sink.py` | `FfmpegSink`：原始帧 → ffmpeg → FLV/RTMP |
| `src/algorithm/runtime/snapshots.py` | `record()`：`Observation` + `ViaductUpdate` → JSONL 一行 |
| `src/algorithm/runtime/factory.py` | 组合根：`PipelineSettings` → 已装配的 pipeline；默认权重/视频路径 |
| `src/drone_worker/render.py` | 可视化：高架外蓝色半透明、有速度的车框 + 速度、左下角每板一行统计（同色轮廓 + `#N` 锚点） |
| `src/drone_worker/batch.py` | `python -m drone_worker [batch]`：视频文件进、标注 mp4 / JSONL 出（默认入口） |
| `src/drone_worker/relay.py` | `python -m drone_worker relay`：RTSP 进、标注 RTMP 出（实时入口） |
| `src/drone_worker/cli.py` | 入口分发：默认 / `batch` → `batch.py`，`relay` → `relay.py` |

## 行为约定

- 任一模型抛异常 → 整帧 `frame_valid=False`，`unknown_reasons` 记录原因，不产出内容（分割 `viaduct_adapter_error`、deck 追踪 `viaduct_tracker_error`、车辆检测 `vehicle_adapter_error`、车辆追踪 `tracker_error`、车道线检测 `line_adapter_error`、车道线追踪 `line_tracker_error`）。
- 分割正常但没检到高架 → 帧有效、`viaduct=()`、跳过检测（全黑图无意义），reason `viaduct_not_detected`（车道线也一起跳过）。
- 掩码只做本帧多实例并集：不做跨帧保持、不做边界膨胀、不按框中心过滤；检测输入就是并集掩码涂黑后的图。
- 新轨迹在第二次匹配后才发布（ByteTrack 标准确认策略），首帧只有已确认轨迹。
- 检测端不按类别过滤：车辆模型所有类别共用 `vehicle_confidence`（默认 0.30），车道线模型用独立的 `line_confidence`（默认 0.30）。
- 追踪端只给 `bus/car/truck` 发身份，`bike/person` 的框被 tracker 忽略（仍保留在 `Observation.vehicle` 里）；车道线只给 `line` 发身份。
- 检测只有一个置信度，因此 ByteTrack 的低分关联阶段实际为空。
- 采样：只有第 0、3、6… 帧会调用一次 `pipeline.process`，因此 Observation 数量、JSONL 行数都等于 `源帧数 / stride`；`timestamp_ms` 仍是源视频时间（间隔约 125ms @24fps），轨迹按这个时间过期。
- 采样间隔变大后车辆位移也大，ByteTrack 的 IoU/卡尔曼关联会变弱：`stride=5`（208ms）时快速车道的车位移约 30px，相邻采样 IoU 只有 0.1~0.2，低于 ByteTrack 未确认轨迹的 0.3 门限，会一直“新建未确认轨迹→丢弃”，表现为检测到了但没有轨迹；`stride=3`（125ms）可以正常确认。
- 三处 `predict` 都固定 `rect=True`：按步幅取最小矩形 letterbox（1080p 16:9 下分割是 960×544、检测是 640×384），相比方图少约 40% 的填充像素；框坐标仍归一化到原图，retina 掩码也由 SDK 映射回原帧。
- 分割模型固定 `quantize=16`（FP16）：它是三模型里最慢的一段，也是唯一能吃满 tensor core 的模型；两个 640 检测模型保持 FP32。Ultralytics 在 `--device cpu` 上会自动回退到 FP32。

## 测试

```powershell
$env:PYTHONPATH="E:\Pycharm\drone-vision\worker\src"
"D:\Software\Anaconda3\envs\pytorch\python.exe" -m pytest tests -q
```

86 个测试覆盖：几何校验、掩码并集/涂黑（含复用缓冲）、pipeline 容错（含车道线分支）与多实例 observation、deck 贪心匹配（顺序交换/漏检保持/新板新 id/大掩码降采样）、ByteTrack 分标签 id 与时间戳约束、车辆地速 runtime（相对流、静止车为 0、方向、冷启动、冻结帧、尺度中位数）、高架实体 runtime（每板密度与算术平均车速、中心归属、重叠掩码只算一次、速度不可用整帧无效、冻结帧、`valid` 时 `mean=None` 表示该板没车可测、拥挤判定四档边界与可配置阈值）、实时 relay（最新帧队列丢旧、ffmpeg 参数、分辨率校验、relay 参数校验）。
