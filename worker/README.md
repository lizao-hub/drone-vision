# drone-worker

无人机视角：先分割高架路面，其余涂黑，再把涂黑图送检测，最后用 ByteTrack 追踪 **car**，同时在涂黑图上检测车道线并追踪。

## 环境

- conda 环境 `yolo`（torch 2.7.1 + ultralytics 8.4.183 + opencv 4.11.0）
- 权重（三份都是 1280 训练、1280 推理，即 `PipelineSettings` 的默认值）：
  - `models/viaduct_seg/yolo26m-seg_1080/weights/best.pt`：YOLO26m segment（27.0M），单类 `viaduct`，imgsz 1280
  - `models/vehicle_detect/yolo26m_1080/weights/best.pt`：YOLO26m detect（21.8M），单类 `car`，imgsz 1280；检测端不按类别过滤，追踪端只给 `car` 发 id
  - `models/line_detect/yolo26m_1080/weights/best.pt`：YOLO26m detect（21.8M），单类 `line`，imgsz 1280；车道线段检测

## 运行

```powershell
cd worker
$env:PYTHONPATH="$PWD\src"
python -m drone_worker --source videos\video1_1080P.mp4
```

`batch` 子命令等价（`python -m drone_worker batch --source videos\video1_1080P.mp4`）。

输出：

- `output/<视频名>.jsonl`：每帧一条 Observation（默认唯一产物）
- `output/<视频名>_pipeline.mp4`：需要 `--save-video` 才写；高架区域不涂色、其余区域蓝色半透明 + 只给有速度的车画框并在框内写速度 + 左下角每块板一行 `#1 15.8% n9 22km/h moderate`（颜色与板上轮廓、`#N` 标签一致）

JSONL 总是写，mp4 默认关（它只用于肉眼检查/演示，回归和平台只吃 JSONL）。

常用参数：`--source`（**必填**）、`--device`（默认 `0`，或 `cpu`/`0,1`）、`--stride`（默认 **3**，即每 3 帧推理一次、每 3 帧产出一个 Observation / 一行 JSON）、`--log-every`（进度行间隔，默认 50）、`--save-video`。输出目录固定取 `PipelineSettings.project`（默认 `worker/output/`），CLI 不提供覆盖；阈值、imgsz、类别、`car_width_m`、`device`、`max_lost_ms` 同样全部走 `PipelineSettings`。源视频的 `fps` 只用于输出 mp4 的帧率和播放配速，不参与追踪参数（轨迹按毫秒过期）。

另有两个子命令，三个入口共用同一套参数（`--source` / `--device` / `--stride` / `--log-every` / 三个 imgsz + 三个 confidence）：

```powershell
# 两卡流水线：卡0 分割、卡1 双检测，多进程 + 共享内存环形缓冲（离线吞吐优化；
# 只写 JSONL，不接受 --save-video；--device 给两个索引是分卡，给一个是两段同卡）
python -m drone_worker pipeline --source videos\video1_1080P.mp4 --device 0,1
# 实时 relay：RTSP（或本地视频，按自身帧率循环回放）拉流 → 串行 pipeline 推理 → ffmpeg 推 RTMP
python -m drone_worker relay --source rtsp://127.0.0.1:8554/drone1 --dest rtmp://127.0.0.1:1935/drone1_yolo --device 0
```

relay 走**串行**推理链路，三个线程分工：读线程解码并只保留最新一帧（`dropped` 计数），主循环按 `--stride`（默认 1，按**源帧号**抽样）推理，渲染线程做 `annotate` + 推 RTMP（跟不上时丢已推理结果，`render_dropped` 计数）。

**输出帧率不单独配置**：`out_fps = 源fps / stride`，编码器时间戳和发送配速都用这个推导值，所以 `--stride 3` 在 30fps 源上就是 10fps 输出，不可能出现"参数配错导致视频变速"。日志同时打标称与实际帧率，偏差超过 10% 会告警一次（说明 stride 相对当前算力太小）。源断线/结束自动重连，ffmpeg sink 断开自动重启。CLI 没给的模型参数全部回落到 `PipelineSettings`；`--track-timeout-ms` 默认 3000（比 batch 的 1000 宽松，抗丢帧造成的轨迹间隙）。

## 单帧数据流

```text
frame
  → 分割(viaduct)                      # retina_masks=True，直接得到原帧尺寸掩码
  → Observation.viaduct: tuple[DetectedObject, ...]   # 每个实例独立 box + mask
  → 贪心掩码 IoU 匹配上一帧              # 实例顺序不稳定，靠这个给 deck 稳定 id
  → Observation.viaduct_tracks: tuple[TrackedObject, ...]   # track_id + box + mask
  → 多实例并集                          # 两段高架必须并集，取单实例会丢一半路面
  → 涂黑掩码外区域                       # masked[~mask] = 0
  → 检测(模型全部类别，单一 confidence)    # 输入就是涂黑图
  → Observation.vehicle（检测结果原样保留）
  → 每类一个 ByteTrack（id 形如 car-bt-000001，只给 car 发 id）→ Observation.vehicle_tracks
  → 车道线检测（同一张涂黑图，模型单类 line，imgsz 1280）→ ByteTrack → Observation.line_tracks
  → Observation
```

`DetectedObject` 是唯一的结果类型：`label / confidence / box / mask=None`。分割适配器给每个实例填 `mask`，检测适配器留 `None`。`Observation.viaduct` 是分割实例（每个带自己的 mask，两段高架就是两个对象），`Observation.viaduct_tracks` 是同一批实例配上跨帧稳定 id 的结果（`track_id` 形如 `deck-000001`，`TrackedObject` 也带 `mask`，因为密度要按像素算），`Observation.vehicle` 是检测结果（未赋身份），`Observation.vehicle_tracks` 是配上 track_id 的轨迹，`Observation.line_tracks` 是车道线的轨迹（`track_id` 形如 `line-bt-000001`，`label="line"`）。用于涂黑的并集掩码是派生量，不重复放进 observation。

车道线检测与车辆检测共用同一张涂黑图，原因：原图上会捎带检测到桥下/路口的路面标线，涂黑后只剩高架路面的线。

高架的实例顺序不稳定（同一块板在两帧里的下标可能互换），[viaduct_tracker.py](worker/src/algorithm/perception/viaduct_tracker.py) 用**掩码 IoU 贪心匹配**上一帧的 deck：≥0.3 就复用 id，未匹配的实例发新 id，消失的 deck 保留 `max_gap_ms=750` 以防单帧漏检换 id。

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

CLI 每帧打印 `car_speed t=1875ms: 12 cars (min 12.2 max 46.4 km/h)`，每个车框里画 `35km/h`，JSON 里是 `vehicle_speeds`（由 `ViaductUpdate.speed` 直接给出）。

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
- 拥挤判定是**启发式**，不是交通流模型：阈值是 `ViaductRuntime(clear_density=0.08, clear_speed_kmh=30.0, severe_speed_kmh=12.0, dwell_ms=700)` 的构造参数；冷启动时没有可测车速直接判 `clear`（不拿密度硬凑一个等级）。
- 标签由 `algorithm.domain.congestion.CongestionStabilizer` 稳住：某帧没有可测车速时**保持上一标签**，连续 3 s 没有可测车速才回落到 `clear`；另一个标签要连续持续 `dwell_ms=700` 才切换，所以阈值附近不再逐帧闪烁。**尚未做阈值回差**（8% 进 / 6% 出、30→33、12→14），那是下一步。状态按 `deck_id` 存，超过 3 s 没再出现就丢弃；`valid=True` 时 `congestion` 一定有值（`clear` / `moderate` / `severe`）。
- 密度是**同一帧内的面积比**，不是像素覆盖率随高度变化：正下视纯变焦时分子（车辆框并集 ∩ 桥面）和分母（桥面掩码）同比例缩放，读数**近似与航高无关**，固定航高下可直接用。真正会改变读数的是：车变小后的检测召回（框变紧或漏检）、桥面掩码质量、斜视时的透视面积权重，以及画面里露出的桥面范围。跨航高/跨相机不要直接比，业务口径应换成"辆/km"或占有率（`scale_px` 能给出米/像素，可换算桥面长度）。另外分子是检测框覆盖而不是车辆像素，框越松密度越高。
- 逐帧原始判定会在阈值附近抖动，所以标签是驻留过滤后的结果（见上一条）；原始判定仍可由 `CongestionStabilizer.classify()` 取到，用于排查。

CLI 每帧打印 `viaduct t=125ms: deck-000001 15.8% n9 22.4km/h moderate, deck-000002 12.7% n4 clear`；视频左下角每块板一行，颜色与板上轮廓、`#N` 标签一致：`#1 15.8% n9 22km/h moderate`、`#2 12.7% n4 -- km/h clear`；没有 deck 统计的帧（速度不可用）左下角留空。`#N` 是**画面里从左到右的序号**，与内部 track id 无关：relay 长时间运行、镜头切换或视频循环后会拿到新的 `deck-000006`，但画面编号始终从 `#1` 重新排。JSON 里是 `viaduct_stats[*].decks[*]`（含 `congestion`）。`ViaductUpdate.speed` 里仍带每辆车的 `speed_mps / speed_kmh`，渲染和 JSON 的 `vehicle_speeds` 直接用它。

## 目录

`algorithm` 是自洽的三层包：`domain`（契约 + 状态机）、`perception`（模型适配 + 追踪）、`runtime`（帧源、快照、组合根）。`drone_worker` 只剩入口和可视化，任何新入口（批量回归、离线评估）都复用 `algorithm.runtime.factory`，不依赖 CLI 包。

| 路径 | 职责 |
| --- | --- |
| `src/algorithm/domain/geometry.py` | `Box`、`Mask` 两个不可变值对象，不做 I/O |
| `src/algorithm/domain/observations.py` | `DetectedObject`/`TrackedObject`/`Diagnostics`/`Observation` 契约 |
| `src/algorithm/domain/vehicle_speed_runtime.py` | `VehicleSpeedRuntime`：车辆流减地面流 × 尺度 → 每辆车的地速 |
| `src/algorithm/domain/viaduct_runtime.py` | `ViaductRuntime`：每个 viaduct 实体的密度 + 平均车速，组合 `VehicleSpeedRuntime`，id 来自 `viaduct_tracks` |
| `src/algorithm/config.py` | `PipelineSettings` + 三个默认权重路径 + `DEFAULT_SETTINGS`：所有调参值、部署路径和默认输出目录（`project`）的唯一来源，纯数据 |
| `src/algorithm/perception/raw.py` | SDK 无关的辅助函数（张量转换/权重校验） |
| `src/algorithm/perception/segment_model.py` | `ViaductSegmentAdapter`：YOLO(seg) → 原始掩码 |
| `src/algorithm/perception/masking.py` | 掩码并集 / 涂黑 |
| `src/algorithm/perception/detect_models.py` | `DetectionAdapter`：YOLO(detect) → `DetectedObject`（车辆/车道线共用） |
| `src/algorithm/perception/tracking.py` | `ByteTracker`：每类独立 BYTETracker，`VEHICLE_LABELS` / `LINE_LABELS` |
| `src/algorithm/perception/viaduct_tracker.py` | `ViaductTracker`：贪心掩码 IoU，给高架实例分配跨帧稳定 id |
| `src/algorithm/perception/pipeline.py` | 串行编排点：一帧进，一个 `Observation` 出（三个 stage 的串行组合，行为与拆分前一致） |
| `src/algorithm/perception/stages.py` | `SegmentStage` / `DetectStage` / `TrackStage`：无状态模型级与有状态跟踪级的拆分，串行与并行路径共用 |
| `src/algorithm/runtime/frame_packet.py` | `FramePacket`：一帧 + `source_id`/`session_id`/帧号/时间戳/原始尺寸/`inference_eligible` |
| `src/algorithm/runtime/sources.py` | `FileSequenceSource`（多文件连续时间轴、实时配速、stride 标记、pause/resume）与 `RtspSource`（指数退避重连、连接状态）；`Capture` 可注入，测试不需要真实视频 |
| `src/algorithm/runtime/snapshots.py` | `record()`：`Observation` + `ViaductUpdate` → JSONL 一行 |
| `src/algorithm/runtime/factory.py` | 组合根：`build_pipeline(settings)` 与按 stage 的构建函数，把 `PipelineSettings` 装配成 pipeline/worker，不放任何数值或路径 |
| `src/algorithm/runtime/shm.py` | 共享内存环形缓冲（帧/掩码槽 + 引用计数回收）与跨进程请求/结果消息 |
| `src/algorithm/runtime/parallel_worker.py` | seg / detect worker 主循环（各加载自己那部分模型） |
| `src/algorithm/runtime/coordinator.py` | 两卡流水线协调进程：解码预取 → 派工 → 按 `frame_sequence` 重排 → 跟踪 → JSONL |
| `src/algorithm/runtime/telemetry.py` | 帧级耗时聚合与进度/汇总输出（batch 与 pipeline 共用） |
| `src/algorithm/runtime/live.py` | 实时帧源：读线程 + 丢旧保新单槽队列，RTSP 断线/文件结束自动重开 |
| `src/algorithm/runtime/rtmp_sink.py` | ffmpeg RTMP sink：raw BGR 帧进 stdin、FLV 出，进程挂了由调用方重启 |
| `src/drone_worker/render.py` | 可视化：高架外蓝色半透明、有速度的车框 + 速度、左下角每板一行统计（同色轮廓 + `#N` 锚点） |
| `src/drone_worker/batch.py` | `python -m drone_worker [batch]`：视频文件进、JSONL 出（`--save-video` 附带标注 mp4，默认关） |
| `src/drone_worker/pipeline.py` | `python -m drone_worker pipeline`：两卡流水线入口（分割/检测分卡，多进程 + 共享内存） |
| `src/drone_worker/relay.py` | `python -m drone_worker relay`：实时链路入口（RTSP 拉流 → 推理 → RTMP 回推） |
| `src/drone_worker/args.py` | 三个入口共享的 CLI 参数（`--source`/`--device`/`--stride`/`--log-every`/模型覆盖） |
| `src/drone_worker/__main__.py` | `python -m` 入口：路由 `batch` / `pipeline` / `relay`，无子命令等价 `batch` |

## 行为约定

- 任一模型抛异常 → 整帧 `frame_valid=False`，`unknown_reasons` 记录原因，不产出内容（分割 `viaduct_adapter_error`、deck 追踪 `viaduct_tracker_error`、车辆检测 `vehicle_adapter_error`、车辆追踪 `tracker_error`、车道线检测 `line_adapter_error`、车道线追踪 `line_tracker_error`）。
- 分割正常但没检到高架 → 帧有效、`viaduct=()`、跳过检测（全黑图无意义），reason `viaduct_not_detected`（车道线也一起跳过）。
- 掩码只做本帧多实例并集：不做跨帧保持、不做边界膨胀、不按框中心过滤；检测输入就是并集掩码涂黑后的图。
- 新轨迹在第二次匹配后才发布（ByteTrack 标准确认策略），首帧只有已确认轨迹。
- 检测端不按类别过滤：车辆模型单类、共用一个 `vehicle_confidence`（默认 0.30），车道线模型用独立的 `line_confidence`（默认 0.30）。
- 追踪端只给 `car` 发身份（`VEHICLE_LABELS`）；车道线只给 `line` 发身份。
- 检测只有一个置信度，因此 ByteTrack 的低分关联阶段实际为空。
- 采样：只有第 0、3、6… 帧会调用一次 `pipeline.process`，因此 Observation 数量、JSONL 行数都等于 `源帧数 / stride`；`timestamp_ms` 仍是源视频时间，轨迹按这个时间过期。
- 三处 `predict` 都固定 `rect=True`：按步幅取最小矩形 letterbox（1080p 下三个模型都是 1280×736），相比方图少约 40% 的填充像素；框坐标仍归一化到原图，retina 掩码也由 SDK 映射回原帧。
- 三个模型都跑 FP32（不传 `quantize`）：1280 下单帧约 19.6 / 12.3 / 11.9ms（分割 / 车辆 / 车道线）。分割关掉 FP16 后只慢约 1ms，所以保留 FP32。

## 测试

```powershell
cd worker
$env:PYTHONPATH="$PWD\src"
python -m pytest tests -q
```

测试覆盖：几何校验、掩码并集/涂黑、pipeline 容错（含车道线分支）与多实例 observation、deck 贪心匹配（顺序交换/漏检保持/新板新 id/大掩码降采样）、ByteTrack 分标签 id 与时间戳约束、车辆地速 runtime（相对流、静止车为 0、方向、冷启动、冻结帧、尺度中位数）、高架实体 runtime（每板密度与算术平均车速、中心归属、重叠掩码只算一次、速度不可用整帧无效、冻结帧、`valid` 时 `mean=None` 表示该板没车可测、拥挤判定四档边界与可配置阈值）、帧源（多文件时间戳连续、stride 只标记不丢帧、循环不重复时间、RTSP 退避重连/连接状态/环境变量）。
