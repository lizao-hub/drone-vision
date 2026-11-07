# drone-vision

无人机俯视视角下的高架车流分析：分割高架路面 → 其余区域涂黑 → 在涂黑图上检测车辆与车道线 → ByteTrack 追踪 → 按每个高架实体统计**密度、平均车速、拥挤度**。

当前只保留离线 batch 链路：本地视频进，标注 mp4 / JSONL 出。

## 目录

```text
worker/                算法主体（uv 项目）
  src/algorithm/domain/      契约 + 状态机（车辆地速、每板密度/均速/拥挤度）
  src/algorithm/perception/  模型适配、ByteTrack、deck 追踪、pipeline
  src/algorithm/runtime/     文件帧源、JSONL 快照、组合根
  src/drone_worker/          batch 入口、分发、渲染
```

## 准备

- conda 环境（torch + ultralytics + opencv，版本见 [worker/README.md](worker/README.md)）
- 权重（不入库）：
  - `worker/models/viaduct/weights/best.pt`：segment，单类 `viaduct`
  - `worker/models/vehicle/weights/best.pt`：detect，单类 `car`
  - `worker/models/line/weights/best.pt`：detect，单类 `line`

## 跑 batch

```powershell
cd worker
$env:PYTHONPATH="$PWD\src"
python -m drone_worker --source videos\video1_1080P.mp4
```

输出到 `worker/output/`（目录来自 `PipelineSettings.project`）：`<视频名>_pipeline.mp4`（标注视频）和 `<视频名>.jsonl`（每帧一条 Observation），两者默认都写；用 `--no-save-video` / `--no-save-json` 关掉。参数、数据结构与各 runtime 的判定规则见 [worker/README.md](worker/README.md)。

## 测试

```powershell
cd worker
$env:PYTHONPATH="$PWD\src"
python -m pytest tests -q
```

89 个测试，纯 CPU，不需要权重和视频。
