# 数据与归档清单 / Data and archive inventory

本项目的训练样本原本由智能体与平台在线交互产生。提供的目录不含离线轨迹集、地图文件、训练日志、TensorBoard 事件或回放视频。这里归档实际存在的权重、材料与结果，并将接口检查示例明确标为合成数据。

Training samples were generated online through the platform. The supplied directory contains no trajectory dataset, map files, training logs, TensorBoard events or replay videos. This archive preserves available weights, materials and results; interface fixtures are labeled synthetic.

| 路径 / Path | 内容 / Contents | 来源 / Provenance |
|---|---|---|
| `results/competition.json` | 双语结果与证据字段 / Bilingual result and evidence fields | 报告第 3–4 页 / Report pp. 3–4 |
| `results/competition.csv` | 一行历史成绩 / One historical result row | 同上 / Same source |
| `metadata/original_files.json` | 57 个原文件的大小、SHA-256、对应路径、保留决定 / File size, SHA-256, destination and preservation decisions | 完整原目录扫描 / Complete source inventory |
| `metadata/feature_schema.json` | 3601 维布局与偏移 / Observation layout and offsets | 原代码 / Original code |
| `metadata/configuration.json` | PPO、奖励、环境与 workflow 配置快照 / Configuration snapshot | Config class + TOML + workflow constants |
| `metadata/metrics.csv` | 监控字段及真实计算含义 / Metric semantics | 算法与工作流 / Algorithm and workflow |
| `metadata/checkpoint_inspection.json` | 44 个权重张量、形状、参数量与校验值 / Tensor shapes, counts and hash | 安全权重加载检查 / Weights-only inspection |
| `metadata/offline_smoke_check.json` | 本次合成接口检查结果 / New synthetic software check | 离线脚本 / Offline script |
| `examples/synthetic_observation.json` | 明确标注来源的合成原始观测 / Labeled synthetic raw observation | 整理时创建，非真实对局 / Created for checks, not a game record |
| `../ckpt/` | 原模型、索引与平台元数据 / Original weights, index and metadata | 最终导出目录 / Final exported directory |
| `../docs/reports/` | 两份未改动 PDF / Two unchanged PDFs | 用户提供文件 / Supplied source files |

原目录 57 文件中保留 40 个：28 Python 源码、5 TOML、1 配置备份、1 `.gitignore`、1 模型、1 索引、1 元数据 JSON 和 2 PDF。排除 16 个可生成的 `.pyc` 及 1 个 `.kaiwu.sign`；原件仍在本地原目录，清单保存全部哈希和排除原因。

Of 57 source files, 40 are preserved: 28 Python sources, five TOML files, one configuration backup, one `.gitignore`, one checkpoint, one index, one metadata JSON and two PDFs. Sixteen reproducible `.pyc` files and one `.kaiwu.sign` are excluded; local originals remain intact and the inventory records every hash and reason.

模型保存为 PyTorch `state_dict`，不含优化器、梯度、采样池或随机状态。原 `kaiwu.json` 中 ZIP 文件名与哈希只是平台记录；该 ZIP 不在原目录内。`train_time` 单位未注明，保留原值。

The checkpoint is a PyTorch `state_dict`, without optimizer, gradients, replay or RNG state. ZIP name/hash in `kaiwu.json` is platform metadata; the ZIP itself was not supplied. `train_time` is retained without an assumed unit.

图表依据具体数据文件和代码生成，配套 SVG 与 PNG 均位于 `../docs/figures/`。环境示意图来自赛题说明第 2 页，成绩截图与证书来自报告第 3–4 页，位于 `../docs/evidence/`。

Figures are generated from the listed files and code, with SVG and PNG in `../docs/figures/`. The environment illustration comes from task-guide p. 2; leaderboard and certificate from report pp. 3–4, in `../docs/evidence/`.
