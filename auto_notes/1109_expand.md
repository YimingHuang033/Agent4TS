# 1109 EXP — 管理员执行交接

任务发起：2026-11-09；下载、修复与验收跨午夜完成于 2026-11-10（Asia/Hong_Kong）。
依据：`auto_notes/2026-11-09_expplan.md`。

**当前状态：2026-11-10 00:33 已启动六阶段正式监督队列；启动时 WAITING_GPU，训练等待 GPU 0 释放。运维以 `2026-11-10_exp.md` 为准，不要重复执行下方旧启动命令。**
新实验已冻结为 `results/gen_eval/v3_expand_20261109/manifest.json`，930 个唯一单元全部 pending。
其中新数据与第 1–3 组核心实验合计 480 个唯一单元；第 4–5 组后续扩展另增 450 个。
旧 v2 目录只读保留，不能把旧分数混入本轮均值。

## 1. 主机与资源约束

- SSH：`BBQCard2`；目录：`/home/tim/Proj/ats`；现有 `tim` conda 环境。
- LLM：仅远程 Ark coding API，`deepseek-v4.1-flash-economy`。
- 密钥：`~/.config/agent4ts/ark.env`，不复制进仓库或日志。
- Ark 传输已修复：`thinking.type=disabled`，共同最多一次额外重试，每次请求均计预算；实际 API 验收 reasoning_tokens=0。
- 数据审计、调试、测试、出图一律 CPU。正式数值时序模型训练才允许 GPU 0。
- **不要启动本地 vLLM。** 验收时发现服务器另有 PID 2497305 的 `VLLM::EngineCore` 占用 GPU 0 约 21 GB；本任务没有启动、停止或使用它。PID/占用可能变化，管理员启动前重新检查并协调资源，不要直接照 PID 杀进程。
- 不创建 BITS 旧依赖环境；MIMIC 仍未提供；RepoHealth/TSRouter 原有阻塞不作替代实验。

## 2. 数据已经准备好，但采用了审计后的新版本

计划指定的两个 HF 快照都已下载，字节数、SHA256 与原计划完全一致，原文件保留在 `/mnt/data/ats_resources/data/`。
它们没有直接进入正式实验：

- USHCN：上游 GRU-ODE-Bayes `process.py` 在划分前全局标准化；原尺度/站点映射未随快照提供。改用 NOAA GHCN-Daily 中明确 HCN 标记的站点原始日观测。
- PAMAP2：快照只有 2 条记录、4 个匿名通道，无法证明准确的通道与受试者/会话来源。改用 UCI 官方原包的 Protocol 数据。

冻结任务：

- `ushcn_forecast`：按排序后的 HCN 站点 ID 等距选 20 站，1990–1999，720 个不重叠 100 天窗口；80 天历史、20 天预测、10 个查询桶。SNOW/PRCP 完整桶求和，SNWD 末次观测，TMAX/TMIN 完整桶极值；物理单位 days、mm、degC。所有同站窗口留在同一 split。
- `pamap2_forecast`：9 位受试者，按时间预选 168 个不重叠 10 秒窗口；8 秒历史、2 秒预测、10 个查询桶。使用心率、手部温度、手/胸/踝 x 轴 ±16g 加速度五个明确通道；不使用活动标签或无效 orientation。原始 NaN、IMU/HR 不同观测 mask 保留。同一受试者只进入一个 split。
- 两者均无 loader 插值、无全局归一化，训练统计仅由训练侧拟合。PAMAP2 子集只依据时间与预定列选取，不依赖测试数值。
- 两者都是自定义 forecasting 子集，不能称为完整 BITS/官方分类协议或全量数据集。

PAMAP2 官方完整 ZIP 在本地下载并校验；服务器使用 2.5 MB 可复现的原始观测子集 `pamap2/raw/pamap2_raw_windows_1109.zip`。
原包 SHA256、每个原始 subject 文件 SHA256、选列、窗口绝对时间均保存在该 ZIP 的 `manifest.json`。
`scripts/gen_eval/subset_pamap.py` 可从官方完整 ZIP 重建。服务器留有未完成的大包传输临时文件，它们不是训练输入；不必为了运行实验继续传输全包。

审计与证据：

- `results/smoke/expand_1109/{ushcn_forecast,pamap2_forecast}/data_audit.json`
- 同目录 `task_spec.json`
- `results/smoke/expand_1109/snapshot_audit.json`、`download_record.json`
- `resources.lock.json` 的 `expand_1109` 段
- 训练侧目标覆盖率：USHCN 0.9341；PAMAP2 1.0000。这里不是模型精度。

## 3. 已完成的调试验收

- `pytest tests/ -q`：32 passed。
- `cli validate`：两个新任务全部检查通过。
- 原有 `scripts/smoke/run_smoke.sh`：完整通过。
- 新数据全部 12 个模型 × 2 个任务：24 项 CPU fit、有限预测、导出重载、未来标签扰动检查通过。
- APN/ASTGI/KAFNet/tPatchGNN 等记录了 loss、梯度、权重变化和逐变量预测方差。近常数预测仍可能发生，不能由 smoke 宣称性能或训练充分性。
- random/greedy 在两任务上的 B=4 CPU 检查通过：f0 计预算、无重复候选、搜索未访问 confirm/test。
- no_vision 不发送图片、no_semantics 移除语义卡、one_shot 单次决策边界测试通过。
- 60 个 task/seed/space 组合的匹配策略动作空间哈希一致。
- 真 Ark API：两任务 × agent/TSci 共 4 次集成通过，包含 finalize，所有 GPU-hours=0。
- 原 TSci 同名参数覆盖问题已修复；不得把仅剩 f0 的 TSci 调用记作成功。
- 两次失败/中止联调目录保留：`api_attempt01_failed`、`api_attempt02_truncated`；最终通过记录为 `api/report.json`。
- 小预算 TSci 日志中的部分 “No successful TSci candidate” 来自前面候选耗尽预算；以 trace 的真实 trial 和总运行状态为准，不填合成分数。

关键日志：`log/smoke/1109_{prepare,allmodels,expand_api,pytest,validate,regression,manifest,status,summary_check}.log`。

## 4. 必须新跑的实验与复用关系

按以下顺序执行，runner 自动按唯一身份跳过已成功单元：

1. `new_data`：两新任务 × 18 方法 × 3 seeds，108 单元。
2. `matched_space`：10 任务 × 3 seeds ×（模型空间三策略 + 完整空间三策略），180 个阶段成员，其中与上一阶段重复 18 个，新增 162。
3. `budget`：agent/random/greedy，B=4/8/12，270 个阶段成员；B=12 复用，新增 180。B=4/8 从头运行，不截取 B=12 轨迹冒充。
4. `data_ablation`：model_only/data_only/full，90 个阶段成员；新增 data_only 的 30 个。
5. `robustness`（后续）：3 Physiome × 8 退化 × 5 方法 × 3 seeds，共 360 新单元。**本轮修复了网格聚合、观测统计和 API 行为，所以不能只跑 TSci 的 72 单元再拼旧四方法结果**；同版本四方法也重跑。
6. `initialization`（后续）：5 个指定任务、3 策略、APN/DLinear/TiWeaver 起点，135 个阶段成员；APN 复用，新增 90。

阶段成员数不能相加当唯一实验数。TSci-adapted 与 tsci_model_only 在本协议中使用同一实现/空间，manifest 将它们统一为 `tsci_model_only`，避免重复运行。

共用规则：seeds=[0,1,2]；B 含 f0、失败/超时训练；24 LLM 请求；2 GPU 小时；每次训练 600 秒；禁用 patience 截断；允许主动 STOP，记录实际预算。数据处理贡献比较固定 APN 参数。

## 5. 管理员启动命令

先检查机器与清单（不训练）：

```bash
ssh BBQCard2
cd /home/tim/Proj/ats
nvidia-smi
bash scripts/gen_eval/14_prepare_expand.sh
CUDA_VISIBLE_DEVICES="" bash scripts/gen_eval/15_run_expand.sh manifest
export ATS_KIND=gen_eval
source scripts/_env.sh
ats_run scripts/gen_eval/status_expand.py
```

GPU 0 可用后启动核心实验（本次交接尚未执行这段）：

```bash
cd /home/tim/Proj/ats
nohup bash -c 'set -e; for phase in new_data matched_space budget data_ablation; do CUDA_VISIBLE_DEVICES=0 bash scripts/gen_eval/15_run_expand.sh "$phase"; done' > log/gen_eval/v3_expand_20261109_core.log 2>&1 &
```

核心完成后，后续扩展单独启动：

```bash
nohup bash -c 'set -e; for phase in robustness initialization; do CUDA_VISIBLE_DEVICES=0 bash scripts/gen_eval/15_run_expand.sh "$phase"; done' > log/gen_eval/v3_expand_20261109_followup.log 2>&1 &
```

不要同时启动两段。runner 使用文件锁，避免相同清单并发写入。可用 `--limit N` 限制本次只跑 N 个新单元。

查看状态与汇总（CPU）：

```bash
export CUDA_VISIBLE_DEVICES="" ATS_KIND=gen_eval
source scripts/_env.sh
ats_run scripts/gen_eval/status_expand.py
ats_run scripts/gen_eval/summarize_expand.py
```

输出：`summary/summary.csv`、`summary.json`、`failures.json`、`curves.json`、`costs.json`、`groups.json`、`behavior.json`、`model_distribution.json`；有轨迹后生成 B=12 描述性前缀曲线 PNG。
均值只计算成功单元，报告 n_success/n_planned 与首轮成功数。没有正式结果时这些均值为 null，不能把当前空汇总当结果表。

## 6. 失败恢复与验收边界

- 所有原 attempt 永久保留；成功单元不重跑。
- 从 `summary/failures.json` 选需要重试的失败 cell ID，将 ID 字符串数组写入一个显式 retry JSON（不得包含 pending 充当失败）。
- 执行示例：`bash scripts/gen_eval/15_run_expand.sh new_data --retry agent_notes/1109_retry.json`。
- code/config 改动后 runner 拒绝恢复。必须另立结果版本与 manifest；不要篡改旧 manifest 或覆盖源码快照。
- 每单元结果保存在 `cells/<id>/attemptNNN/`，含身份、配置、源码哈希、trace、search、finalize、冻结预测器。
- 完成全部单元终态只是 execution_finished，不自动代表科研验收。检查失败、预算利用率、空间哈希、split/版本、覆盖率与实际选择后再写验收结论。
- 当前汇总脚本的 search wall time 包含 LLM、窗口构造、训练与评估；不是纯 GPU kernel 时间。详细 token/训练 attempt 在 trace；不要把它当精确训练能耗。
- 旧 v2 trace 可以作 legacy diagnostic，但不能与新协议合并画同一预算策略曲线。旧 v2 不做修改。

## 7. 本次冻结标识

- USHCN dataset_version：`ushcn_forecast_f13564587d5529d0`；train/search/confirm/test 窗口数 432/144/72/72。
- PAMAP2 dataset_version：`pamap2_forecast_2c0059581d7843f0`；train/search/confirm/test 窗口数 100/28/20/20。
- manifest code_hash：`de1819936e5bf3d155e6f2eacc21e096f9eb203317c57947b222716677533009`。
- `results/smoke/expand_1109/fairness_check.json` 记录 60 组空间哈希一致验收。
- `source_snapshot.tar.gz`、`environment.json`、脱敏后的 `llm.json` 与 manifest 同目录。

来源核对：

- NOAA HCN 下载与字段定义：<https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt>
- UCI PAMAP2 原始列/单位/缺失值：<https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring>
- USHCN 上游全局归一化代码：<https://github.com/edebrouwer/gru_ode_bayes/blob/ddd0b34e884dbee1c09b6a3927d1e9ab10443af8/data_preproc/Climate/process.py>
- Ark 深度思考参数：<https://docs.volcengine.com/docs/ark/deep-thinking>

- 最终汇总检查补充：B=12 前缀图只汇总 APN 起点、无退化的主任务，避免混入初始化/鲁棒性实验；最初未执行的清单草稿及源码快照保存在 `results/smoke/expand_1109/manifest_draft01/`。

## 正式启动更新

2026-11-10 00:33:15 +08:00，用户授权后已启动 `scripts/gen_eval/16_launch_expand.sh`，PID 2562933。全部六阶段顺序执行；启动时等待 GPU 0 空闲至少 20000 MiB 且无其他计算进程。监控、恢复与日志见 `2026-11-10_exp.md`。
