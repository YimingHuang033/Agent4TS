# Agent4TS：最小实现设计

日期：2026-10-06。本文是待实现规格，不代表代码、数据下载或实验已完成。目标：实现一个能读取数据语义与时序图、选择数据处理和预测模块、根据执行反馈迭代改进的单 agent 系统，输出可独立运行的预测 pipeline。

## 1. 范围

实现两类鲁棒预测任务，共用数据接口、模型池和评估器；它们是两种任务起点，可以来自同一数据集。

- **A / `native_irr`：从原生不规则观测建立预测流程。** 数据获取、官方切分和预测协议统一通过 BITS 接入。保留真实时间、变量异步与缺失 mask；逐数据记录不规则性是天然形成还是上游预处理产生。
- **B / `robust_improve`：改进合法但表现不足的既有预测器。** 给定固定初始 pipeline `f0`，在长缺口、观测陈旧、异步采样和采样密度变化等条件下改进预测。包括自然困难子集和可控历史观测退化；不把预测误差自动当成程序错误。
- **`irr_fault_detection`：仅占位。** 预留时间单位/跨度误解、缺失占位误用、未来信息泄漏三类注错及识别接口，默认禁用。本期不生成故障任务、不实现自动修复、不报告检出率。

首版只做离线搜索。搜索完成后冻结数据变换、模型及参数，测试推理不再调用 LLM。自己的 controller、语义检查、动作调度和 harness 独立实现；公开预测模型可通过适配器复用，TSci 仅作为独立 baseline。

## 2. 数据、任务与下载

### 2.1 BITS 作为统一底座

入口：[论文与协议](https://arxiv.org/abs/2609.33303)、[作者代码入口](https://anonymous.4open.science/r/BITS-8F2E/)。论文列出 11 个数据集：PhysioNet、USHCN、Human Activity、Pamap2、EPA-Air、ClusterTrace、CESNET、APTC、FNSPID、Seabirds、GDELT。先全部登记，分批运行：P12 → USHCN / Human Activity / EPA-Air → 其余。

**接入状态（已更新）：已从匿名站的 ZIP 入口下载并校验 1,533 个源码文件，版本为 `bits-eval 0.2.0rc1`；未安装训练环境或运行实验。** 页面 URL 不是可直接使用的 Git remote；使用[完整源码 ZIP](https://anonymous.4open.science/api/repo/BITS-8F2E/zip)。下载快照 SHA256 见 `resources.lock.json`。本地源码位于 `/Users/tim/Projects/ATS/vendor/BITS-8F2E`，不把第三方代码整体提交到本仓库。

可复用以下上游安装步骤（在解压后的源码目录执行）：

```bash
conda create -n BITS python=3.10
conda activate BITS
pip install -e .
bits --version
bits dataset list
# 图模型按需安装，需匹配本机 PyTorch/CUDA：
pip install -e '.[graph]'
```

数据在 [BITS-data](https://huggingface.co/datasets/ykj111/BITS-data)。上游 `dataset_configs/legacy/catalog.yaml` 已固定下载 revision、文件大小和 SHA256；`bits run` / `bits suite run` 自动下载并校验所需数据，默认缓存 `~/.cache/bits/datasets/`，可用 `BITS_CACHE_DIR` 改位置。catalog 包含 19 个数据条目，正式 `bits-irregular-v1` suite 是 11 数据集 × 3 horizon，共 33 个任务；本项目以 suite 为准，不把额外条目自动纳入主实验。数据文件尚未实际下载验证。

所有 A 类数据通过 `BitsBackend` 接入，沿用快照的窗口、变量和切分。外部模型适配接口已核实为 `build_model(context)` 与 `forward_batch(model, batch, stage)`；仓库包含 `examples/external_models/tfmixer/tfmixer_bits_adapter.py` 示例。自己的 agent 搜索必须隔离 BITS 完整运行流程中的测试评分，逐轮只返回搜索验证反馈，最终才使用测试集。下载失败返回 `blocked_resource`，不静默替换数据。每个数据集还须审计时间语义、切分和原始来源。

### 2.2 必须登记的语义数据集

- **P12 / PhysioNet 2012（首批）**：A 类直接走 BITS；B 类从同一份事件建立观测陈旧、跨变量异步和近期采样密度变化子集。语义卡记录临床变量、单位、相对入院时间和缺失编码。B 类先用 24 小时历史预测随后 6 小时，仅在真实观测目标上评分；这是本项目协议，不能混报官方挑战结果。[官方说明与下载](https://physionet.org/content/challenge-2012/1.0.0/)、[Set A 压缩包](https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz)。原挑战是死亡风险预测，此处为数值预测。原始包用于追溯语义，不替换 A 类 loader。
- **Physiome-ODE（首批，机制模拟）**：B 类先选 `dupont_1991a`、`borghans_dupont_goldbeter_1997a`、`lenbury_pacheenburawana_1991`，依据配置核对实际模型标识和时间缩放。潜在轨迹生成多个历史观测版本，未来目标保持一致；完整未来轨迹仅供评估器使用。[代码](https://github.com/kloetergensc/Physiome-ODE)、[官方数据下载](https://zenodo.org/records/11492058)、[生成说明](https://github.com/kloetergensc/Physiome-ODE/tree/main/data_creation)、[模型配置](https://github.com/kloetergensc/Physiome-ODE/blob/main/resources/Top50_final.csv)。官方预制数据不能支持成对完整轨迹时，从生成脚本构建并单独标记自定义版本。
- **Time-IMM / EPA-Air（第二批）**：选 Los Angeles、Dallas、Denver；关注污染物缺口、温度与污染物异步及数值/文本时间语义。A 类用 BITS 的 EPA-Air；B 类语义扩展读取 Time-IMM，并使用独立 `dataset_version`，禁止混用两套 split。首版只用数值和审计过的元数据，默认 7 天历史预测 1 天，先在开发集确认覆盖率再冻结。单位和文本发布日期不能凭字段名猜测。[数据目录](https://github.com/blacksnail789521/Time-IMM/tree/master/data/EPA-Air/processed)、[Los Angeles 文件](https://github.com/blacksnail789521/Time-IMM/tree/master/data/EPA-Air/processed/Los_Angeles)。
- **Time-IMM / RepoHealth（第二批，审计通过后启用）**：先 React，再扩展 Transformers、Kubernetes、PyTorch。核对 `schedule_intervals.csv`、存量/区间流量/时长字段及初始化规则。仅在定义证实后开放“区间总量→速率→还原固定目标”；未来区间长度须在截止点已知。React 曾观察到负 `open_prs`，先追溯定义，不直接裁剪。默认 28 天历史预测未来 7 天内的既定查询，覆盖不足则在开发阶段调整后冻结。[数据及采样表目录](https://github.com/blacksnail789521/Time-IMM/tree/master/data/RepoHealth/processed)。
- **MIMIC-IV（条件接入）**：保留数据适配器和语义卡模板，取得官方权限后启用；研究观测陈旧和结果延迟可用，逐表核实事件时间与记录时间。禁止把其他 MIMIC 版本当成 MIMIC-IV。[明确版本 v3.0 的官方获取页](https://physionet.org/content/mimiciv/3.0/)、[Time-IMM 预处理说明](https://github.com/blacksnail789521/Time-IMM#mimic-preprocessing)。没有权限时显式 `blocked_access`，不阻塞前四项。

Time-IMM 的数值、文本可从[数据仓库](https://github.com/blacksnail789521/Time-IMM)下载；[IMM-TSF](https://github.com/blacksnail789521/IMM-TSF)只作格式/预处理参考，不再维护第二套主训练框架。每份语义卡填写变量含义、单位、瞬时值/存量/区间总量、缺失含义、时间来源、可用时间和来源链接；未知项填 `unknown`，对应语义变换不开放。

### 2.3 B 类鲁棒任务生成

`f0` 首选固定配置 APN；其未接通时可先用 DLinear 完成工程验证，但正式实验必须冻结并记录 `f0`，不能按测试表现替换。每个任务开放与 A 类相同的动作。

自然子集由截止点前的观测年龄、最大缺口、变量采样密度差、近期/较早窗口密度比定义，阈值只用训练集确定。子集可重叠，不能挑测试误差大的样本临时造组。

可控退化先在 Physiome 实现：随机删除历史观测比例 `{0, 0.2, 0.5}`；删除连续历史区间，占历史时长 `{0.1, 0.3}`；部分辅助变量保留 `{1/2, 1/4}` 的观测；历史噪声标准差为训练变量标准差的 `{0, 0.1, 0.3}`。先分别施加，不做全组合。这些是拟定配置，只改变历史副本，保留同一实体、截止点、查询与真实标签。所有派生版本跟随母轨迹进入同一 split。自然数据退化实验与天然缺失结果分开报告。

## 3. 预测模型与 agent baseline

“传统”在本文指非 agent 数值预测器，包含近期深度时序模型。以下 **10 个**均出现在 BITS 论文的模型列表；优先封装 BITS 实现，作者链接用于资源追溯和缺失实现补接，尚未承诺本地可运行。已在快照确认 APN、ASTGI、TiWeaver、tPatchGNN、iTransformer、PatchTST、DLinear 的实现目录；TFMixer 有外部适配示例，KAFNet 与 HyperIMTS 仍需从作者仓库补接，论文列出不等于当前快照已内置。

1. **TFMixer（ICML 2026）**，irr：[作者代码](https://github.com/decisionintelligence/TFMixer)。
2. **KAFNet（AAAI 2026）**，irr：[作者代码](https://github.com/zhouziyu02/KAFNet)。
3. **APN（AAAI 2026）**，irr、默认起始模型：[作者代码](https://github.com/decisionintelligence/APN)。
4. **ASTGI（ICLR 2026）**，irr：[作者代码](https://github.com/decisionintelligence/ASTGI)。
5. **TiWeaver（KDD 2026）**，不规则/异步建模：[论文](https://arxiv.org/abs/2606.03121)、[论文链接的代码归档](https://doi.org/10.5281/zenodo.20424563)。
6. **HyperIMTS（ICML 2025）**，irr：[作者实现 PyOmniTS](https://github.com/Ladbaby/PyOmniTS)。
7. **t-PatchGNN（ICML 2024）**，irr：[作者代码](https://github.com/usail-hkust/t-PatchGNN)。
8. **iTransformer（ICLR 2024）**，规则序列：[作者代码](https://github.com/thuml/iTransformer)。
9. **PatchTST（ICLR 2023）**，规则序列：[作者代码](https://github.com/yuqinie98/PatchTST)。
10. **DLinear（AAAI 2023）**，轻量规则序列：[作者代码](https://github.com/cure-lab/LTSF-Linear)。

额外实现 LastValue 作为评分和数据管线的 sanity baseline，不计入上述 10 个。首版不增加 foundation model。每个模型注册输入类型、是否接收 mask/Δt、可用数据动作和有限超参选项。规则模型经统一物理时间网格适配，保存 mask 和观测年龄供预处理/审计；不能声称原模型使用了它不接收的特征。所有模型最终映射到相同查询与评分目标。

**唯一外部 agentic baseline：TimeSeriesScientist（TSci）**。[论文](https://arxiv.org/abs/2510.01538)、[代码和安装步骤](https://github.com/Y-Research-SBU/TimeSeriesScientist)。上游自定义数据入口为 `date` / `OT` CSV，不能当作原生多变量 irr 接口。实现 `TSciAdapter`：保留其分析、规划、拟合验证流程，通过桥接工具调用本项目的候选执行器，接入相同数据、语义卡、图片、模型池、LLM 和预算；明确标记 **TSci-adapted**，列出适配改动，不声称原版复现。其内部切分/评分须路由到统一评估器。若适配无法保持上游策略，则标记未完成，不能用自己写的循环冒充 TSci。

为辨别迭代收益，另外保留固定 `f0`、同动作空间随机搜索，以及我们的 `no_vision`、`no_semantics`、`one_shot` 消融；它们不是新增外部 agent 框架。

## 4. 最小代码结构与接口

BITS 执行环境固定 Python 3.10：上游限制 `>=3.8,<3.11`，PyTorch `>=2.4,<2.5`、NumPy 1.24.4、Pandas 1.5.3。controller 可用独立 Python 3.11 环境与 Pydantic，通过 JSON/子进程调用 BITS；TSci 也单独环境，避免依赖冲突。Matplotlib/PyYAML 等遵守上游依赖锁定。训练先支持单 GPU 串行，CPU 只用于轻量 smoke test。

```text
configs/                   # 数据任务、模型注册、有限动作空间、预算
src/agent4ts/
  schemas.py               # DatasetCard / TaskSpec / PipelineSpec / Feedback
  data/{bits,physiome,timeimm,mimic}.py
  models/{registry,bits_bridge,regular_adapter}.py
  transforms.py            # 受限数据模块
  observe.py               # 统计、语义卡读取、时序/缺失/误差图
  controller.py            # 单个多模态 LLM，输出结构化动作
  harness.py               # 校验、执行、缓存、评分、归档、回退、预算
  baselines/{tsci,random_search}.py
  faults.py                # 禁用的注错与识别占位
  cli.py
tests/                     # 信息边界、mask、切分、评分和闭环测试
resources.lock.json        # 上游版本、下载来源和校验值
runs/<run_id>/             # 不提交数据、密钥、权重和敏感记录
```

最小数据契约：

```python
Event = (entity_id, event_time, available_time, variable_id, value)
TaskSpec = {dataset_version, track, split_ids, target_variables,
            cutoff_rule, history_duration, horizon, query_policy, metric_version}
PipelineSpec = {data_view, transform_config, model_id, model_config, seed}
Feedback = {status, checks, metrics, groups, delta_vs_parent, delta_vs_f0,
            support_counts, plot_paths, cost, remaining_budget}

load_task(task_spec) -> TaskBundle
inspect(task_id, allowed_view) -> {dataset_card, statistics, plot_paths}
propose(observation, archive_summary) -> Action
run_trial(task_id, pipeline_spec) -> Feedback
predict(frozen_pipeline, history, query_times) -> predictions
inject_fault(...) -> NotImplementedError
detect_irr_fault(...) -> {"status": "not_implemented"}
```

事件值只存真实观测；批次 padding 另带布尔 mask。时间统一为有单位的相对时间，保留原始时间、换算规则与 interval 边界。可用历史同时满足 `event_time <= cutoff` 和 `available_time <= cutoff`；只有事件时间的数据明确记录“可用时间等于事件时间”的假设。

`TaskBundle` 分离 agent 可见的训练/搜索验证视图与评估器私有标签。agent 不能读取最终测试标签、目标 mask 或未知未来测量计划。自定义协议使用预先固定查询网格，桶边界为 `(左, 右]`，P12/EPA-Air 的瞬时值按桶内真实观测均值生成标签，无观测桶不评分；其他字段遵守已冻结的聚合语义。目标聚合规则不由 agent 修改。若 BITS 使用事件查询，则按其协议单独标记 `query_policy`，仅提供逐查询 `(time, variable)`，不将完整未来观测排程编码成历史特征。

## 5. 我们的最小 pipeline

流程：`任务与语义 → 统计和图像 → 提议一个动作 → 校验 → 训练/预测 → 验证反馈 → 接受或回退 → 下一轮 → 导出最佳 pipeline`。

### 5.1 观察：数值 + vision + 语义

`inspect` 返回历史趋势小图、真实时间轴上的观测散点/缺口图、变量覆盖率、Δt/age 统计和语义卡。每轮最多 4 张图、每图最多 6 个变量，按训练侧规则固定抽样；显示单位、截止线、图例，长缺口不画成连续实测曲线。多模态 LLM 必须实际接收 PNG，不是只收到文件路径。

执行后可增加搜索验证集的预测/残差图以及分组误差图。禁止提供确认集/测试集图；图内所有预测都须用该样本截止点前信息生成。视觉观察形成可验证假设，最终接受候选只看确定性检查与数值反馈。

### 5.2 动作：有限模块选择

动作写成 JSON：`{op, parent_id, field, value, evidence_refs, expected_effect}`；记录简短决策依据和证据编号，不要求自由写代码。每次只改一个字段：

- `SELECT_DATA`：选择注册任务的训练视图（完整历史/近期历史、全部合法协变量/预定义子集）；不能删验证/测试困难样本，不能改变目标变量或任务。
- `SET_TRANSFORM`：历史时长 `{0.5L, L}`、事件直读/固定网格、训练均值填充/有限期限前向保持、观测 age 特征开关、训练侧 z-score/min-max。网格宽度及保持期限由每个数据配置提供有限候选；兼容性矩阵禁止不适用组合。区间速率变换只对已审计的 RepoHealth 开放。
- `SELECT_MODEL`：选择第 3 节模型；`SET_MODEL_OPTION`：选择注册过的 patch 尺度、隐藏维度等少量配置，不开放任意超参字典。
- `INSPECT` / `STOP`：查看允许的诊断或停止；保留原 pipeline 是合法结果。

整模型与数据模块选择是本期最小原型；之前讨论的连续时间模式增删、自由函数结构搜索不作为首版前置条件。

### 5.3 Harness 与反馈

1. 检查 JSON、动作白名单、数据/模型兼容性、时间与 split 边界；不合法则返回结构化原因，不启动训练。
2. 预处理仅在训练侧拟合；训练规则固定。候选用配置、数据版本、split、seed、上游版本哈希去重与缓存。
3. 返回搜索验证集归一化 MAE/MSE、原单位逐变量误差、长缺口/陈旧/异步组误差、有效标签数、相对父候选及 `f0` 的变化、运行错误与成本。空组返回 `null + n=0`；失败不能伪装成零误差。
4. 候选通过检查且主指标降低则更新 incumbent；分数相同时选成本较低者。所有候选记入 archive，退步/失败则保持原 incumbent，下一轮获得该失败反馈。
5. 默认每任务、每 seed 最多 12 次真实训练（含 `f0`）、24 次 LLM 调用、2 GPU 小时；连续 3 次已完成训练无改善即停止，LLM 主动停止或任一预算耗尽也停止。阈值是先导默认值，开发后冻结；超时/失败消耗已用预算。最多重试一次格式错误并计入调用数。
6. 导出 `best_pipeline.json`、预处理参数、模型权重、`metrics.json`、`trace.jsonl`、资源版本及复现配置。trace 保存输入摘要、图片哈希、动作、反馈、接受/回退、token/耗时。单独提供无需 LLM 的 `predict` 入口。

配置示例（拟实现，不是 BITS 原生参数）：

```yaml
task: p12_robust_24h_6h
track: robust_improve
backend: bits
initial_model: APN
observations: [statistics, semantics, vision]
budget: {trials: 12, llm_calls: 24, gpu_hours: 2, patience: 3}
seeds: [0, 1, 2]
fault_detection: false
```

## 6. 实验约束与验收

**协议隔离。** 保留 `bits_official` 用于固定模型复现，另设 `agent_search` 用于迭代比较；保留 BITS 训练/测试边界，把官方 validation 按原切分原则再分成搜索/确认两部分。实体数据按实体切，长序列按时间切；标签区间不跨边界，必要时清除重叠窗口。非 BITS 数据采用固定 60/20/10/10 的训练/搜索/确认/测试划分。时间序列窗口与轨迹派生版本遵守同一隔离规则。

**模型选择。** 参数训练及早停只使用训练侧内部划分。agent 反复查询搜索集，结束后锁定最多 3 个候选，确定性程序在确认集选一次；之后只对最终模型评测测试集，不再把结果反馈给 agent。不做在线测试适配。

**公平与指标。** TSci-adapted、随机搜索、我们的系统共享动作空间、语义/图片权限、LLM 型号及训练/调用/算力上限。固定模型分别报告实际成本，不能把 10 模型总训练成本与一次搜索混为一谈。BITS 官方指标单独保留；自定义任务主指标为训练侧标准化后的 MAE，先在每个实体-变量的有效目标上平均，再对有效实体-变量对宏平均。同时报告 MSE、原单位误差、评分覆盖率、各鲁棒组、相对 `f0` 改善率与成本；零分母改善率记 `null`。三 seed 报均值和离散度，失败和无改善任务全部保留。

按以下顺序交付：

1. **数据与执行底座**：锁定 BITS，P12 + 三个 Physiome 模型、LastValue/APN/DLinear 跑通；验证无未来泄漏、改变 padding 不影响统计/评分、未知目标不计分、母轨迹不跨 split、网格预测回到固定目标。
2. **单 agent 原型**：语义卡、实际 vision 输入、数据/模型动作、反馈迭代、回退/停止与可重放日志；一次成功、一次退步和一次超时均能正确结束，导出模型可离线预测。
3. **对照及扩展**：补齐 10 模型、TSci-adapted、随机与三项消融；接 EPA-Air，完成 RepoHealth 审计后运行；MIMIC-IV 权限不足保留状态，注错识别保持禁用。

计划提供 `python -m agent4ts.cli {prepare,benchmark,search,evaluate,predict}`，统一接收 `--config` / `--run-dir`。验收产物是可复跑命令、数据/上游锁文件、结果和动作轨迹；不以“agent 必须胜过 baseline”作为工程验收条件。

本文范围依据项目内《鲁棒时序预测：benchmark、语义子集与 agent 改进任务》《Idea Draft：控制论启发的组合式时序预测函数搜索》及 Raymond 调研；以本次要求更新为 BITS 主底座、vision 首版接入、错误注入识别占位。资源链接只证明来源可追溯，不代表已完成训练复现。

---

## 附录：实现状态（2026-10-06，实现提交时更新）

本仓库已按本文实现最小系统，验收以 `scripts/smoke/run_smoke.sh` 与 `tests/`（12 项通过）为准。与设计的差异如实记录：

- **BITS 运行时未安装**（用户决定保留 `tim` 环境 py3.11/torch2.13，不建 py3.10 环境、不全量下载 BITS-data）。源码快照已下载审计（`/mnt/data/ats_resources/vendor/bits_tmp`，版本 0.2.0rc1）；`load_bits_task` 如实返回 `blocked_resource`。A 类任务走原生 P12 loader（本项目协议，非官方挑战任务）。
- **P12 Set A 已下载校验**（4000 条记录，sha256 见 resources.lock.json），24h→6h 桶均值协议跑通。
- **Physiome-ODE 为本地 RK4 复现**（三个指定模型），Zenodo 预制数据未下载，dataset_version 标记自定义版本；母轨迹派生版本不跨 split（有测试）。
- **Time-IMM EPA-Air** loader 已实现，CSV 需先运行下载脚本；与 BITS EPA-Air 使用独立 dataset_version。
- **模型池**：LastValue/MeanValue/DLinear/PatchTST 风格/iTransformer 风格可跑（忠实重实现，非作者代码）；TFMixer/KAFNet/APN/ASTGI/TiWeaver/HyperIMTS/tPatchGNN 登记为 `not_vendored`，选择它们会得到带原因和来源链接的结构化拒绝。f0 以 DLinear 顶替 APN 并记录在案。
- **TSci-adapted 未接通**：上游未 vendored，runner 输出 `status: unavailable` 与接入步骤，不以自写循环冒充。
- **MIMIC-IV**：`blocked_access`；**RepoHealth**：未审计未注册；**注错识别**：按设计禁用占位。
- 预算、单字段 JSON 动作、搜索/确认/测试隔离、trace.jsonl、best_pipeline.json 导出、无 LLM 的 predict 入口均已实现并冒烟验证。
