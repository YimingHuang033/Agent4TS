# p12 tsci rerun @ trial_timeout_seconds=1200（队列结束后执行）

## 为什么

`matched_space` 阶段唯一的失败 cell 是 `6266c5d9543ac34eea365707`（p12_robust / tsci_model_only / seed2）：
attempt002 的 11 个 PatchTST 候选全部撞 `trial_timeout_seconds=600`，报 `No successful LLM-selected TSci candidate beyond f0`；
而同一网格在 attempt001 里有 4/10 成功（patch2-d16-l1 等），说明 600s 对 p12 上的 PatchTST 处在临界值，
训练时间随机器负载在阈值两侧浮动。这不是代码缺陷，也不是那次用户中断的残留。

## 做什么

在**独立目录 + 独立结果根**下重跑 p12 的 tsci 子集：

| 项 | 值 |
|---|---|
| 源码 | `git archive a8f19bd6…` + `v3_expand_20261109/source_snapshot.tar.gz`（与冻结目录同源） |
| 工作目录 | `/home/tim/Proj/ats_v3_p12t1200_1111`（自动创建） |
| 结果根 | `results/gen_eval/v3_p12t1200_20261111`（自动创建；通过符号链接落在主仓 results） |
| 变更 | `trial_timeout_seconds: 600 → 1200`；任务过滤 `ATS_TASK_FILTER=p12_native,p12_robust` |
| 范围 | 只跑 `matched_space` 阶段 method=`tsci_model_only` 的 6 个 cell（2 任务 × 3 seed） |
| 预期成本 | 每 cell 上限约 12×1200s + finalize；实际 1–2 h/cell，共约 6–12 GPU 小时 |

## 怎么用

```bash
# 1) 只做 plan：守卫检查 + 建目录 + 打补丁 + 冻结新 manifest + 列出待跑 cell（不动 GPU）
bash /home/tim/Proj/ats/auto_notes/p12_t1200/21_rerun_p12_t1200.sh plan

# 2) 确认后真正训练（GPU0，串行，需卡空闲）
bash /home/tim/Proj/ats/auto_notes/p12_t1200/21_rerun_p12_t1200.sh run
```

守卫行为：
- 队列 PID 还活着，或 `20_launch_1111.sh` 仍在进程表 → 直接拒绝退出（已验证）。
- `run` 前若 `nvidia-smi` 显示仍有 compute 进程 → 拒绝，不抢卡。
- 重复执行幂等：目录已存在则复用，补丁带存在性检查。

## 边界（已核对）

- 不写 `ats_v3_frozen_1111`：原冻结 config/runner 字节不变，`de181993…` 校验继续有效。
- 不写 `results/gen_eval/v3_expand_20261109`：已完成的 212 个 ok cell 原样保留。
- 不写主仓 `src/` `config/` `scripts/gen_eval/`：这三个目录参与 `run_metacaster_1111.py` 的
  `source_hashes`，改动会让 `meta_full` / `meta_k30`（共 120 cell）在启动时报
  `Frozen MetaCaster code/config changed`。所有改动只发生在新目录的副本里。
- 新根的 cell id 与旧根不同（identity 含 code_hash），因此**不能**把新根 tsci 行与旧根拼接：
  报告 p12 tsci 时整体取新根的 6 个 cell，并标注“1200s 协议变体，与 600s 子集不可合并”。

## 验证记录（plan 阶段离线可查的部分）

- `bash -n` 通过；篡改逻辑在 scratch 副本上演练过，diff 仅 3 行（root、timeout、import os + 2 行过滤）。
- `py_compile` 通过；补丁后 YAML 可解析（root/timeout 已改）。
- `ATS_TASK_FILTER` 的任务名与 `config/gen_eval/main.yaml` 的 key 精确匹配（`p12_native`、`p12_robust`）。
- 队列运行期间执行 plan 被守卫拒绝（pid 2671533）。
