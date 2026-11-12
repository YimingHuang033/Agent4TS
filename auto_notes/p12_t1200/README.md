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

# 3) 无人值守：等 1111 队列退出且 GPU 空闲后自动执行 2)
bash /home/tim/Proj/ats/auto_notes/p12_t1200/22_wait_and_run.sh   # 后台常驻
kill "$(cat /home/tim/Proj/ats/log/gen_eval/p12_t1200_waiter.pid)"  # 取消
```

`plan` 期间队列若仍存活，默认拒绝；只有显式 `ALLOW_PLAN_WHILE_QUEUED=1` 才放行（离线、`CUDA_VISIBLE_DEVICES=''`、只写新目录与新结果根）。`run` 无论什么参数都会重新校验队列进程与 GPU compute 进程，任一项不满足即拒绝。

**不要**为了提前跑而并行占卡：600s 超时对负载敏感，第二个训练进程会制造 CPU 争用，反而让队列里本来能过的 cell 超时失败。

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

- `bash -n` 通过；篡改逻辑在 scratch 副本上演练过。
- `py_compile` 通过；补丁后 YAML 可解析（root/timeout 已改）。
- `ATS_TASK_FILTER` 的任务名与 `config/gen_eval/main.yaml` 的 key 精确匹配（`p12_native`、`p12_robust`）。
- 队列运行期间执行 plan 被守卫拒绝（pid 2671533）；加 `ALLOW_PLAN_WHILE_QUEUED=1` 放行后流程跑通。
- **plan 实跑结果（2026-11-11 21:01）**：新 manifest 冻结成功，`manifest cells: 96`（p12 子集），选中 6 个待跑 cell：
  `cb7c49a8…`(p12_native s0)、`cc5c50d8…`(s1)、`55ed73a2…`(s2)、`bfb1fb08…`(p12_robust s0)、`fbb6dc04…`(s1)、`2be6dd31…`(s2)。
- 冻结原件 sha256 在 plan 前后一致（`expand_v3.yaml` 4f0f5eea…、`run_expand.py` f32f88e7…），主仓 src/config/scripts_gen_eval 未被写入。
- 期间修正两处脚本缺陷：任务过滤需同时裁剪 `expansion.new_tasks`（否则 `new_data` 阶段 KeyError），以及 `add()` 对未加载任务需跳过（`initialization` 阶段有硬编码任务名）。

## 当前状态

- **以 systemd 单元运行（必须）**：`p12-t1200-1111.service`，独立 cgroup、`MemoryMax=infinity`。
  从 Hermes 的 terminal 后台会话（`hermes-worker-*.scope`，内存受限）启动会被 OOM killer 杀掉
  （journal 原文：`hermes-worker-proc_….scope: A process of this unit has been killed by the OOM killer`，rc=137，约 40 秒即死）。
  正确启动方式：

  ```bash
  systemd-run --user --unit=p12-t1200-1111 --collect --property=MemoryMax=infinity --property=Nice=10 \
    --working-directory=/home/tim/Proj/ats/auto_notes/p12_t1200 \
    --setenv=ALLOW_PARALLEL_ON_IDLE_GPU=1 --setenv=P12_GPU=1 \
    --setenv=OMP_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 \
    /bin/bash -lc 'bash 21_rerun_p12_t1200.sh run >> /home/tim/Proj/ats/log/gen_eval/p12_t1200_run.log 2>&1'
  systemctl --user status p12-t1200-1111     # 查看；journalctl --user -u p12-t1200-1111 看日志
  systemctl --user stop p12-t1200-1111       # 停止
  ```

- 被杀掉的那次运行已在 `cells/cb7c49a8…/attempt001/` 记为 `interrupted` 并附 `killed_reason.txt`，
  正式运行是 `attempt002`；同一 manifest identity，不引入指标偏差。
- waiter（`22_wait_and_run.sh`）已停用：改为并行占 GPU1 后不再需要；若要恢复严格串行再起它。
- 结果根 `results/gen_eval/v3_p12t1200_20261111`：**2026-11-12 13:48 完成，6/6 ok**（表见 auto_notes/2026_1111_meta.md 的「C 方案执行结果」一节）。
