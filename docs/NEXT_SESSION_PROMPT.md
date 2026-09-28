# 新会话启动提示词（复制粘贴用）

> 本文件是给下一个会话的**开场指令**。直接整段复制提交即可。

---

## 角色与任务

你在维护 `Point-conditioned-OCR-finetuning`（Unsloth `FastVisionModel` + TRL，只用 `uv`）。

**任务**：截图上画固定品红准星（X）→ 模型只输出准星所在的**最小语义块**（Q1，提示词 `a2_v3`）；
Q2 在同一几何上再输出 `<source>`（英文原文）+ `<translation>`（简体中文）XML（提示词 `ocr_mt_v1`）。
空白 / 壳体 / 表格 / 纯图 → 输出**精确空字符串 `""`**。

**部署目标**：**0.8B**（消费级终端）。4B 作为 teacher（SFT 本地/云端皆可，GRPO/OPD 需 32GB）。
翻译质量优先级低于定位；翻译后续单独用数据/两阶段解决。

## 第一步（务必先做）

1. **完整读** `docs/HANDOVER_2026-09-28.md`（现状/结论/资产/坑/下一步/文档地图）。
2. 再看 `README.md` §一、§二（工程入口与磁盘内容），以及你要动的模块对应的 `docs/*.md`。
3. 跑 `uv run pytest -q`（应为 **183 passed**）+ `git log -1`（应为 `2eb9cf9`，作者 DerekChin12138）。
4. 向我**汇报现状 + 你的执行方案**，等我确认后再动手。

## 当前进度（2026-09-28 收盘）

| 项 | 状态 |
|---|---|
| 2B Q1+Q2 | ✅ 完成，可部署 `checkpoints/q2_2b_merged`（block_hit 0.9396 / source 0.9140 / chrF 0.3922；`both_hit 0.1725` 未过门禁） |
| 0.8B Q1+Q2 @768px | ⚠️ 已有产物 `q2_08b_merged`，但**训练分辨率被 collator 静默压到 768**，数字偏低（作废为基线） |
| 0.8B Q1+Q2 @2880² | 🔄 **在云端（AutoDL RTX 4080S 32GB）跑**：`~/autodl-tmp/ocr-mt-finetuning`，日志 `logs/run_08b_full.log`，run-id `q1_08b`/`q2_08b`；用户报 Q1 稳态约 `11 s/it` → 7–8h |
| 4B | ✅ 基座已下载；本地 SFT 实测 1.74 s/sample（全程约 43h，0.2–0.3 epoch 约 9h）；GRPO/OPD 需 32GB |
| GRPO HQ-200 | ✅ 数据+脚本就绪（`data/splits_grpo_{q1,q2}_hq200/` 200/60，`train/run_grpo_hq.sh`），**未开跑** |
| OPD | ✅ TRL `GKDTrainer` VL 适配层 `train/unsloth_gkd_vl.py`，`--selfcheck` 本地已通过；**未真跑** |
| 上云 | ✅ 全量包已在云端解压；增量包 `dist/cloud_pack/grpo_hq.tar.gz`（51KB） |

## 硬约束（违反会白跑或损坏环境）

1. **不要自动开训**：任何长跑先给方案、等确认；入口命令需显式执行（脚本里有 `CONFIRM=1` 守卫）。
2. **分辨率**：`--collator-resize max`（默认）；**视觉 token ≈ `max_pixels/1024`**，
   且 `max_pixels/1024 + ~2048 ≤ max_seq_length`。**2880² → 必须 `--max-seq-length 12288`**（8192 会截掉答案）。
   **train / `eval/run_*.py` / `marker_studio` 三处 `--max-pixels` 必须一致**，否则 OOD。
3. **GRPO 必须与 SFT 同分辨率、同 prompt**；降分辨率=分布外优化（旧 GRPO 失败主因之一）。
4. 空例 GT 必须是 `""`；**不要破坏 Q1 跟点能力**（Q2 冻结 vision）。
5. 只要 `uv` + Unsloth；不用 LLaMA-Factory。
6. 训练/评测本地文件时导出 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；**下载模型**要 `HF_HUB_DISABLE_XET=1`（hf-mirror 不代理 Xet → 401）。
7. 宿主 15GB：长跑别用 `--num-workers 8 --pin-memory --persistent-workers`（会泄漏），用 `--num-workers 4 --no-pin-memory --no-persistent-workers`。
8. **别边跑边改 shell 脚本**（bash 增量读→执行错乱）：先停→改→重跑（脚本可重入，会跳过已完成阶段并从最新 checkpoint 续）。
9. 测试保持全绿；新增行为要加测试（现有 `tests/test_docs_consistency.py` 会校验 README/HANDOVER 里的路径存在）。
10. git：`user.name=DerekChin12138`、`user.email=qxc2864114982@outlook.com`；提交前 `git add -An | wc -l` 应为 **~285**（`.gitignore` 已盖住数据/权重；若出现图片/权重/语料立刻停）。
11. 汇报要**简洁**；每个阶段产出写进 `docs/` 或 `eval/results/`（别只留在对话里）。

## 下一步优先级（详见 HANDOVER §6）

- **P0**：确认云端 0.8B@2880² 跑完（`checkpoints/q2_08b/metrics/q2_report.md`），
  看 **`real_labeled` / `multi_frag` 是否比 768px 口径明显回升**（768px 时 real 0.500 / mf 0.693）→ 写 `eval/results/q08b_2880_stage_report.md`。
- **P1**：0.8B GRPO（HQ-200）
  ```bash
  KIND=q1 SIZE=08b bash train/run_grpo_hq.sh status
  KIND=q1 SIZE=08b bash train/run_grpo_hq.sh cand && KIND=q1 SIZE=08b bash train/run_grpo_hq.sh seed
  KIND=q1 SIZE=08b GEN_BATCH=4 ADAPTER=checkpoints/q1_08b/adapter_final bash train/run_grpo_hq.sh probe
  KIND=q1 SIZE=08b bash train/run_grpo_hq.sh compose
  KIND=q1 SIZE=08b bash train/run_grpo_hq.sh dryrun
  CONFIRM=1 KIND=q1 SIZE=08b bash train/run_grpo_hq.sh all
  ```
  门禁：探针产出率 ≥40%（<40% 就不做）；训练后 mf/sg/real **各 +2pp**、`empty_on_chrome ≥0.97`、`format_leak=0`，否则**回退 SFT**。
- **P2**：4B 两轮 SFT（云端 32GB，`docs/CLOUD_RUN_32GB.md`）→ 量 gap（4B-Q2 vs 0.8B-Q2 的 `source_hit`）。
- **P3**：4B GRPO，**定位优先权重**（`KIND=q2 SIZE=4b`，默认 `GRPO_REWARD_WEIGHTS="1.5,0.05,1.0,1.5,1.5,0.75,0.5"`）→ 拿到"近乎完美定位"的 teacher。
- **P4**：OPD（teacher=4B-GRPO-Q2 → 0.8B），`--mask-translation` 只蒸馏 `<source>`+XML；门禁 `source_hit +2pp`。
- **P5**：翻译单独解决（数据/两阶段/API teacher），**不要交给 RL/OPD**。

## 新工作区恢复（若这里不是原工作区）

```bash
git clone https://github.com/DerekChin12138/Point-conditioned-OCR-finetuning.git
cd Point-conditioned-OCR-finetuning
uv sync --extra dev --extra train && uv pip install unsloth
uv run pytest -q                      # 183 passed
# 不在 git 里的两样：
#   models/Qwen3.5-{0.8B,2B,4B}  → HF_HUB_DISABLE_XET=1 uv run hf download … --local-dir models/…
#   data/                        → 从云端/旧工作区拷 pools_*，或用 data/scripts 以 seed 42 重渲
# 换机后 split 里的绝对图片路径要重定位：
uv run python scripts/relocate_paths.py --old-root /home/derek_qxc/workspace/ocr-finetuning/Point-conditioned-OCR-finetuning --apply
```

## 沟通约定

- 用户偏好：**每一步先给方案再做**；长跑要说明耗时/显存/回滚方案；结果用表格给数字与门禁对比。
- 需要用户做的决定：是否开训、是否上云、是否接受精度/时间取舍（例如 Q2 epoch 从 1.0 降到 0.6）。
- 先问我：「云端那个 0.8B@2880² 跑到哪一步了？」——它决定 P0/P1 的起点。
