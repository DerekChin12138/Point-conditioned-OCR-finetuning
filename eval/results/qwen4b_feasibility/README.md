# Qwen3.5-4B 本地可行性实测（RTX 4060 Laptop 8GB / WSL2）

日期：2026-09-26。目的：回答「4B 两轮 SFT 能否在本地跑」。
脚本：`train/unsloth_stage_a.py`（与 2B/0.8B 同一入口）。权重：`models/Qwen3.5-4B`（HF `Qwen/Qwen3.5-4B`，8.8GB bf16，已下载）。

配置与 2B SFT_Q1 **完全对齐**：4bit QLoRA、r32/α64、seq 8192、batch 2 × grad-accum 8、
`max_pixels 8294400`（8.3MP）、`--finetune-vision`、`--num-workers 0 --no-pin-memory`。

---

## 1. 实测

| 测试 | 配置 | 结果 |
|---|---|---|
| **load-only** | `--smoke-load-only` | ✅ 加载成功；加载瞬时峰值 **7865 MiB**，随后回落 |
| **2 步训练** | 8 samples / 2 steps | ✅ 成功；稳态 ~5.9GB |
| **6 步训练** | 96 samples / 6 steps | ✅ 成功；**GPU 峰值 6419 MiB / 8188**，util 100%，**1.74 s/sample** |
| **merge** | 0.8B 冒烟 adapter 走 `merge_unsloth_lora.py` | ✅（4B 未单独测 merge，路径同构，纯 CPU 内存操作） |

6 步详细：

| 指标 | 值 |
|---|---|
| 每 optimizer step（16 samples） | **27.8–29.1 s**（step 1 含 Triton 编译 ~100s，不计） |
| 吞吐 | **1.74 s/sample**（2B 实测 0.78 s/sample → 4B **慢 ~2.3×**） |
| GPU 峰值 | **6419 MiB**（装载后稳态 5926–6419） |
| 宿主 RAM 峰值 | **~7.0GB used**（`available` 仍 ~8.8GB，无 swap 压力） |
| 现象 | Unsloth 打印 `Will smartly offload gradients to save VRAM`（靠梯度 offload 才装下） |

---

## 2. 结论

**能跑，但很紧、且慢。** 4B 两轮 SFT 在本地**技术可行**：

| 阶段 | samples（×epoch） | 估时 @1.74 s/sample |
|---|---|---|
| SFT_Q1（1.5 ep × 25562） | 38343 | **~18.6h** |
| SFT_Q2（1.0 ep × 49616） | 49616 | **~24.1h** |
| post-eval（test/val 生成） | ~2700 × 2 | ~2–4h |
| **合计** | | **~45h（约 2 天）** |

若按论文对 4B 分支的做法只训 **0.2–0.3 epoch**，则 Q1 ~3.7h + Q2 ~4.8h ≈ **~9h**。

### 风险

1. **显存余量只有 ~1.7GB**（6.4/8.0）。任何 `--eval-batch-size 2`、更大的 `max_pixels`、
   或 Unsloth 丢掉梯度 offload，都可能 OOM → 触发 WSL `CUDA device not ready`（会挂）。
   → 真跑时用 `--eval-batch-size 1`、`--post-eval-batch-size 1`，并先跑 50 步观察稳态。
2. **45h 长跑跨多次 WSL 重启**概率高 → 必须依赖 `--resume-from-checkpoint`（已可用）。
3. **不能做 4B GRPO / 4B-teacher OPD**：G≥4 × 8MP 的 generate 峰值远超 8GB（2B 已证伪）。
   → 这两步只能上云（24GB 级）。

### 建议

- **本地做 4B SFT**（0.2–0.5 epoch，~9–15h）是可以的，产出 4B teacher 底模；
- **4B GRPO + OPD 直接上云**（把本仓库环境打包上传，见 `docs/AUTODL.md`）。
- 本地若只想要「最省事的可靠性」，仍应先把 0.8B 干净基线跑出来（`train/run_08b_pipeline.sh`）。
