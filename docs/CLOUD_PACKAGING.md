# 把工作区打包上云（AutoDL / 任意 CUDA 主机）

日期：2026-09-28。目标：把本目录迁到云端 GPU 上继续训练（本地 8GB 显存虽能装下高保真配置，
但**时间**不划算）。配套脚本：`scripts/pack_for_cloud.sh`、`scripts/relocate_paths.py`。
弱网镜像/环境见 [`AUTODL.md`](AUTODL.md)。

---

## 0. 一分钟版

```bash
# 本地：生成清单（不打包，秒级）
bash scripts/pack_for_cloud.sh --dry-run
#   → dist/cloud_pack/code_files.txt   （代码+config+splits，~215MB）
#   → dist/cloud_pack/images.txt       （split 引用的 marked 图，81,788 张 / 20.0 GiB）

# 本地：rsync 直传（可断点续传），比传 tarball 更省事
rsync -avP --files-from=dist/cloud_pack/code_files.txt ./ root@<HOST>:/root/autodl-tmp/ptocr/ -e 'ssh -p <PORT>'
rsync -avP --files-from=dist/cloud_pack/images.txt      ./ root@<HOST>:/root/autodl-tmp/ptocr/ -e 'ssh -p <PORT>'

# 云端
cd /root/autodl-tmp/ptocr
source scripts/setup_autodl_mirrors.sh
uv sync --extra dev --extra train --extra synth
uv pip install unsloth
uv run python scripts/relocate_paths.py --old-root /home/derek_qxc/workspace/ocr-finetuning/Point-conditioned-OCR-finetuning --apply
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B     # 2B/4B 同理
uv run pytest -q && bash train/run_08b_pipeline.sh smoke
```

---

## 1. 打包什么 / 不打包什么

| 类别 | 是否上传 | 体积 | 说明 |
|---|---|---|---|
| 代码 `src train eval export scripts tests data/scripts docs notebooks` + `pyproject.toml uv.lock` | ✅ | ~10 MB | `code_files.txt` 已覆盖 |
| `data/splits_stage_q1_withreal/`、`data/splits_stage_q2_ocr_mt_v2/` | ✅ | ~215 MB | 训练/评测清单 |
| **split 引用的 marked 图**（`data/pools_q*`、`data/pools_real`） | ✅ | **20.0 GiB** | 已按 split 精确列出（池里总共 25 GiB，只传用到的） |
| `data/recipes/` | ✅ | 56 KB | compose 配比 |
| `data/synth/`（模板 + 内容池） | 可选 `--with-synth` | 1.3 GB | 只有在云端**重渲页面**时才要 |
| `models/Qwen3.5-{0.8B,2B,4B}` | ❌ | 15 GB | 云端 `hf download`（见 §4），省上行 |
| `checkpoints/` | 可选 `--with-checkpoints` | 19 GB | 只有在云端**续训/微调现有权重**时才要 |
| `.venv/`、`logs/`、`dist/`、`exports/`、`data/_archive_v1/`、`data/studio_galleries/`、`data/label_*` | ❌ | — | 无用 |
| `.env`（含 `POINT_OCR_LLM_API_KEY`） | ⚠️ 默认**不打包** | — | 只有造数/蒸馏要 API；云端按需手建 |

> **为什么是 20 GiB 而不是 34 GiB**：`data/pools_q*` 里有 99k 张图，但 Q1/Q2 split 只引用 81,788 张。
> `pack_for_cloud.sh` 按 split 精确列文件，少传 ~5 GiB。

---

## 2. 本地：生成清单

```bash
bash scripts/pack_for_cloud.sh --dry-run            # 只盘点
bash scripts/pack_for_cloud.sh --code-only --dry-run # 只要代码+split（先上云试环境）
bash scripts/pack_for_cloud.sh                       # 生成 tarball（dist/cloud_pack/）
```

产物：

| 文件 | 内容 |
|---|---|
| `dist/cloud_pack/code_files.txt` | 代码+config+splits 相对路径清单 |
| `dist/cloud_pack/images.txt` | split 引用的图片相对路径清单 |
| `dist/cloud_pack/code_and_splits_<date>.tar.gz` | 代码+split（`--dry-run` 时不生成） |
| `dist/cloud_pack/images_<date>.tar` | 图片（JPEG 已压缩，不再 gzip） |
| `dist/cloud_pack/SHA256SUMS` | 校验 |

---

## 3. 传输（三选一）

**A（推荐）rsync 按清单直传** —— 可断点续传、增量、最省事：
```bash
rsync -avP --files-from=dist/cloud_pack/code_files.txt ./ root@<HOST>:/root/autodl-tmp/ptocr/ -e 'ssh -p <PORT>'
rsync -avP --files-from=dist/cloud_pack/images.txt      ./ root@<HOST>:/root/autodl-tmp/ptocr/ -e 'ssh -p <PORT>'
```
（AutoDL 的 SSH 端口/密码在控制台；`/root/autodl-tmp` 是数据盘，容量大。）

**B）tarball + scp / 控制台上传面板**：
```bash
bash scripts/pack_for_cloud.sh
scp -P <PORT> dist/cloud_pack/*.tar* root@<HOST>:/root/autodl-tmp/ptocr_pack/
# 云端： mkdir -p /root/autodl-tmp/ptocr && tar -xf images_<date>.tar -C /root/autodl-tmp/ptocr \
#        && tar -xzf code_and_splits_<date>.tar.gz -C /root/autodl-tmp/ptocr
```

**C）代码走 git**（`data/`、`checkpoints/`、`models/` 都被 `.gitignore` 忽略，仍需 A/B 传数据）。

---

## 4. 云端：环境 → 模型 → 路径重定位 → 自检

```bash
cd /root/autodl-tmp/ptocr
source scripts/setup_autodl_mirrors.sh          # HF_ENDPOINT=hf-mirror / pip 镜像
uv sync --extra dev --extra train --extra synth
uv pip install unsloth
uv run playwright install chromium              # 只有要重渲页面时才需要
nvidia-smi

# 基座（走 hf-mirror；断网会卡住）
#   ⚠️ hf-mirror 不代理 Xet 的 CAS（xethub.hf.co）→ 不禁用 Xet 会报 401 Unauthorized：
#   RuntimeError: ... CAS Client Error ... 401 Unauthorized, domain: https://cas-server.xethub.hf.co/...
export HF_HUB_DISABLE_XET=1     # setup_autodl_mirrors.sh 已默认导出
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
hf download Qwen/Qwen3.5-2B   --local-dir models/Qwen3.5-2B
hf download Qwen/Qwen3.5-4B   --local-dir models/Qwen3.5-4B

# ⚠️ 关键：split 里存的是「本机绝对路径」，必须重定位
uv run python scripts/relocate_paths.py \
  --old-root /home/derek_qxc/workspace/ocr-finetuning/Point-conditioned-OCR-finetuning --apply
#   先不加 --apply 是 dry-run，会打印每个文件改了多少行

# 自检
uv run pytest -q                                 # 应 157 passed
bash train/run_08b_pipeline.sh smoke             # 8 samples / 2 steps
```

开训（注意新的分辨率默认）：
```bash
# 默认 MAX_PIXELS=2048²(4.19MP)、MAX_SEQ_LENGTH=8192、COLLATOR_RESIZE=max
CONFIRM=1 NUM_WORKERS=4 bash train/run_08b_pipeline.sh all

# 最大保真（2880²，需更长的 max_seq）
CONFIRM=1 MAX_PIXELS=8294400 MAX_SEQ_LENGTH=12288 NUM_WORKERS=8 bash train/run_08b_pipeline.sh all

# 大卡上可以直接用满：MAX_SEQ_LENGTH=16384 等
```
长跑守护：`scripts/supervise_08b.sh`（自愈续训）+ `scripts/watch_08b_training.sh`（显存/内存看门狗）。

---

## 5. 上云后必查清单

- [ ] `relocate_paths.py` dry-run 显示已无待改行（或已 `--apply`）。
- [ ] `uv run pytest -q` → 157 passed。
- [ ] `smoke` 能跑，且日志里 `collator_resize=max`、`~N vision tokens/image` 符合预期。
- [ ] 抽 1 条 split，确认 `images[0]` 指向的文件存在：
      `uv run python -c "import json;r=json.loads(open('data/splits_stage_q2_ocr_mt_v2/val.jsonl').readline());import os;print(r['images'][0], os.path.exists(r['images'][0]))"`
- [ ] `nvidia-smi` 显存/驱动正常；GRPO/OPD 在 ≥24GB 卡上再开。

---

## 6. 云端大卡带来的新可能（本地做不了/不划算）

| 任务 | 本地 8GB | 云端 24GB+ |
|---|---|---|
| 0.8B SFT | ✅ 可跑，但 2880² 高保真很慢（见下） | ✅ 快很多 |
| 2B SFT | ✅ 显存够（2880²/12288 峰值 ~6.7GB） | ✅ |
| 4B SFT | ✅ 勉强（4bit，峰值 ~6.4GB） | ✅ 舒服 |
| **GRPO（4B 分支）** | ❌ 8.3MP×G≥4 会 `make_resident ENOMEM` | ✅ 这是 OvisOCR2 论文的正路 |
| **OPD（4B→0.8B，top-k reverse-KL）** | ❌ 需 teacher logits + 自研 trainer | ✅ |

本地实测（batch 2，4bit QLoRA，**最密页**——最坏情况）：

| 配置 | 视觉 token | peak reserved | 参考速度 |
|---|---|---|---|
| 0.8B 2880²/12288 | 7854 | ~5.8 GB | ~23.8 s / step（≈0.084 samp/s，最坏页） |
| 2B 2048²/8192 | 4015 | 4.6 GB | ~20 s / step（≈0.099 samp/s） |
| 2B 2880²/12288 | 7854 | 6.7 GB | 未测完（用户中止） |

> 高保真的显存能装下，但**本地时间成本是小时→天的量级**，所以上云是对的。
> 具体时长要在云端用真实数据分布重测（最密页只是上界）。

---

## 7. 注意事项

1. **绝对路径**：`data/splits_*/**.jsonl` 的 `images[0]` 是构建机的绝对路径；换机必跑 `relocate_paths.py`。
   （更彻底的方案是让 loader 支持相对路径，尚未做。）
2. **分辨率与序列必须匹配**：`视觉 token ≈ max_pixels/1024`，规则 `max_pixels/1024 + ~2048 ≤ max_seq_length`。
   训练与评测的 `--max-pixels` 要一致（trainer / `eval/run_*.py` / studio 默认都已对齐到 2048²）。
3. **`.env` 不入包**：会用到 API 的只有造数/蒸馏；云端手写 `.env`（格式见 `.env.example`）。
4. **HF 可达性**：`from_pretrained` 会校验 Hub 文件；保持 `HF_ENDPOINT=https://hf-mirror.com`，
   必要时 `HF_HUB_DISABLE_XET=1`。
5. **续训**：本地 `checkpoints/` 不传的话云端从零训；要接着现有权重就得 `--with-checkpoints`
   （约 19 GB；也可以只挑 `checkpoints/q2_08b_merged` 等）。
