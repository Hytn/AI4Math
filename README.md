<p align="center">
  <img src="https://img.shields.io/badge/Lean-4.24.0-blue" alt="Lean 4" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab" alt="Python" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
</p>

# AI4Math — 让 DeepSeek-Prover-V2-7B 跑得比论文更高

一套 **Lean 4 形式化定理证明的 agent 框架**。它的核心命题不是"拼一个更大的模型", 而是**在同一个模型上拼方法学**:

```
         论文 (DeepSeek-Prover-V2, 2504.21801)            本框架
    ─────────────────────────────────────────────  ─────────────────────────────
       miniF2F-test 7B pass@8192 = 82.0%             同一个 7B 模型, 你可以叠加:
                  ↑                                    ① verify-and-fix 反馈循环
       单一 profile (whole-proof CoT)                  ② 跨题 lemma bank + dialog index
       i.i.d. 独立采样, sample 之间无信息流            ③ 4 路异构并行 + broadcast bus
                                                       ④ MCTS / best-first / beam 探索调度
                                                       ⑤ sklearn / Qwen 世界模型 tactic gating
                                                       ⑥ 19 个 profile 一键 A/B 切换
```

如果 ①+②+③ 任意组合在同 sample 预算下能赢过论文 baseline, 这个框架就证明了存在意义。这正是 **`reproduce_minif2f_7b_ablation.sh`** 设计出来要测的事。

---

## TL;DR

```bash
# 0. 5 分钟冒烟 — 没装 Lean, 没 API key 也行
bash reproduce_minif2f.sh --smoke

# 1. 用 vLLM 部署 DeepSeek-Prover-V2-7B (1×80GB H100), 然后:
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 \
    --samples 32

# → 跑出 5 档 profile 的 pass@k 对照表, 自动对比论文 Table 1。
```

---

## 前置依赖 — 你需要什么

| 资源 | 何时需要 | 说明 |
|---|---|---|
| Python ≥ 3.10 | 永远 | 任何阶段 |
| Linux/macOS | 永远 | Windows 用 WSL2 |
| 1×80GB H100 (or 2×40GB) | 跑 7B 真实评测 | vLLM 部署 DSP-V2-7B; 8×H100 才能跑 671B |
| 网络 | 首次 setup | 拉数据集 + Mathlib 编译 |
| Lean 4 v4.24.0 + Mathlib | `--lean` 真实评测 | miniF2F 锁定版本; 首次 `lake build` 30-60 分钟 |
| 50GB 磁盘 | `--lean` 真实评测 | Mathlib oleans + Lean toolchain |
| `ANTHROPIC_API_KEY` 等 | 用通用 LLM 做对照实验 | DSP-V2 路线不需要 |

**冒烟测试 (`--smoke`) 不需要以上任何资源**, 只要 Python。

---

## 完整线性流程 — 新工程师从零到出数字

如果你从未见过这个项目, 按这 7 步顺序执行, 不要跳:

```bash
# ─── Step 1: 拿到代码, 装 Python 依赖 ─────────────────────────────────
git clone <this-repo>
cd AI4Math-v17
pip install -r requirements.txt

# ─── Step 2: 冒烟, 验证 Python 管线没坏 (3 分钟, 无外部依赖) ──────────
bash reproduce_minif2f.sh --smoke
# 预期: 5/5 通过, 报表标 [unverified]。这一步只验证代码能跑, 不验证证明对错。

# ─── Step 3: 装 Lean 4 v4.24.0 (一次性, 后续都复用) ─────────────────
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain leanprover/lean4:v4.24.0
source "$HOME/.elan/env"
lean --version    # 应该显示 v4.24.0

# ─── Step 4: 在 miniF2F 项目里编 Mathlib (30-60 分钟, 一次性) ─────────
# 让 reproduce_minif2f.sh 把数据拉下来
bash reproduce_minif2f.sh --smoke
# 然后编译
cd data/miniF2F
lake exe cache get   # 拉 mathlib 预编译 cache, 节省 30 分钟
lake build           # 应该 5-10 分钟而不是 60
cd ../..

# ─── Step 5: 部署 DeepSeek-Prover-V2-7B (vLLM, 单 H100) ──────────────
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \
    --max-model-len 32768 --port 8000 \
    --served-model-name DSP-V2-7B &
# 等 ~2 分钟模型加载完, 验证:
curl http://localhost:8000/v1/models     # 应该看到 DSP-V2-7B

# ─── Step 6: 复现论文 baseline (健康检查, 1-3 小时) ──────────────────
# 先跑论文 7B CoT pass@32 = 75.6% 那一行, 验证你的部署 + 框架接线没问题
bash reproduce_minif2f.sh \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B \
    --profile dsp_v2_cot \
    --samples 32 --temperature 1.0 --lean
# 预期: pass@32 落在 70-80% 之间。如果 < 60%, 跳到下面"故障排查"。

# ─── Step 7: 跑全 5 档 ablation (主菜, 5-15 小时) ────────────────────
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 \
    --samples 32
# 出最终对照表。哪一档赢过 dsp_v2_cot 的 75.6%, 就是框架对论文的真实增量。
```

每一步独立可重跑。Step 4 和 Step 5 是一次性的, 之后调参/换 sample 数, 只重跑 Step 6/7。

---

## 术语速查

| 概念 | 在框架的哪个文件 | 简单解释 |
|---|---|---|
| **Profile** | `prover/unified/profiles.py` | 一个完整的"证明算法定义" — tools + max_turns + framing + temperature 的具名 bundle。换方法 = 换 profile |
| **Framing** | `prover/unified/system_prompts.py` | LLM 的 system prompt 模板。`dsp_v2_cot` framing 是论文 Appendix A.2 verbatim |
| **Backend** | `engine/backends/` | Lean 4 验证器实现 (local subprocess / Kimina HTTP / Pantograph / Mock) |
| **dialog.json** | 输出文件 | 一道题的完整轨迹: messages, tool calls, results, success, proof. 一题一文件 |
| **AsyncLeanPool** | `engine/async_lean_pool.py` | 并行 Lean REPL 池 (默认 4 路), `--pool-size` 调 |
| **Knowledge store** | `knowledge/store.py` | 跨题持久化的 SQLite (lemma 有效性、dialog snippet、tactic 统计) |
| **Lemma bank** | `prover/lemma_bank/` | 跨题引理库, 用 BM25 检索, 把以前题里证过的 helper 复用到当前题 |
| **Dialog index** | `knowledge/dialog_index.py` | 跨题成功 dialog 检索, 当 in-context demo 注入下一题 |
| **Broadcast bus** | `engine/broadcast.py` | 异构并行 4 路 sub-runner 间的发现总线 |
| **Pass@k** | `benchmarks/metrics.py` | 论文标准指标 (无偏估计): k 个 i.i.d. sample 中至少一个对的概率 |

---

## 这份 README 解决的问题

> "我手上有 DeepSeek-Prover-V2-7B 的权重 / vLLM 部署。论文里它在 miniF2F-test 上 pass@8192 = 82.0%。我想用这个框架做出比论文更高的数字。"

这条路径明确, 工程化, 不需要赌新模型权重:

1. **复现论文 baseline**: 用 `dsp_v2_cot` profile + 论文 Appendix A.2 原 prompt + temperature 1.0, 在你的 vLLM 部署上跑 pass@32, 应该接近论文报的 75.6%。这是健康检查。
2. **叠加 ①: 反馈循环**: 切到 `dsp_v2_repair`. 论文的采样是 i.i.d. 独立的, 一次失败就丢; 这个 profile 把 Lean 编译错误回灌给同一个 7B 模型让它重写。**对"近正确证明"(策略对、引理名错)的命中率提升明显**。
3. **叠加 ②: 跨题知识沉淀**: 切到 `dsp_v2_repair_knowledge`. 第 50 题 时已经看过前 49 题成功的 lemma 和 dialog snippet, 当作 in-context demo 注入。
4. **叠加 ③: 异构并行**: 切到 `dsp_v2_heterogeneous`. 4 路 sub-profile 用不同 framing/温度同时跑同一道题, 任一路成功即整 sample 成功, broadcast bus 在 sub 之间共享发现。
5. **跑 A/B sweep**: `reproduce_minif2f_7b_ablation.sh` 一键跑全 5 档, 自动对照论文 Table 1 打表。**任意一档赢过 baseline, 框架的价值就被证明了**。

---

## 0. 5 分钟冒烟 (无 Lean / 无 API key)

```bash
git clone <this-repo>
cd AI4Math-v17
pip install -r requirements.txt

bash reproduce_minif2f.sh --smoke
```

预期输出:

```
═══ Step 1/5  Python + 依赖检查 ═══
[ OK ] Python 3.10+
[ OK ] 依赖就绪
═══ Step 2/5  miniF2F-lean4 数据集 ═══
[ OK ] miniF2F-test 已就绪: 244 道 .lean 文件
...
  miniF2F-test  (unverified)
  ─────────────────────────────
  Solved      : 5/5  (100.0%)
  pass@1      : 1.0000

  ⚠ 这次跑的是 --lean-mode=skip; 数字只表示 LLM 输出了非空、无 sorry 的代码,
    没有 Lean 4 编译器认可。要真实数字请加 --lean.
```

**这个 100% 不是论文意义上的通过率**, 它只表示管线是通的:
- 244 题数据集加载正常
- mock LLM 返回了非空代码
- prefilter (语法检查) 没拦下来
- 写到 `results/minif2f_run_*/traces/minif2f/<id>/dialog.json` 没崩

要拿真实数字, 必须 (1) 装 Lean 4 + Mathlib, (2) 接真实 LLM。下面分段讲。

---

## 1. 部署 DeepSeek-Prover-V2-7B (vLLM, 1×80GB H100)

### 1.1 起 vLLM 服务

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --port 8000 \
    --served-model-name DSP-V2-7B
```

可以用任何 OpenAI 兼容的部署: `--provider sglang`, `--provider ollama`, 自家的 OpenAI 兼容代理等。本框架只走 OpenAI Chat Completions 协议。

如果你想跑论文里更强的 671B 版本, 把 `--tensor-parallel-size 1` 改成 `8` (需要 8×80GB 集群), 模型 ID 换成 `deepseek-ai/DeepSeek-Prover-V2-671B`。其余命令一字不变。

### 1.2 装 Lean 4 + Mathlib (miniF2F 锁的是 v4.24.0)

```bash
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain leanprover/lean4:v4.24.0
source ~/.elan/env

# 让 reproduce_minif2f 的 step 2 把数据拉下来
bash reproduce_minif2f.sh --smoke

# 在 miniF2F 项目内 build mathlib (首次约 30-60 分钟)
cd data/miniF2F
lake exe cache get
lake build
cd ../..
```

或者用 docker 镜像 (避免 elan 版本踩坑, 但需要把容器内 Lean 通过 socket 暴露给框架; 见 `docker/lean_daemon.py` 文档, 不在本 README 范围内):

```bash
# 高级用户路径, 默认推荐 elan
docker build -t ai4math-lean -f docker/Dockerfile.lean docker/
```

### 1.3 跑论文 baseline

先复现论文 Table 1 的 7B CoT pass@32 = 75.6%, 验证你的 vLLM 部署 + 框架接线没问题:

```bash
bash reproduce_minif2f.sh \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B \
    --profile dsp_v2_cot \
    --samples 32 \
    --temperature 1.0 \
    --lean
```

**关键**: `--temperature 1.0`。论文用的就是这个; 默认 `dsp_v2_*` profile 也是 1.0, 但如果你切成别的 profile, 默认 0.7 在 pass@k 上会显著低于论文。

**预期**: 你的 pass@32 应该落在 70-80% 之间。如果远远低于 70%, 检查:
- vLLM 是否真在跑 7B (HF id 拼对了吗?)
- 温度是否真的 1.0 了 (看日志 `[unified] starting profile='dsp_v2_cot', max_turns=1` 后面)
- Lean 是否真在编译 (`dialog.json` 的 `meta.backends.lean_pool.is_fallback` 应该是 `false`)

---

## 2. 一键 A/B sweep — 5 档 profile, 一张对照表

```bash
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 \
    --samples 32
```

跑完产出:

```
═══ A/B 对比 — DeepSeek-Prover-V2-7B 在 miniF2F-test ═══

  论文 baseline (arXiv:2504.21801, Table 1, 7B 行):
    non-CoT pass@1=55.5%, pass@32=68.0%, pass@1024=73.2%, pass@8192=75.0%
    CoT     pass@1=58.6%, pass@32=75.6%, pass@1024=79.9%, pass@8192=82.0%

  本次 sweep 的 5 档 (--samples=32):

  Profile                         Solved   pass@1   pass@32
  ──────────────────────────────────────────────────────────
  dsp_v2_non_cot                  ???/244  ?.????  ?.????
  dsp_v2_cot                      ???/244  ?.????  ?.????   ← 论文对照点
  dsp_v2_repair                   ???/244  ?.????  ?.????   ← 增量1
  dsp_v2_repair_knowledge         ???/244  ?.????  ?.????   ← 增量2
  dsp_v2_heterogeneous            ???/244  ?.????  ?.????   ← 增量3
```

**问题**: 哪一档的 pass@32 ≥ 论文的 pass@32 = 75.6%?
**问题**: 进一步加 sample 到 1024, 哪一档能突破论文 pass@1024 = 79.9%?

这就是这个项目的命题。它没法替你回答, 只能帮你**在同一个 7B 模型上跑这个对照**, 让数字自己讲话。

每档跑完, dialog 累积写到同一个 `results/dsp_v2_7b_kb/main.sqlite` (lemma bank + dialog index)。`dsp_v2_repair_knowledge` 和 `dsp_v2_heterogeneous` 会读取这份 KB —— 因此**跑 ablation 的顺序很重要**: 知识从 non_cot → cot → repair 累积起来, 给后两档当 in-context demos 用。脚本默认就按这个顺序跑。

### 2.1 增量加 sample (先 32 看趋势, 再 1024 求精)

ablation 脚本默认 `--resume`。把 `--samples` 改大重跑同一个 `--root`, 已完成的题会跳过, 只补缺的:

```bash
# 先 32, 看哪一档有希望
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 \
    --samples 32 \
    --root results/abl_run1

# 看完 32 的表, 把希望最大的 profile 单独加到 1024:
python run_eval.py \
    --benchmark minif2f --profile dsp_v2_repair \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B --temperature 1.0 \
    --max-samples 1024 --lean-mode real \
    --project-dir data/miniF2F \
    --output-dir results/abl_run1/dsp_v2_repair \
    --knowledge-db results/dsp_v2_7b_kb/main.sqlite \
    --dialog-index  results/dsp_v2_7b_kb/main.sqlite \
    --lemma-bank-db results/dsp_v2_7b_kb/lemmas.sqlite \
    --resume
```

### 2.2 预算估算 (重要 — 别直接跑 8192)

DSP-V2-7B 在单 H100 vLLM 上 ~50 tok/s, 每个 CoT proof ~4500 tok。244 题:

| Sample 预算 | 总 token | 单 H100 时长 |
|---:|---:|---:|
| pass@32 | ~3.5 亿 | ~2 小时 |
| pass@128 | ~14 亿 | ~8 小时 |
| pass@1024 | ~110 亿 | ~64 小时 |
| pass@8192 | ~880 亿 | ~22 天 |

实际 vLLM 因为 KV cache 复用、batch 流水, 吞吐会高 3-5×。但**先跑 pass@32 看每档的相对排序**, 比一上来就 pass@8192 实用得多。

`heterogeneous` profile 因为 4 路并行, sample 预算实际等价于其他 profile 的 4×, 所以 ablation 脚本里它的 `--samples` 数实际意义是 "每路 sub-profile 的采样数"。

---

## 3. 19 个 Profile —— 换方法只改 `--profile`

| Profile | 对应论文 / 方法学 |
|---|---|
| **DSP-V2 专用 (v17 新增)** | |
| `dsp_v2_non_cot` | 论文 Appendix A.1 verbatim, 单次出整证, 无反馈 |
| `dsp_v2_cot` | 论文 Appendix A.2 verbatim, proof plan + 整证, **论文 7B SOTA = 82.0%** |
| `dsp_v2_repair` | + verify-and-fix 反馈循环 (本框架增量 #1) |
| `dsp_v2_repair_knowledge` | + 跨题 lemma bank + dialog index (增量 #2) |
| `dsp_v2_heterogeneous` | + 4 路异构并行 + broadcast bus (增量 #3) |
| **通用** | |
| `whole_proof` | DeepSeek-Prover / Kimina / Goedel 风格, 单次整证 |
| `whole_proof_repair` | 通用 LLM (Claude/GPT) 默认主路径, 编译反馈循环 |
| `dsp` | Draft-Sketch-Prove (Jiang 2023) |
| `reprover` | ReProver 风格 RAG + step-level |
| `leandojo` | 纯 step-level, 一次 apply 一个 tactic |
| `heterogeneous` | 通用 4 路异构并行 |
| `conjecture_driven` | 主动猜辅助引理 (PutnamBench/FATE-X 类难题) |
| `kimina_batch` | Kimina Lean Server 批量 |
| `pantograph_dsp` | Pantograph mvar focus + drafting |
| `lookeng_lemma` | LooKeng stateless lemma-by-lemma |
| `nfl_hybrid` | NFL-HR (Yao et al., EMNLP 2025) |
| `mcts` | MCTS-UCB1 树搜索 |
| `best_first` | best-first 树搜索 |
| `beam` | beam search |

每个 profile 配置在 `prover/unified/profiles.py::PRESETS` (权威源), `config/profiles/<name>.yaml` 是从代码 dump 的视图, 可以编辑后用 `--profile-yaml` 加载。

---

## 4. 每个增量分别加在哪里 — 框架内部接线

下面这张表说明你切 profile 时, 框架背后实际改了什么。这也是给想"再叠一层 trick"的研究者看的扩展点说明。

| 增量 | 在框架的哪一层实现 | 谁可以打开 |
|---|---|---|
| **verify-and-fix 反馈循环** | `agent/runtime/agent_loop.py` + `ObservationPolicy.auto_inject_lean_compile=True` | 任何含 `LEAN_VERIFY` tool 的 profile, 设 `max_turns > 1` |
| **跨题 lemma bank** | `prover/lemma_bank/` (BM25 + SQLite) | `--lemma-bank-db <path>` + profile 含 `LEMMA_BANK` tool |
| **跨题 dialog index** | `knowledge/dialog_index.py` (TF-IDF + SQLite) | `--dialog-index <path>` + `ObservationPolicy.inject_similar_dialogs=True` |
| **跨题 premise pool** | `prover/premise/selector.py` (hybrid: BM25 + char n-gram) | 默认开; 用 `scripts/export_mathlib_premises.py` 扩到 10⁵ 量级 |
| **knowledge briefing** | `knowledge/reader.py` (按 domain 抽 top-N tactic 有效性统计) | `--knowledge-db <path>` + `ObservationPolicy.include_knowledge_briefing=True` |
| **异构并行 + broadcast bus** | `engine/broadcast.py` + `SearchConfig.kind="parallel"` | `parallel_profiles=[...]` 写在 SearchConfig 里 |
| **MCTS-UCB1 / best-first / beam** | `engine/search/` 一份代数, prover 与 sampler 共用 | `SearchConfig.kind` ∈ `{ucb, best_first, beam}` |
| **sklearn 世界模型 (tactic gating)** | `engine/world_model.py` + `scripts/train_world_model.py` | `--world-model <pickle>`, 步级 profile 才生效 |
| **Qwen / GPT 世界模型 (前瞻)** | 需要替换 `engine/world_model.py::predict()` 的 backend | 见 `engine/world_model_trainer.py` 的 fit 接口 |
| **PolicyEngine 5 条声明式规则** | `engine/policy/` (early-terminate / 切策略) | `--policy-engine` |
| **领域 plugin (按定理领域注入 prompt)** | `plugins/strategies/*/plugin.yaml` | `--plugins-dir plugins/strategies` |

要叠新 trick: 改一个 dataclass + 一段 prompt 模板, **不动 runner.py / agent_loop.py / tool_kits.py**。详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

---

## 5. 选哪个 Lean 后端 —— `--backend`

| 后端 | 何时选 | 命令 |
|---|---|---|
| `auto` | 默认, 自动探测 local/socket/http | `--backend auto` |
| `local` | 本机有 Lean 4 + Mathlib | `--backend local` |
| `socket` | 已用 `docker/lean_daemon.py` 起进程池 | `--backend socket` |
| `kimina` | 用 Kimina Lean Server (社区) | `--backend kimina --backend-url http://kimina:8000` |
| `pantograph` | 需要 mvar focus / drafting | `--backend pantograph` |
| `lookeng` | 长证明 (PutnamBench / FATE-X) I/O 优化 | `--backend lookeng` |
| `mock` | 完全离线冒烟, 所有 verify 走脚本 | `--backend mock` |

**怎么判断 backend 真的在用还是降级**: 看 `dialog.json` 的 `meta.backends.<name>.is_fallback`。`true` 表示实际没在跑 Lean (`pass@k` 是 `[unverified]`)。

---

## 6. 选哪个 LLM —— `--provider` + `--model`

| Provider | 命令 | API key 环境变量 |
|---|---|---|
| `vllm` (DSP-V2-7B/671B 自托管) | `--provider vllm --model DSP-V2-7B --api-base http://localhost:8000/v1` | (本地无 key) |
| `sglang` / `ollama` | 同上, 端口不同 | (本地无 key) |
| `anthropic` | `--provider anthropic --model claude-opus-4-5` | `ANTHROPIC_API_KEY` |
| `openai` | `--provider openai --model gpt-4o-mini` | `OPENAI_API_KEY` |
| `deepseek` | `--provider deepseek --model deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| `openai_compat` | 任何 OpenAI 兼容端点 + `--api-base` | (按服务) |
| `mock` | 离线冒烟 | (无) |

**重要**: `dsp_v2_*` profile 的 framing 是论文 Appendix A 的 verbatim prompt, 只对 DeepSeek-Prover-V2 模型有意义。给 Claude / GPT 用, 改成 `whole_proof_repair`。

---

## 7. 命令速查

```bash
# 单题 dry-run, 看完整 dialog
python run_unified.py --builtin nat_add_comm \
    --profile dsp_v2_repair \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B --temperature 1.0 --lean

# 复现论文 7B CoT pass@32 baseline
python run_eval.py \
    --benchmark minif2f --profile dsp_v2_cot \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B --temperature 1.0 \
    --max-samples 32 --lean-mode real \
    --project-dir data/miniF2F

# 一键跑全 5 档 ablation
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 --samples 32

# 加上世界模型 + plugin + policy engine 的"完整产品线"配置
python run_eval.py \
    --benchmark minif2f --profile dsp_v2_heterogeneous \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B --temperature 1.0 \
    --max-samples 32 --lean-mode real \
    --project-dir data/miniF2F \
    --knowledge-db results/dsp_v2_7b_kb/main.sqlite \
    --dialog-index results/dsp_v2_7b_kb/main.sqlite \
    --lemma-bank-db results/dsp_v2_7b_kb/lemmas.sqlite \
    --plugins-dir plugins/strategies \
    --policy-engine \
    --output-dir results/full_stack_run
```

| Flag | 用途 |
|---|---|
| `--profile NAME` | 19 个 profile 之一, 见上表 |
| `--temperature T` | override profile 默认 (DSP-V2 用 1.0, 通用 LLM 用 0.7) |
| `--max-turns N` | override profile 默认 max_turns |
| `--max-samples K` | pass@K 的 K |
| `--project-dir DIR` | **关键**: Lean REPL 在哪里启动, 自动按 benchmark 推断, 一般不用手填 |
| `--pool-size N` | Lean REPL 并行池大小 (默认 4) |
| `--knowledge-db <path>` | 跨 eval run 共享同一个 KB (A/B sweep 必用) |
| `--dialog-index <path>` | 跨题成功 dialog 检索, 自动注入 prompt |
| `--lemma-bank-db <path>` | 跨题引理库, BM25 检索辅助引理 |
| `--world-model <path>` | sklearn world-model, 做 tactic gate (步级 profile) |
| `--plugins-dir <dir>` | 领域插件; 按定理领域注入 few-shot/premises/strategy hint |
| `--policy-engine` | 启用声明式 PolicyEngine (5 条规则) |

---

## 8. RL 飞轮一键跑

ablation 跑完, 你手上会有几千份 dialog (成功 / 失败 / repair 轨迹)。这是免费的 SFT/RL 训练数据。

```bash
python scripts/rl_pipeline.py iter \
    --iter-dir results/rl/iter_0 \
    --profile dsp_v2_repair_knowledge \
    --benchmark minif2f \
    --provider vllm --api-base http://localhost:8000/v1
```

四阶段:
1. **eval** — 跑评测, 产出每题 dialog.json
2. **collect** — 走完所有 dialog, 导一份 SFT-ready jsonl
3. **train_wm** — 从成功证明里抽步级特征, 训 sklearn 世界模型
4. **train_llm** — 给外部 trainer (TRL / verl / slime) 提供输入

每阶段独立可重跑。

---

## 9. 加一个新方法 (扩展点)

```python
# 1. prover/unified/profiles.py
PRESETS["my_method"] = Profile(
    name="my_method",
    tools=[ToolKit.LEAN_VERIFY, ToolKit.PREMISE_SEARCH],
    max_turns=8,
    framing="my_framing",
    temperature=1.0,
)

# 2. prover/unified/system_prompts.py
_FRAMINGS["my_framing"] = "You are a Lean 4 prover. Output..."

# 3. python run_unified.py --profile my_method
```

`runner.py`、`agent_loop.py`、`tools/` 一行不动。

---

## 10. 目录布局

```
engine/      Lean 4 REPL 池, 三级验证, 错误智能层
  search/    搜索代数 (prover 与 sampler 共用)
  policy/    声明式规则引擎
  broadcast/ 异构并行的发现总线
agent/       AgentLoop, tools, brain (LLM), persistence (dialog.json)
prover/      Profile 驱动 runner; conjecture/formalize/decompose/repair
  unified/   PRESETS 权威源 + system_prompts + tool_kits
  lemma_bank/  跨题引理库 (BM25 + SQLite)
  premise/   Mathlib 引理检索
knowledge/   SQLite 知识库 + DialogIndex
sampler/     RL trajectory 采样 (verl / slime / TRL adapter)
benchmarks/  miniF2F + PutnamBench + ProofNet + FATE + FormalMATH 加载器
config/      default.yaml + 19 个 profile YAML 模板
data/        基准题目 (1631 道)
plugins/     YAML-driven 领域插件
docs/        ARCHITECTURE.md + dialog.json schema
tests/       853 单元 / 集成测试

reproduce_minif2f.sh                   ← miniF2F-test 复现脚本 (单 profile)
reproduce_minif2f_7b_ablation.sh       ← 5 档 profile A/B sweep + 自动对照论文
eval.sh                                ← 通用 eval 入口 (covering 全 7 benchmark)
run_unified.py                         ← 单题 CLI
run_eval.py                            ← 批量 CLI
```

读代码从 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 开始, 零基础数学家从 [`TUTORIAL_CN.md`](TUTORIAL_CN.md) 开始。

---

## 11. 诚实声明 — 这个项目不做的事

* **它不会让一个不能证 Lean 的 LLM 突然能证**。框架是杠杆, 不是发动机。模型权重是发动机。
* **`pass@k` 在 `--lean-mode skip` 下没有意义**。每份 eval 报表都会标 `[unverified]` —— 不要把它当成"通过率"。
* **"框架增量胜过论文 baseline"是个实证假设, 不是定理**。这就是为什么有 `reproduce_minif2f_7b_ablation.sh` —— 它是**实验工具**, 不是结论。某档增量在 miniF2F 上不奏效, 也是有价值的负面发现。
* **"世界模型"目前是 sklearn LogisticRegression**。可训, 作 tactic 有效性预测器够用 — 不是架构图暗示的"完整状态动力学网络"。要换 Qwen/GPT 当世界模型, 改 `engine/world_model.py::predict()` 一个文件即可, 接口已经留好。
* **`reprover` profile 的检索是 BM25 + char n-gram TF-IDF, 不是 dense neural retriever**。要换成 SBERT/ColBERT 改一个文件即可。
* **`mathlib_core.jsonl` 默认只有 ~334 条**, 真实 Mathlib4 是 10⁵ 量级。评测前请用 `scripts/export_mathlib_premises.py` 扩池。
* **`conjecture_driven` profile 让 LLM 在证一道已给定定理时自己发明辅助引理, 但不发明定理本身**。1631 道题都来自 7 个内置基准。

---

## 12. 故障排查

### 12.1 `bash reproduce_minif2f.sh --smoke` 都跑不过

| 症状 | 大概率原因 | 解决 |
|---|---|---|
| `Python 3.10+` 报错 | 系统 Python 太旧 | 装 conda/pyenv 后再试 |
| `ModuleNotFoundError` | 依赖没装上 | `pip install -r requirements.txt --break-system-packages` |
| 数据集 git clone 失败 | 网络不通 GitHub | 配代理或用镜像 |
| 244 题数据集只发现 < 240 题 | clone 不完整 | `rm -rf data/miniF2F && bash reproduce_minif2f.sh --smoke` |

### 12.2 `lake build` 卡住或失败

| 症状 | 大概率原因 | 解决 |
|---|---|---|
| `lake: command not found` | elan 没在 PATH | `source ~/.elan/env` 或重启 shell |
| `lean --version` 不是 v4.24.0 | 装错 toolchain | 在 `data/miniF2F/` 内 `elan override set leanprover/lean4:v4.24.0` |
| `lake build` 跑 60 分钟还没完 | 没用 cache get | 中断, `lake clean && lake exe cache get && lake build` |
| `error: failed to compile module Mathlib.Foo` | toolchain 错配 | 检查 `lean-toolchain` 文件内容是否就是 `leanprover/lean4:v4.24.0` |

### 12.3 vLLM 部署后, baseline pass@32 远低于论文 75.6%

按可能性从高到低检查:

1. **温度不是 1.0**。打开 dialog.json 看 `messages[].metadata.temperature`, 应该 = 1.0。如果是 0.7, 你用了 `whole_proof_repair` 而不是 `dsp_v2_cot`, 或者忘了 `--temperature 1.0`。
2. **Lean 没真在跑**。打开任一题的 `dialog.json`, 看 `meta.backends.lean_pool.is_fallback`。如果是 `true`, REPL 起不来 — 检查 `data/miniF2F/.lake/build/bin/repl` 是否存在; 若不存在, 重跑 step 4。
3. **vLLM 跑的不是 7B 而是 default 的某个小模型**。`curl http://localhost:8000/v1/models` 看实际加载的模型名。
4. **`max-model-len` 太小**, CoT 输出被截断。论文 7B CoT 平均 4488 tok, 留余量到 32768。
5. **prompt 不是论文 verbatim**。grep 验证: `python -c "from prover.unified.system_prompts import _FRAMINGS; print(_FRAMINGS['deepseek_prover_v2_cot'])"` 应该开头是 `Complete the following Lean 4 code:`。
6. **temperature = 1.0 + 模型本身没充分微调**。如果 1-5 都对, 你撞上的可能就是模型在你的 deployment 下的真实下限。同部署改跑 `dsp_v2_repair` 看会不会回到 75% 以上。

### 12.4 Ablation 跑了一半某档失败

```bash
# ablation 脚本默认 --resume, 直接重跑同一个 --root, 已完成的题会跳过
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 \
    --samples 32 \
    --root results/dsp_v2_7b_ablation_<timestamp>   # ← 用上次的 root
```

单档想单独继续:

```bash
python run_eval.py --benchmark minif2f \
    --profile dsp_v2_repair --provider vllm \
    --api-base http://localhost:8000/v1 --model DSP-V2-7B \
    --temperature 1.0 --max-samples 32 --lean-mode real \
    --project-dir data/miniF2F \
    --output-dir results/dsp_v2_7b_ablation_<timestamp>/dsp_v2_repair \
    --knowledge-db results/dsp_v2_7b_kb/main.sqlite \
    --resume
```

### 12.5 看每题到底发生了什么

```bash
# 找一道失败的题
ls results/<run>/traces/minif2f/ | head
cat results/<run>/traces/minif2f/<problem_id>/dialog.json | jq '
{
  problem: .problem_name,
  success: .result.success,
  proof: .result.successful_proof[:200],
  turns: (.messages | length),
  errors: [.messages[] | select(.role=="tool") | .content[:200]]
}'
```

`dialog.json` 是这个项目的"飞行记录仪", 任何评测都该从这里开始 debug。

### 12.6 想让评测早停, 不跑完 244 题

```bash
# 加 --limit N
bash reproduce_minif2f.sh ... --limit 20
```

也可以直接 `Ctrl+C`, 已写盘的 dialog 都会被下次 `--resume` 复用。

---

## 13. v17 改动总结

| 改动 | 文件 | 效果 |
|---|---|---|
| 修 `correct_count` 双计数 bug | `run_eval.py` | pass@k 数字之前会偏高一点, 现在准确 |
| 加 `--project-dir` + 自动按 benchmark 推断 | `run_eval.py`, `eval.sh` | `--lean-mode real` 之前对 miniF2F **会全静默失败**, 现在正确指向 `data/miniF2F` |
| 加 `--pool-size`, `--temperature`, `--max-turns` CLI | `run_eval.py`, `run_unified.py`, `eval.sh` | 不动 profiles.py 也能 override |
| 加 `--knowledge-db` CLI | `run_eval.py` | 多个 eval run 可以共享同一份 KB (A/B sweep 必用) |
| `compute_metrics` k 值自适应 | `run_eval.py`, `benchmarks/metrics.py` | `--max-samples 8192` 现在会算到 pass@8192 |
| 添加 5 个 DSP-V2 专用 profile | `prover/unified/profiles.py` | 论文 baseline + 3 档增量, 头对头 A/B |
| 添加 3 个 DSP-V2 framing prompt | `prover/unified/system_prompts.py` | 论文 Appendix A 原 prompt verbatim |
| 加 `reproduce_minif2f.sh` | (新文件) | 一站式 miniF2F-test 复现 (smoke / API / vLLM) |
| 加 `reproduce_minif2f_7b_ablation.sh` | (新文件) | 5 档 profile sweep + 自动对照论文 Table 1 |

回归保护: 853/853 tests pass + 1 skipped。新加任何 profile 必触发 `tests/test_all_profiles_smoke.py` 自动覆盖。

---

## 引用

```bibtex
@software{ai4math2026,
  title = {AI4Math: An Agent Operating System for Formal Theorem Proving},
  year  = {2026},
  url   = {https://github.com/ai4math/ai4math}
}
```

## 许可证

[MIT](LICENSE)
