<p align="center">
  <img src="https://img.shields.io/badge/Lean-4.28.0-blue" alt="Lean 4" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab" alt="Python" />
  <img src="https://img.shields.io/badge/tests-822-green" alt="Tests" />
  <img src="https://img.shields.io/badge/problems-1631-orange" alt="Problems" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a> ·
  <b>🇨🇳 中文</b> ·
  <a href="TUTORIAL_CN.md">📖 零基础教程</a> ·
  <a href="docs/ARCHITECTURE.md">🏗️ 架构</a> ·
  <a href="docs/CHANGELOG.md">📝 更新日志</a>
</p>

# AI4Math

一套 Lean 4 形式化定理证明的基础设施,**所有主流证明方法收敛到一个
`--profile` 开关**。同一个 `UnifiedProofRunner` 跑全证生成
(DeepSeek-Prover 风格)、修复循环、Draft-Sketch-Prove、ReProver 风格
RAG、LeanDojo 风格步级证明、4 路异构并行、MCTS / best-first / beam
树搜索、猜想驱动证明。加新算法 = 加一个 `Profile` entry。

## 这个项目实际做了什么

- **验证 OS** (`engine/`) — Lean 4 REPL 异步连接池, `env_id` fork
  实现 ~50 ms 增量验证, 三级预过滤 (L0 语法 / L1 REPL / L2 全编译),
  结构化 `AgentFeedback` (~100 bits, 比传统 1 bit pass/fail 信号丰富)。
- **统一 runner** (`prover/unified/`) — 14 个 active profile, CLI 是
  `python run_unified.py --profile <name>`, YAML 模板在
  `config/profiles/` 下。
- **知识系统** (`knowledge/`) — 四层 SQLite 金字塔 (原始轨迹 → 战术有效性
  → 策略模式 → 概念图谱), 带衰减遗忘。过去的 dialog 自动注入新 prompt。
- **7 个基准** (`benchmarks/`, `data/`) — miniF2F (244), PutnamBench (672),
  ProofNet (360), FATE-{M,H,X} (350), FormalMATH (sample), 内置 (5)。
- **RL 飞轮** (`sampler/`, `scripts/rl_pipeline.py`) — 评测 → 收集 trajectory
  → SFT JSONL → 世界模型 (sklearn) → 外部 trainer (TRL / verl / slime)。
  前 3 步全自动, 第 4 步交给你选的 trainer。

## 快速开始

```bash
git clone https://github.com/ai4math/ai4math.git
cd ai4math
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."

# 证一道题 — 换方法只改 --profile
python run_unified.py --builtin nat_add_comm --profile whole_proof_repair
python run_unified.py --builtin nat_add_comm --profile reprover
python run_unified.py --builtin nat_add_comm --profile heterogeneous
python run_unified.py --builtin nat_add_comm --profile mcts

# 批量评测
bash eval.sh --benchmark builtin                       # mock, 不需要 Lean
bash eval.sh --real --benchmark minif2f --samples 8    # 完整
```

每次运行产出 `results/traces/<problem_id>/dialog.json` —— 唯一标准
输出格式。打开一个文件就看到一切 (system prompt、tools、每一轮、最终
结果、搜索树如果有的话)。

## 社区 Lean backend

| Backend | Profile | 安装 |
|---|---|---|
| Kimina Lean Server | `kimina_batch` | `docker run projectnumina/kimina-lean-server:2.0.0` |
| Pantograph | `pantograph_dsp` | `pip install pypantograph` |
| LooKeng | `lookeng_lemma` | clone seed-prover, `pip install -e ".[lookeng]"` |

`dialog.json` 里 `meta.backends.<name>.is_fallback` 字段告诉你 backend
真的在用还是静默降级到本地。

## 加一个新方法

```python
# 1. prover/unified/profiles.py
PRESETS["my_method"] = Profile(
    name="my_method",
    tools=[ToolKit.LEAN_VERIFY, ToolKit.PREMISE_SEARCH],
    max_turns=8,
    framing="my_framing",
    temperature=0.5,
)

# 2. prover/unified/system_prompts.py
FRAMING_PROMPTS["my_framing"] = "You are a Lean 4 prover. Output..."

# 3. python run_unified.py --profile my_method
```

`runner.py`、`agent_loop.py`、`tools/` 一行不动。

## 目录布局

```
engine/      Lean 4 REPL 池, 三级验证, 错误智能层
agent/       AgentLoop, tools, brain (LLM), persistence (dialog.json)
prover/      Profile 驱动 runner; conjecture/formalize/decompose/repair
knowledge/   四层 SQLite 知识金字塔
sampler/     RL trajectory 采样 (verl / slime / TRL adapter)
benchmarks/  miniF2F + PutnamBench + ProofNet + FATE + FormalMATH 加载器
config/      default.yaml + 14 个 profile YAML 模板
data/        基准题目 (1631 道)
docs/        ARCHITECTURE.md + CHANGELOG.md + dialog.json schema
tests/       1024 测试
```

读代码从 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 开始。

## 这个项目**不**做的事

诚实大于宣传:

- **不做开放式数学发现。** 1631 道题都来自 7 个内置基准。
  `conjecture_driven` profile 让 LLM 在证一道**已给定**定理时自己发明
  *辅助引理*, 但不发明定理本身。
- **没有发表的 SOTA 数字。** AI4Math 自己在这些基准上的 pass@k 是
  "评测中" —— 这个项目是基础设施, 不是 leaderboard 选手。
- **RL 飞轮闭合 3/4。** 第 1-3 步 (评测 → 收集 → 世界模型) 全自动,
  第 4 步把可训的 JSONL 交给你选的 trainer。
- **"世界模型" 是 sklearn LogisticRegression。** 可训, 作 tactic 有效性
  预测器能用 —— 但不是架构图暗示的"完整状态动力学网络"。

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
