# AI4Math — Formal Proof Agent Platform

> 形式化证明智能体平台：并行 Rollout + 经验共享 + Lean 4 内核验证

## Demo (GitHub Pages)

**无需任何安装，直接在线演示：**

1. 把本仓库 push 到 GitHub
2. Settings → Pages → Source: `main` branch, `/docs` folder → Save
3. 等待 1-2 分钟，访问 `https://<username>.github.io/<repo>/`

Demo 前端包含预录制的证明轨迹，支持：
- 选择题目 → 点击 "Run proof agent"
- 观看并行 rollout 动画（每轮 N 个采样同时发送到 Lean 内核）
- 展开任意采样查看生成的 proof、Lean 报错、token 消耗
- 观察 lemma banking（跨轮经验共享）
- 最终看到 Lean 内核验证通过的绿色标记

## 架构

```
                        ┌─────────────────────────────────────┐
                        │         Orchestrator                │
                        │  strategy: sequential | rollout     │
                        └──────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┬┴────────────────┐
                    ▼                  ▼                  ▼
             ┌────────────┐   ┌──────────────┐   ┌────────────┐
             │  Retriever  │   │  LLM Policy   │   │   Lean     │
             │  (premise   │   │  (Claude/GPT/  │   │  Checker   │
             │  selection) │   │   local)       │   │  (Docker)  │
             └────────────┘   └──────────────┘   └────────────┘
                                       │
                              ┌────────┴────────┐
                              ▼                  ▼
                       ┌────────────┐    ┌─────────────┐
                       │   Error     │    │  Lemma Bank  │
                       │  Analyzer   │    │  (experience │
                       │             │    │   sharing)   │
                       └────────────┘    └─────────────┘
```

**Rollout 策略（推理 + RL 训练共用）：**
- 每轮并行采样 N 个 proof → 全部送 Lean 编译
- 从失败的 proof 中提取已证 lemma → 注入下轮 prompt
- 固定高温度 + 宽采样（不做 temperature escalation）
- `prove_with_experience()` 同时返回 ProofTrace + RL 训练数据

## 快速开始

### 冒烟测试（无需 Lean / API key）

```bash
pip install pyyaml pydantic
python scripts/smoke_test.py
```

### 单题测试

```bash
# Mock 模式（不消耗 API）
python run_single.py --builtin nat_add_comm --provider mock

# 真实模式（需要 API key）
export ANTHROPIC_API_KEY="sk-..."
python run_single.py --builtin nat_add_comm
```

### 批量评测

```bash
python run_eval.py --benchmark builtin --provider mock --limit 3
```

### 接入真实 Lean 环境

```bash
# 1. 构建 Lean Docker 镜像（首次约 30-60 min）
cd docker && docker build -t ai4math-lean . && cd ..

# 2. 配置
cp config/default.yaml config/local.yaml
# 编辑 local.yaml: 填入 API key，设置 lean.mode = "docker"

# 3. 运行
python run_single.py --builtin nat_add_comm
```

### 启动后端 API（可选，接实时前端用）

```bash
pip install -r requirements.txt
python server.py
# API: http://localhost:8000
# WebSocket: ws://localhost:8000/ws/prove
```

## 项目结构

```
ai4math-demo/
├── docs/                       # GitHub Pages 静态前端
│   └── index.html              # 投资人演示页面（自包含）
├── core/                       # 核心智能体模块
│   ├── models.py               # 数据模型 (ProofTrace, ProofAttempt, ...)
│   ├── orchestrator.py         # 主调度器 (sequential / rollout)
│   ├── rollout.py              # 并行 Rollout 引擎 (推理 + RL 共用)
│   ├── lemma_bank.py           # 已证引理银行 (跨 rollout 经验共享)
│   ├── lean_checker.py         # Lean 4 编译验证 (Docker / local)
│   ├── llm_policy.py           # LLM 调用 (Claude / GPT / local / mock)
│   ├── error_analyzer.py       # Lean 报错结构化 + 修复建议
│   └── retriever.py            # 前提检索 (none / BM25 / embedding)
├── benchmarks/                 # 基准评测
│   ├── loader.py               # 数据加载 (miniF2F / builtin / JSON)
│   └── eval_runner.py          # 批量评测执行器
├── docker/                     # Lean 4 + Mathlib 容器化环境
│   └── Dockerfile
├── config/
│   └── default.yaml            # 默认配置
├── scripts/
│   └── smoke_test.py           # 冒烟测试
├── server.py                   # FastAPI 后端 (REST + WebSocket)
├── run_single.py               # 单题测试 CLI
├── run_eval.py                 # 批量评测 CLI
└── requirements.txt
```

## 配置说明

```yaml
# config/local.yaml
llm:
  provider: "anthropic"          # anthropic / openai / mock
  model: "claude-sonnet-4-20250514"
  api_key: ""                    # 或设环境变量 ANTHROPIC_API_KEY

lean:
  mode: "docker"                 # docker / local
  docker_image: "ai4math-lean"
  timeout_seconds: 120

orchestrator:
  strategy: "rollout"            # rollout / sequential
  samples_per_round: 8           # 每轮并行采样数
  max_rounds: 4                  # 最大轮数
  rollout_temperature: 0.9       # 固定高温
  enable_lemma_bank: true        # 跨轮经验共享
  collect_rl_data: true          # 收集 RL 训练数据
```

## RL 训练数据收集

```python
from core.orchestrator import Orchestrator, OrchestratorConfig

orc = Orchestrator(lean, llm, retriever, config=OrchestratorConfig(
    strategy="rollout",
    collect_rl_data=True,
))

# prove_with_experience() 同时返回推理结果和 RL 数据
trace, experience = orc.prove_with_experience(problem)

# experience.trajectories: [{prompt, proof, reward, lean_errors, ...}, ...]
# experience.banked_lemmas: [{name, statement, proof, verified}, ...]
# → 可直接用于 SFT / DPO / RL 训练
```

## License

Proprietary — AI4Math Team
