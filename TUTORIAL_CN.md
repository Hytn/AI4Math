# AI4Math 定理证明 Agent — 傻瓜式使用教程

> **适用读者**：数学专业学生/研究者，了解定理证明的概念，但不熟悉编程和计算机操作。
>
> **你将学会**：从零开始安装环境、输入一个数学定理、让 AI 自动生成 Lean 4 形式化证明。

---

## 目录

1. [这个工具能做什么？](#1-这个工具能做什么)
2. [你需要准备什么](#2-你需要准备什么)
3. [第一步：安装 Python](#3-第一步安装-python)
4. [第二步：下载本项目](#4-第二步下载本项目)
5. [第三步：安装依赖](#5-第三步安装依赖)
6. [第四步：获取 AI 密钥](#6-第四步获取-ai-密钥)
7. [第五步：运行你的第一个证明](#7-第五步运行你的第一个证明)
8. [第六步：证明你自己的定理](#8-第六步证明你自己的定理)
9. [进阶：安装 Lean 4 进行完整验证](#9-进阶安装-lean-4-进行完整验证)
10. [进阶：批量评测基准数据集](#10-进阶批量评测基准数据集)
11. [常见问题](#11-常见问题)
12. [核心概念速查表](#12-核心概念速查表)

---

## 1. 这个工具能做什么？

想象你有一个数学定理（比如 "对所有自然数 n，n + 0 = n"），你想要一个**机器可检验的形式化证明**。

这个 Agent 会：

```
你的定理 ──→ [AI 思考] ──→ Lean 4 形式化证明 ──→ [机器验证] ──→ ✓ 正确
              ↑                    ↓
              └─── 如果错了，自动修复 ←─┘
```

它背后的工作流是：
1. **AI（Claude）** 阅读你的定理，生成 Lean 4 证明代码
2. **Lean 4 编译器** 检查证明是否正确
3. 如果不正确，AI 会自动**诊断错误**并**修复**
4. 重复直到成功（或者用完尝试次数）

**类比**：就像你把一道题交给一个非常聪明的助教，他会一遍遍尝试写证明，直到每一步都严格正确。

---

## 2. 你需要准备什么

| 东西 | 说明 | 必须？ |
|------|------|--------|
| 一台电脑 | Windows / macOS / Linux 都可以 | ✅ 必须 |
| 网络连接 | 需要联网调用 AI 接口 | ✅ 必须 |
| Python 3.10+ | 运行本工具的编程语言环境 | ✅ 必须 |
| Anthropic API 密钥 | 让 AI 为你思考的"通行证" | ✅ 必须 |
| Lean 4 | 形式化验证器（可选，不装也能用 mock 模式） | ❌ 可选 |

> **不需要**会写代码！你只需要在"终端"里复制粘贴命令。

---

## 3. 第一步：安装 Python

### Windows 用户

1. 打开浏览器，访问 https://www.python.org/downloads/
2. 点击大大的黄色按钮 **"Download Python 3.12.x"**
3. 运行下载的安装包
4. ⚠️ **重要**：勾选 **"Add Python to PATH"**（在安装界面底部的复选框）
5. 点击 "Install Now"

### macOS 用户

打开"终端"应用（在启动台 → 其他 → 终端），粘贴：

```bash
# 如果没有 Homebrew，先安装它（一个软件管理工具）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Python
brew install python@3.12
```

### Linux 用户

```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv
```

### 验证安装成功

打开终端，输入：

```bash
python3 --version
```

如果看到 `Python 3.10.x`（或更高），就成功了！

> **什么是"终端"？** 就是一个可以输入文字命令的窗口。
> - Windows：按 `Win + R`，输入 `cmd`，回车
> - macOS：打开 "终端" 应用
> - Linux：按 `Ctrl + Alt + T`

---

## 4. 第二步：下载本项目

在终端中执行（复制粘贴整行，然后按回车）：

```bash
# 方法一：用 git（推荐）
git clone https://github.com/your-org/AI4Math-tactics-search-lean.git
cd AI4Math-tactics-search-lean

# 方法二：如果没有 git，直接下载 zip 并解压
# 然后在终端中 cd 进入解压后的文件夹
```

> **什么是 `cd`？** 就是"进入某个文件夹"的意思。
> 比如你把文件解压到了桌面上的 `AI4Math` 文件夹，就输入：
> ```bash
> cd ~/Desktop/AI4Math
> ```

---

## 5. 第三步：安装依赖

在项目文件夹里执行：

```bash
# 创建一个独立的 Python 环境（避免和系统冲突）
python3 -m venv venv

# 激活这个环境
# Windows 用户：
venv\Scripts\activate
# macOS/Linux 用户：
source venv/bin/activate

# 你会看到终端最前面多了 (venv)，说明激活成功

# 安装所需的软件包
pip install -r requirements.txt

# 安装测试工具（可选但推荐）
pip install pytest
```

### 验证安装成功

```bash
python3 -c "import pyrsistent; print('✓ 安装成功')"
```

如果看到 `✓ 安装成功`，就可以继续了。

---

## 6. 第四步：获取 AI 密钥

AI4Math 使用 Claude（Anthropic 的 AI 模型）来思考证明策略。你需要一个 API 密钥：

1. 访问 https://console.anthropic.com/
2. 注册一个账号（需要邮箱）
3. 在左侧菜单找到 **"API Keys"**
4. 点击 **"Create Key"**
5. 复制生成的密钥（类似 `sk-ant-api03-xxxxxxxxxxxx`）

### 设置密钥

在终端中执行（用你自己的密钥替换 `sk-ant-...`）：

```bash
# macOS/Linux
export ANTHROPIC_API_KEY="sk-ant-api03-你的密钥粘贴在这里"

# Windows CMD
set ANTHROPIC_API_KEY=sk-ant-api03-你的密钥粘贴在这里

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-你的密钥粘贴在这里"
```

> ⚠️ **注意**：每次打开新的终端窗口都需要重新设置。
> 如果嫌麻烦，可以把这行加到你的 `~/.bashrc` 或 `~/.zshrc` 文件中。

> **不想花钱？** 你可以先用 mock 模式（不需要密钥），见下一节。

---

## 7. 第五步：运行你的第一个证明

### 方式一：Mock 模式（不需要 API 密钥、不需要 Lean）

这是最简单的开始方式，用来验证安装是否正确：

```bash
python3 run_single.py --builtin nat_add_comm --provider mock
```

你会看到类似这样的输出：

```
[nat_add_comm] Lean check → lean_error (mock mode: no real Lean)
证明尝试完成，用时 0.1s
```

这说明系统工作正常！（mock 模式下 AI 会生成一个 sorry 占位证明）

### 方式二：真实 AI 模式（需要 API 密钥）

确保你已经设置了 `ANTHROPIC_API_KEY`，然后：

```bash
python3 run_single.py --builtin nat_add_comm --provider anthropic
```

AI 会尝试证明 `∀ n m : Nat, n + m = m + n`。你会看到：

```
[nat_add_comm] 正在生成证明...
[nat_add_comm] AI 建议的证明:
  := by
    induction n with
    | zero => simp
    | succ n ih => simp [Nat.succ_add, ih]
[nat_add_comm] Lean 验证 → 需要安装 Lean 4 进行验证
```

### 方式三：完整验证模式（需要 API 密钥 + Lean 4）

如果你安装了 Lean 4（见第 9 节），可以运行完整流程：

```bash
python3 run_single.py --builtin nat_add_comm --provider anthropic --lean-mode local
```

---

## 8. 第六步：证明你自己的定理

### 直接在命令行输入

```bash
python3 run_single.py \
  --theorem "theorem my_thm (n : Nat) : n + 0 = n" \
  --provider anthropic
```

### 证明更复杂的定理

```bash
python3 run_single.py \
  --theorem "theorem add_comm_int (a b : Int) : a + b = b + a" \
  --provider anthropic \
  --samples 8
```

`--samples 8` 表示让 AI 尝试 8 次（不同的随机策略），取最好的一个。

### 从文件输入

创建一个文本文件 `my_theorem.lean`，写入你的定理：

```lean
theorem my_cool_theorem (n m : Nat) : n * m = m * n
```

然后运行：

```bash
python3 run_single.py --file my_theorem.lean --provider anthropic
```

### 参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--builtin NAME` | 使用内置题库中的题目 | — |
| `--theorem "..."` | 直接输入定理声明 | — |
| `--file PATH` | 从文件读取定理 | — |
| `--provider` | AI 提供商：`anthropic` 或 `mock` | `mock` |
| `--samples N` | 尝试次数（越多越可能成功） | 8 |
| `--lean-mode` | Lean 验证模式：`local`、`docker`、`skip` | `skip` |

### 内置题库

你可以用 `--builtin` 快速测试这些预设题目：

```bash
# 查看所有内置题目
python3 run_single.py --list-builtins

# 一些例子
python3 run_single.py --builtin nat_add_comm --provider anthropic    # n + m = m + n
python3 run_single.py --builtin nat_add_assoc --provider anthropic   # 加法结合律
python3 run_single.py --builtin p_implies_p --provider anthropic     # P → P
```

---

## 9. 进阶：安装 Lean 4 进行完整验证

不安装 Lean 4 也可以使用 AI 生成证明，但**无法机器验证**证明的正确性。安装 Lean 4 后，每一步都会被编译器严格检查。

### 安装 Lean 4

```bash
# 安装 elan（Lean 版本管理器）
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh

# 重新打开终端，或者执行：
source ~/.elan/env

# 验证
lean --version
# 应该显示类似 leanprover/lean4:v4.x.y
```

### 初始化 Mathlib 项目（首次需要，耗时约 10-30 分钟）

```bash
# 在项目目录中运行
python3 -c "
from agent.executor.lean_env import LeanEnvironment
env = LeanEnvironment.create(mathlib=True)
env.ensure_ready()
print('Lean 环境就绪:', env.status())
"
```

或者使用 Docker（推荐，避免环境冲突）：

```bash
cd docker
docker compose up -d lean
# 等待构建完成（首次约 30-60 分钟，之后秒启动）
```

### 使用完整验证模式

```bash
# 本地 Lean
python3 run_single.py --builtin nat_add_comm --provider anthropic --lean-mode local

# Docker Lean
python3 run_single.py --builtin nat_add_comm --provider anthropic --lean-mode docker
```

---

## 10. 进阶：批量评测基准数据集

AI4Math 支持在标准数学基准上评测：

```bash
# 评测 miniF2F 数据集（本科数学竞赛题）
python3 run_eval.py --benchmark minif2f --provider anthropic --limit 10

# 评测 PutnamBench（Putnam 竞赛题）
python3 run_eval.py --benchmark putnambench --provider anthropic --limit 5

# 评测所有内置题目
python3 run_eval.py --benchmark builtin --provider mock
```

结果会保存在 `results/` 文件夹中，包含每道题的证明尝试记录。

---

## 11. 常见问题

### Q: 终端报错 `command not found: python3`
用 `python` 代替 `python3` 试试（Windows 上常见）。

### Q: 报错 `ModuleNotFoundError: No module named 'pyrsistent'`
你忘了激活虚拟环境。执行：
```bash
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### Q: 报错 `ANTHROPIC_API_KEY not set`
你需要设置 API 密钥，见第 6 步。

### Q: AI 生成了证明但说 "Lean not found"
这是正常的——你没有安装 Lean 4，所以无法验证。证明**可能是对的**，但没有被机器确认。安装 Lean 4 见第 9 步。

### Q: AI 尝试了很多次都证不出来
- 增加尝试次数：`--samples 32`
- 你的定理可能太难了（目前 AI 对 IMO 级别的题目仍有困难）
- 确保定理声明是合法的 Lean 4 语法

### Q: 每次打开终端都要重新设置 API 密钥，很麻烦
把密钥写入配置文件（一劳永逸）：

```bash
# macOS/Linux
echo 'export ANTHROPIC_API_KEY="sk-ant-你的密钥"' >> ~/.bashrc
source ~/.bashrc

# Windows (PowerShell, 管理员模式)
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-你的密钥", "User")
```

### Q: 可以用这个工具写数学论文吗？
可以辅助！你可以：
1. 把你的定理声明输入系统
2. AI 生成形式化证明
3. 用 Lean 4 验证正确性
4. 在论文中引用形式化证明作为正确性保证

### Q: 要花多少钱？
Anthropic API 按使用量计费。大约：
- 一道简单定理（8 次尝试）：约 $0.05
- 一道竞赛题（32 次尝试）：约 $0.50
- miniF2F 全量评测（488 题）：约 $20-50

---

## 12. 核心概念速查表

| 术语 | 你已经知道的类比 | 在这个项目中的意思 |
|------|----------------|-------------------|
| **Lean 4** | LaTeX 之于论文排版 | 一种可以让计算机检查数学证明的编程语言 |
| **Mathlib** | 数学百科全书 | Lean 4 的数学库，包含 10 万+ 已证定理 |
| **tactic** | 证明中的一步操作 | Lean 4 中的证明策略，如 `simp`（化简）、`induction`（归纳） |
| **sorry** | "留作练习" | 证明中的占位符，表示"这一步还没证" |
| **API 密钥** | 图书馆借书证 | 使用 AI 服务的身份凭证 |
| **形式化** | 把直觉翻译成严格语言 | 将自然语言数学转换为 Lean 4 代码 |
| **mock 模式** | 模拟考试 | 不联网、不花钱的测试模式 |
| **MCTS 搜索** | 下棋时的"看几步" | AI 搜索证明策略的算法（蒙特卡洛树搜索） |
| **前提检索** | 查公式手册 | 从 Mathlib 中找可能有用的已知定理 |
| **repair 循环** | 改错本 | AI 分析证明错误 → 自动修改 → 重新检查 |

---

## 快速参考卡片

```
┌─────────────────────────────────────────────────────────────┐
│                    AI4Math 快速上手                          │
│                                                             │
│  安装：                                                     │
│    python3 -m venv venv && source venv/bin/activate          │
│    pip install -r requirements.txt                          │
│                                                             │
│  设置密钥：                                                 │
│    export ANTHROPIC_API_KEY="sk-ant-你的密钥"                │
│                                                             │
│  证明一个定理：                                             │
│    python3 run_single.py \                                  │
│      --theorem "theorem t (n : Nat) : n + 0 = n" \         │
│      --provider anthropic                                   │
│                                                             │
│  用内置题库：                                               │
│    python3 run_single.py --builtin nat_add_comm \           │
│      --provider anthropic                                   │
│                                                             │
│  纯测试（不花钱）：                                         │
│    python3 run_single.py --builtin nat_add_comm \           │
│      --provider mock                                        │
│                                                             │
│  批量评测：                                                 │
│    python3 run_eval.py --benchmark builtin \                │
│      --provider anthropic                                   │
└─────────────────────────────────────────────────────────────┘
```

---

*最后更新：2026 年 4 月*
*如有问题，请提 GitHub Issue 或联系项目维护者。*
