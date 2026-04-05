# AI4Math Demo — Lean 4 + Mathlib 编译环境
#
# 构建：docker build -t ai4math-lean .
# 用法：docker run --rm -v ./check.lean:/workspace/lean-project/Check.lean ai4math-lean lake env lean /workspace/lean-project/Check.lean
#
# 重要：
#   1. 锁定 Lean 版本和 Mathlib commit，确保可复现性
#   2. 预编译 mathlib oleans cache，避免每次从源码编译

FROM ubuntu:24.04

# 避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 基础工具
RUN apt-get update && apt-get install -y \
    curl git cmake python3 \
    && rm -rf /var/lib/apt/lists/*

# ── 安装 elan (Lean 版本管理器) ─────────────────────
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
ENV PATH="/root/.elan/bin:${PATH}"

# ── 创建 Lean 项目 ─────────────────────────────────
WORKDIR /workspace/lean-project

# lakefile — 锁定 mathlib 版本
# !! 修改 MATHLIB_COMMIT 以锁定到你需要的版本 !!
RUN cat > lakefile.toml <<'EOF'
[package]
name = "AI4MathCheck"
leanOptions = [["autoImplicit", false]]

[[require]]
name = "mathlib"
scope = "leanprover-community"
type = "git"
source = "https://github.com/leanprover-community/mathlib4.git"
# 锁定到稳定的 mathlib commit — 确保评测结果可复现
# 更新方式: 运行 `lake update` 后将 lake-manifest.json 中的 rev 值复制到此处
# 此 commit 对应 2025-03 的 mathlib4 稳定版本
revision = "13042290464e1e615b8cd1e0d5aba0ef16472bd1"
EOF

# lean-toolchain — 锁定 Lean 版本
# !! 确保这个版本与上述 mathlib commit 兼容 !!
# 查看对应 toolchain: https://github.com/leanprover-community/mathlib4/blob/<commit>/lean-toolchain
RUN cat > lean-toolchain <<'EOF'
leanprover/lean4:v4.17.0
EOF

# ── 拉取依赖并预编译 mathlib ────────────────────────
# 这一步耗时较长 (30-60 min)，但只在构建镜像时执行一次
RUN lake update
RUN lake exe cache get || true
RUN lake build

# ── 工作目录 ────────────────────────────────────────
WORKDIR /workspace/lean-project

CMD ["bash"]
