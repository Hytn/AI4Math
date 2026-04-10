# AI4Math — Lean 4 + Mathlib + REPL Environment
#
# Build:  docker build -t ai4math-lean -f docker/Dockerfile.lean docker/
# Test:   docker run --rm ai4math-lean echo '{"cmd": "#check Nat", "env": 0}' | .lake/build/bin/repl
#
# Key features:
#   1. Locked Lean version + Mathlib commit for reproducibility
#   2. Pre-compiled mathlib oleans cache (avoids 30min build per run)
#   3. Built lean4-repl binary for interactive tactic-level proving
#   4. lean_daemon.py for socket-based multi-client access

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Base tools ──
RUN apt-get update && apt-get install -y \
    curl git cmake python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Install elan (Lean version manager) ──
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none
ENV PATH="/root/.elan/bin:${PATH}"

# ── Create Lean project ──
WORKDIR /workspace/lean-project

# Lean toolchain — MUST match the mathlib commit below
RUN echo 'leanprover/lean4:v4.17.0' > lean-toolchain

# lakefile.toml — locked mathlib + lean4-repl
RUN cat > lakefile.toml <<'LAKEFILE'
[package]
name = "AI4MathCheck"
leanOptions = [["autoImplicit", false]]

[[require]]
name = "mathlib"
scope = "leanprover-community"
type = "git"
source = "https://github.com/leanprover-community/mathlib4.git"
# Locked to 2025-03 stable
revision = "13042290464e1e615b8cd1e0d5aba0ef16472bd1"

[[require]]
name = "repl"
type = "git"
source = "https://github.com/leanprover-community/repl.git"
# Use a version compatible with v4.17.0
revision = "main"
LAKEFILE

# ── Fetch dependencies and build ──
# This is the expensive step (~30-60 min), cached by Docker layers
RUN lake update
RUN lake exe cache get || true
RUN lake build
# Build the REPL binary specifically
RUN lake build repl

# Verify REPL binary exists
RUN test -f .lake/build/bin/repl && echo "REPL binary OK" || \
    (echo "REPL binary not found!" && exit 1)

# ── Install lean_daemon ──
COPY lean_daemon.py /workspace/lean_daemon.py

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD echo '{"cmd": "#check Nat", "env": 0}' | timeout 10 /workspace/lean-project/.lake/build/bin/repl || exit 1

WORKDIR /workspace/lean-project
CMD ["python3", "/workspace/lean_daemon.py"]
