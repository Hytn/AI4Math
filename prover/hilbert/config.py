"""prover/hilbert/config.py — Hilbert 复现的 per-role 模型配置

Hilbert (arXiv:2509.22819) 的四组件 = informal reasoner LLM + 专用
prover LLM + verifier + 检索器。论文消融的核心结论是 **reasoner 的
强弱比 prover 更影响最终成功率** —— 所以配置层必须允许两个角色挂
完全不同的 provider/model/api_base (e.g. reasoner=Claude/GPT 级通用
模型, prover=DeepSeek-Prover-V2/Goedel-Prover 级专用模型)。

这正是不能复用 Profile dataclass 的原因: Profile 是单 model 字段的
声明式配置, 加 per-role 字段会改 dataclass → 触发全部 20 个 YAML
round-trip 测试与既有运行链路。Hilbert 作为独立入口 (run_hilbert.py)
+ 独立配置 (config/hilbert.yaml) 落地, 与 unified 体系零耦合。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RoleModel:
    """单个角色的 LLM 配置 — 字段与 create_async_provider 的 config
    dict 一一对应。"""
    provider: str = "anthropic"
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    omit_temperature: bool = False
    temperature: float = 0.7
    max_tokens: int = 8192

    def provider_config(self) -> dict:
        return {
            "provider": self.provider, "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key or _env_key(self.provider),
            "omit_temperature": self.omit_temperature,
        }


def _env_key(provider: str) -> str:
    return {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "deepseek": os.environ.get("DEEPSEEK_API_KEY", ""),
    }.get(provider, "")


@dataclass
class HilbertConfig:
    """递归调度参数 (默认值对齐 Hilbert 论文设置)。"""
    reasoner: RoleModel = field(default_factory=lambda: RoleModel(
        temperature=0.7, max_tokens=16000))
    prover: RoleModel = field(default_factory=lambda: RoleModel(
        provider="vllm", temperature=1.0, max_tokens=8192))

    max_depth: int = 5            # 论文 D=5
    prover_passes: int = 4        # 每节点 prover 整证尝试数
    shallow_passes: int = 2       # 每节点 reasoner shallow-solve 尝试数
    decompose_attempts: int = 2   # 分解重试次数 (带失败反馈)
    max_subgoals: int = 6         # 单次分解的子目标上限
    check_subgoal_statements: bool = True   # 子目标先验证 `:= by sorry` 可编译
    node_budget: int = 200        # 全树节点预算 (防递归爆炸)
    preamble: str = "import Mathlib\n"

    @classmethod
    def from_dict(cls, d: dict) -> "HilbertConfig":
        cfg = cls()
        for role in ("reasoner", "prover"):
            sub = d.get(role) or {}
            rm = getattr(cfg, role)
            for k, v in sub.items():
                if hasattr(rm, k):
                    setattr(rm, k, v)
        for k in ("max_depth", "prover_passes", "shallow_passes",
                  "decompose_attempts", "max_subgoals",
                  "check_subgoal_statements", "node_budget", "preamble"):
            if k in d:
                setattr(cfg, k, d[k])
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "HilbertConfig":
        import yaml
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})
