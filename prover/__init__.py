"""prover/ — 证明编排层 (Profile 驱动的统一管线)

主入口:
    prover.unified      ★ UnifiedProofRunner + 14 个 Profile (大一统核心)

辅助子系统 (Profile 装配时按需挂载):
    prover.conjecture   主动猜想辅助引理 (conjecture_driven profile)
    prover.decompose    目标拆分 (DSP / pantograph_dsp / conjecture_driven 的 tool)
    prover.premise      引理检索 (BM25 / 字符 n-gram TF-IDF / 融合)
    prover.verifier     Sorry/Axiom 完整性检查

数据类型: prover.models (BenchmarkProblem, ProofTrace, ProofAttempt)
"""
