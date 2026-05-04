"""prover.lemma_bank — 跨问题/跨会话的已证引理复用 (v14 回归)

预留接口 A 知识库 Layer 1 的 lemma 维度。每条 ``ProvedLemma`` 走
``PersistentLemmaBank`` 持久化到 SQLite, 跨问题用 BM25 检索复用。

架构 (v14):
  bank.py             ProvedLemma 数据类 + 内存版 LemmaBank (单进程, 单问题内)
  persistent_bank.py  PersistentLemmaBank — SQLite + BM25, 跨问题/跨会话
  lemma_extractor.py  LemmaExtractor — 从失败的证明里抽出已证 have-step
  lemma_verifier.py   LemmaVerifier — 二次验证抽出的引理 (避免污染库)

主路径接通点 (v14):
  - ConjectureProposeTool 后置写入: 当 proposer 提出的 lemma 通过 verifier
    文本过滤, 写到 PersistentLemmaBank。
  - LemmaBankTool 读: 每次调起前先用 BM25 检索相关已证 lemma, 拼到 prompt。

Usage::

    from prover.lemma_bank import PersistentLemmaBank, ProvedLemma
    bank = PersistentLemmaBank("~/.ai4math/lemma_bank.db")
    bank.add(ProvedLemma(name="h1", statement="lemma h1 ...", proof=":= by ..."))
    relevant = bank.search("Nat.add_comm", top_k=5)
    preamble = bank.to_lean_preamble(relevant)
"""
from prover.lemma_bank.bank import ProvedLemma, LemmaBank
from prover.lemma_bank.persistent_bank import PersistentLemmaBank
from prover.lemma_bank.lemma_extractor import LemmaExtractor
from prover.lemma_bank.lemma_verifier import LemmaVerifier

__all__ = [
    "ProvedLemma",
    "LemmaBank",
    "PersistentLemmaBank",
    "LemmaExtractor",
    "LemmaVerifier",
]
