"""prover/pipeline/proof_loop.py — 单次证明循环: sketch → codegen → verify → repair"""
from __future__ import annotations
import time
from prover.models import ProofAttempt, AttemptStatus
from agent.brain.prompt_builder import build_prompt
from agent.brain.response_parser import extract_lean_code
from agent.brain.roles import AgentRole, ROLE_PROMPTS

class ProofLoop:
    def __init__(self, lean_env, llm, retriever=None, config=None):
        self.lean = lean_env; self.llm = llm; self.retriever = retriever
        self.config = config or {}

    def single_attempt(self, problem, memory, temperature=0.7, attempt_num=1) -> ProofAttempt:
        attempt = ProofAttempt(attempt_number=attempt_num)
        premises = self.retriever.retrieve(problem.theorem_statement) if self.retriever else []
        attempt.retrieved_premises = premises
        banked = "\n".join(f"-- {l.get('name','')}: {l.get('statement','')}"
                          for l in memory.banked_lemmas[:10]) if memory.banked_lemmas else ""
        prompt = build_prompt(theorem_statement=problem.theorem_statement,
                              premises=premises, banked_lemmas=banked)
        try:
            resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.PROOF_GENERATOR],
                                     user=prompt, temperature=temperature)
            proof = extract_lean_code(resp.content)
            attempt.generated_proof = proof; attempt.llm_model = resp.model
            attempt.llm_tokens_in = resp.tokens_in; attempt.llm_tokens_out = resp.tokens_out
            attempt.llm_latency_ms = resp.latency_ms
        except Exception as e:
            attempt.lean_result = AttemptStatus.LLM_ERROR; attempt.lean_stderr = str(e)
            return attempt
        if not proof.strip():
            attempt.lean_result = AttemptStatus.LLM_ERROR; attempt.lean_stderr = "Empty proof"; return attempt
        try:
            from prover.verifier.lean_checker import LeanChecker
            checker = LeanChecker(self.lean)
            status, errors, stderr, check_ms = checker.check(problem.theorem_statement, proof)
            attempt.lean_result = status; attempt.lean_errors = errors
            attempt.lean_stderr = stderr; attempt.lean_check_ms = check_ms
        except Exception as e:
            attempt.lean_result = AttemptStatus.LEAN_ERROR; attempt.lean_stderr = str(e)
        return attempt
