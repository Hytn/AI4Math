"""prover/pipeline/proof_loop.py — 单次证明循环: sketch → codegen → verify → repair

After initial verification fails, the loop invokes the repair pipeline:
  1. Diagnose errors (error_diagnostor)
  2. Try rule-based quick fixes
  3. If quick fix fails, generate LLM-based repairs
  4. Re-verify each repair candidate
"""
from __future__ import annotations
import logging
import time
from prover.models import ProofAttempt, AttemptStatus
from agent.brain.prompt_builder import build_prompt
from agent.brain.response_parser import extract_lean_code
from agent.brain.roles import AgentRole, ROLE_PROMPTS

logger = logging.getLogger(__name__)


class ProofLoop:
    def __init__(self, lean_env, llm, retriever=None, config=None):
        self.lean = lean_env
        self.llm = llm
        self.retriever = retriever
        self.config = config or {}
        self.max_repair_rounds = self.config.get("max_repair_rounds", 2)

    def single_attempt(self, problem, memory, temperature=0.7,
                       attempt_num=1) -> ProofAttempt:
        attempt = ProofAttempt(attempt_number=attempt_num)
        premises = (self.retriever.retrieve(problem.theorem_statement)
                    if self.retriever else [])
        attempt.retrieved_premises = premises

        # Build banked lemma context, filtered for relevance
        banked = ""
        if memory.banked_lemmas:
            banked = "\n".join(
                f"-- {l.get('name','')}: {l.get('statement','')}"
                for l in memory.banked_lemmas[:10]
            )

        # Include last failed proof for context in retries
        last_failed_proof = ""
        last_error_analysis = ""
        if memory.attempt_history:
            last = memory.attempt_history[-1]
            last_failed_proof = last.get("generated_proof", "")
            if last.get("errors"):
                last_error_analysis = "; ".join(
                    e.get("message", "")[:100] for e in last.get("errors", [])[:5]
                )

        prompt = build_prompt(
            theorem_statement=problem.theorem_statement,
            premises=premises,
            banked_lemmas=banked,
            error_analysis=last_error_analysis if last_error_analysis else "",
            failed_proof=last_failed_proof if last_error_analysis else "")

        # ── Step 1: Generate initial proof ──
        try:
            resp = self.llm.generate(
                system=ROLE_PROMPTS[AgentRole.PROOF_GENERATOR],
                user=prompt, temperature=temperature)
            proof = extract_lean_code(resp.content)
            attempt.generated_proof = proof
            attempt.llm_model = resp.model
            attempt.llm_tokens_in = resp.tokens_in
            attempt.llm_tokens_out = resp.tokens_out
            attempt.llm_latency_ms = resp.latency_ms
        except Exception as e:
            attempt.lean_result = AttemptStatus.LLM_ERROR
            attempt.lean_stderr = str(e)
            return attempt

        if not proof.strip():
            attempt.lean_result = AttemptStatus.LLM_ERROR
            attempt.lean_stderr = "Empty proof"
            return attempt

        # ── Step 2: Verify ──
        status, errors, stderr, check_ms = self._verify(
            problem.theorem_statement, proof)
        attempt.lean_result = status
        attempt.lean_errors = errors
        attempt.lean_stderr = stderr
        attempt.lean_check_ms = check_ms

        if status == AttemptStatus.SUCCESS:
            # Extract reusable lemmas from successful proof
            self._extract_lemmas(proof, memory)
            return attempt

        # ── Step 3: Repair loop ──
        if errors and self.max_repair_rounds > 0:
            repaired = self._repair_loop(
                problem.theorem_statement, proof, errors, attempt)
            if repaired is not None:
                self._extract_lemmas(repaired.generated_proof, memory)
                return repaired

        return attempt

    def _extract_lemmas(self, proof: str, memory):
        """Extract 'have' steps from successful proofs as reusable lemmas."""
        import re
        have_pattern = re.compile(
            r'have\s+(\w+)\s*:\s*(.+?)\s*:=\s*by\s+(.*?)(?=\n\s*(?:have|exact|show|apply|$))',
            re.DOTALL)
        for match in have_pattern.finditer(proof):
            name, stmt, prf = match.group(1), match.group(2).strip(), match.group(3).strip()
            if "sorry" not in prf:
                lemma = {"name": name, "statement": stmt, "proof": prf}
                if lemma not in memory.banked_lemmas:
                    memory.banked_lemmas.append(lemma)

    def _verify(self, theorem: str, proof: str):
        """Run Lean verification. Returns (status, errors, stderr, ms)."""
        try:
            from prover.verifier.lean_checker import LeanChecker
            checker = LeanChecker(self.lean)
            return checker.check(theorem, proof)
        except Exception as e:
            return AttemptStatus.LEAN_ERROR, [], str(e), 0

    def _repair_loop(self, theorem: str, failed_proof: str,
                     errors, attempt: ProofAttempt):
        """Try to repair a failed proof.

        Returns a successful ProofAttempt, or None if repair fails.
        """
        from prover.repair.error_diagnostor import diagnose
        from prover.repair.repair_generator import RepairGenerator

        repairer = RepairGenerator(self.llm)
        current_proof = failed_proof
        current_errors = errors

        for round_idx in range(self.max_repair_rounds):
            # Diagnose errors
            error_analysis = diagnose(current_errors)
            logger.debug(f"Repair round {round_idx + 1}: {error_analysis[:200]}")

            # Try rule-based quick fix first
            quick_fixed = repairer.quick_fix(
                theorem, current_proof, current_errors)
            if quick_fixed != current_proof:
                status, new_errors, stderr, ms = self._verify(
                    theorem, quick_fixed)
                if status == AttemptStatus.SUCCESS:
                    attempt.generated_proof = quick_fixed
                    attempt.lean_result = status
                    attempt.lean_errors = new_errors
                    attempt.lean_stderr = stderr
                    attempt.lean_check_ms += ms
                    return attempt

            # LLM-based repair
            try:
                repair_candidates = repairer.generate_repair(
                    theorem, current_proof, current_errors,
                    error_analysis, max_repairs=2, temperature=0.5)
            except Exception as e:
                logger.debug(f"Repair generation failed: {e}")
                break

            # Try each candidate
            for candidate in repair_candidates:
                if not candidate.strip():
                    continue
                status, new_errors, stderr, ms = self._verify(
                    theorem, candidate)
                if status == AttemptStatus.SUCCESS:
                    attempt.generated_proof = candidate
                    attempt.lean_result = status
                    attempt.lean_errors = new_errors
                    attempt.lean_stderr = stderr
                    attempt.lean_check_ms += ms
                    return attempt

                # If this candidate has fewer errors, adopt it for next round
                if len(new_errors) < len(current_errors):
                    current_proof = candidate
                    current_errors = new_errors

        return None
