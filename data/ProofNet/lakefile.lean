import Lake
open Lake DSL

package «proofnetsharp» {
  -- Lean environment for PAug/ProofNetSharp evaluation.
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.20.0"
require REPL from git
  "https://github.com/leanprover-community/repl.git" @ "bump_to_v4.20.0"

@[default_target]
lean_lib «ProofNetSharp» where
  roots := #[`ProofNetSharp]
