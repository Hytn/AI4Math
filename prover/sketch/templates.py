"""prover/sketch/templates.py — 常见证明模式模板"""
TEMPLATES = {
    "induction": "Proof by induction on {var}. Base case: {base}. Inductive step: {step}.",
    "contradiction": "Assume the negation. Derive a contradiction via {method}.",
    "direct": "Direct proof using {tactics}.",
    "construction": "Construct {witness} satisfying {conditions}.",
    "epsilon_delta": "Given ε > 0, choose δ = {delta}. Then verify {bound}.",
    "cyclic_inequality": "WLOG assume ordering. Key: equality at {extremum}. Bound via {method}.",
}
