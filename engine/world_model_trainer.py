"""engine/world_model_trainer.py — 世界模型训练管道

最小可行实现：从 ProofContextStore 的 RichProofTrajectory 数据中
训练 tactic 成功率预测模型，替换 MockWorldModel。

训练管道::
    1. 从 SQLite 提取 RichProofTrajectory
    2. 展平为 (goal_text, tactic_text, success) 三元组
    3. TF-IDF 向量化 goal + tactic → 特征矩阵
    4. 训练 LogisticRegression 分类器
    5. 序列化到 .pkl 文件
    6. SklearnWorldModel 加载 .pkl 进行推理

Usage::
    # 训练
    trainer = WorldModelTrainer(db_path="proofs.db")
    trainer.extract_training_data()
    trainer.train()
    trainer.save("world_model.pkl")

    # 推理
    model = SklearnWorldModel("world_model.pkl")
    pred = model.predict("⊢ n + 0 = n", "omega")
"""
from __future__ import annotations
import json, logging, os, pickle, re, time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from engine.world_model import WorldModelPredictor, WorldModelPrediction, MockWorldModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    goal_text: str
    tactic: str
    tactic_base: str
    success: bool
    error_category: str = ""
    goal_shape: str = ""
    domain: str = ""


def _tactic_base(tactic: str) -> str:
    tokens = tactic.strip().split()
    return tokens[0] if tokens else ""

def _goal_shape(goal: str) -> str:
    s = re.sub(r'\b[a-z_]\w*\b', 'V', goal.strip())
    return re.sub(r'\b\d+\b', 'N', s)[:200]

def _goal_domain(goal: str) -> str:
    g = goal.lower()
    for kw, dom in [(['nat','ℕ','succ'],'nat'), (['int','ℤ'],'int'),
                     (['real','ℝ'],'real'), (['finset'],'finset'),
                     (['group','ring','field'],'algebra'),
                     (['∀','∃','→','∧','∨'],'logic')]:
        if any(k in g for k in kw): return dom
    return 'other'


class WorldModelTrainer:
    def __init__(self, db_path: str = "proofs.db"):
        self.db_path = db_path
        self.samples: list[TrainingSample] = []
        self.model = None
        self.vectorizer = None
        self._tactic_vectorizer = None
        self._tactic_base_map = {}
        self._stats: dict = {}

    def extract_training_data(self, min_depth: int = 1, limit: int = 50000) -> int:
        import sqlite3
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found: {self.db_path}")
            return 0
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT pc.theorem, pt.step_details, pt.success, pt.depth "
            "FROM proof_traces pt JOIN proof_contexts pc ON pt.context_id = pc.id "
            "WHERE pt.step_details != '[]' AND pt.depth >= ? "
            "ORDER BY pt.created_at DESC LIMIT ?", (min_depth, limit)).fetchall()
        conn.close()
        self.samples = []
        for row in rows:
            try: steps = json.loads(row["step_details"])
            except (json.JSONDecodeError, TypeError): continue
            for step in steps:
                tac = step.get("tac", "")
                goals = step.get("goals_before", [])
                goal = goals[0] if goals else ""
                success = step.get("env_after", -1) > 0 and not step.get("error", "")
                if not tac or not goal: continue
                self.samples.append(TrainingSample(
                    goal_text=goal, tactic=tac, tactic_base=_tactic_base(tac),
                    success=success, error_category=step.get("error_cat", ""),
                    goal_shape=_goal_shape(goal), domain=_goal_domain(goal)))
        self._stats["total_samples"] = len(self.samples)
        self._stats["positive_rate"] = sum(s.success for s in self.samples) / max(1, len(self.samples))
        logger.info(f"Extracted {len(self.samples)} samples (pos={self._stats['positive_rate']:.1%})")
        return len(self.samples)

    def extract_from_trajectories(self, trajectories: list) -> int:
        self.samples = []
        for traj in trajectories:
            for step in traj.steps:
                goal = step.goals_before[0] if step.goals_before else ""
                success = step.env_id_after > 0 and not step.error_message
                if not step.tactic or not goal: continue
                self.samples.append(TrainingSample(
                    goal_text=goal, tactic=step.tactic, tactic_base=_tactic_base(step.tactic),
                    success=success, error_category=step.error_category,
                    goal_shape=_goal_shape(goal), domain=_goal_domain(goal)))
        self._stats["total_samples"] = len(self.samples)
        return len(self.samples)

    def train(self, test_size: float = 0.2) -> dict:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        from scipy.sparse import hstack, csr_matrix, lil_matrix

        if len(self.samples) < 50:
            return {"error": "insufficient_data", "count": len(self.samples)}

        goals = [s.goal_text for s in self.samples]
        tactics = [s.tactic for s in self.samples]
        tactic_bases = [s.tactic_base for s in self.samples]
        labels = np.array([int(s.success) for s in self.samples])

        # Feature 1: char n-gram TF-IDF on goals
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                           max_features=5000, sublinear_tf=True)
        X_goal = self.vectorizer.fit_transform(goals)

        # Feature 2: word TF-IDF on tactics
        self._tactic_vectorizer = TfidfVectorizer(analyzer='word', max_features=500,
                                                    sublinear_tf=True)
        X_tac = self._tactic_vectorizer.fit_transform(tactics)

        # Feature 3: tactic-base one-hot
        top_tacs = [t for t, _ in Counter(tactic_bases).most_common(50)]
        self._tactic_base_map = {t: i for i, t in enumerate(top_tacs)}
        n_tb = len(top_tacs)
        X_tb = lil_matrix((len(self.samples), n_tb))
        for i, tb in enumerate(tactic_bases):
            idx = self._tactic_base_map.get(tb)
            if idx is not None: X_tb[i, idx] = 1.0
        X_tb = X_tb.tocsr()

        # Feature 4: hand-crafted
        hand = np.zeros((len(self.samples), 6))
        for i, s in enumerate(self.samples):
            hand[i] = [len(s.goal_text)/500, s.goal_text.count('→'),
                        s.goal_text.count('∀'), s.goal_text.count('='),
                        float(s.domain == 'nat'),
                        float(s.tactic_base in ('simp','ring','omega','norm_num','linarith'))]

        X = hstack([X_goal, X_tac, X_tb, csr_matrix(hand)])
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42,
            stratify=labels if len(set(labels)) > 1 else None)

        self.model = LogisticRegression(C=1.0, max_iter=1000, solver='saga',
                                         class_weight='balanced', random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        metrics = {"accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "train_size": int(X_train.shape[0]), "test_size": int(X_test.shape[0]),
                    "n_features": int(X.shape[1]), "positive_rate": float(labels.mean())}
        logger.info(f"Trained: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}")
        self._stats.update(metrics)
        return metrics

    def save(self, path: str = "world_model.pkl"):
        if self.model is None: raise RuntimeError("Not trained")
        with open(path, 'wb') as f:
            pickle.dump({"model": self.model, "vectorizer": self.vectorizer,
                          "tactic_vectorizer": self._tactic_vectorizer,
                          "tactic_base_map": self._tactic_base_map,
                          "stats": self._stats, "version": 1}, f)
        logger.info(f"Saved to {path}")

    @property
    def stats(self) -> dict: return dict(self._stats)


class SklearnWorldModel(WorldModelPredictor):
    """训练后的世界模型。加载失败时降级到 MockWorldModel。"""
    def __init__(self, model_path: str = "world_model.pkl"):
        self._model = self._vectorizer = self._tactic_vectorizer = None
        self._tactic_base_map = {}
        self._fallback = MockWorldModel()
        self._loaded = False
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f: b = pickle.load(f)
                self._model = b["model"]; self._vectorizer = b["vectorizer"]
                self._tactic_vectorizer = b["tactic_vectorizer"]
                self._tactic_base_map = b["tactic_base_map"]
                self._loaded = True
            except Exception as e:
                logger.warning(f"Failed to load world model: {e}")

    def predict(self, goal_state: str, tactic: str,
                hypotheses: list[str] = None, context: dict = None) -> WorldModelPrediction:
        if not self._loaded:
            return self._fallback.predict(goal_state, tactic, hypotheses, context)
        try:
            from scipy.sparse import hstack, csr_matrix, lil_matrix
            tb = _tactic_base(tactic)
            X_goal = self._vectorizer.transform([goal_state])
            X_tac = self._tactic_vectorizer.transform([tactic])
            n_tb = len(self._tactic_base_map)
            X_tb = lil_matrix((1, n_tb))
            idx = self._tactic_base_map.get(tb)
            if idx is not None: X_tb[0, idx] = 1.0
            hand = np.array([[len(goal_state)/500, goal_state.count('→'),
                              goal_state.count('∀'), goal_state.count('='),
                              float(_goal_domain(goal_state)=='nat'),
                              float(tb in ('simp','ring','omega','norm_num','linarith'))]])
            X = hstack([X_goal, X_tac, X_tb.tocsr(), csr_matrix(hand)])
            prob = float(self._model.predict_proba(X)[0, 1])
            return WorldModelPrediction(tactic=tactic, likely_success=prob>=0.5,
                                         confidence=prob, reasoning=f"sklearn(p={prob:.3f})")
        except Exception:
            return self._fallback.predict(goal_state, tactic, hypotheses, context)

    @property
    def is_trained(self) -> bool: return self._loaded


def train_world_model(db_path="proofs.db", output_path="world_model.pkl", **kw) -> dict:
    t = WorldModelTrainer(db_path)
    n = t.extract_training_data(**{k: v for k, v in kw.items() if k in ('min_depth','limit')})
    if n < 50: return {"error": "insufficient_data", "samples": n}
    m = t.train()
    if "error" not in m: t.save(output_path)
    return m

if __name__ == "__main__":
    import argparse; logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="proofs.db"); p.add_argument("--output", default="world_model.pkl")
    a = p.parse_args()
    print(json.dumps(train_world_model(a.db, a.output), indent=2))
