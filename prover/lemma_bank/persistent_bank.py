"""prover/lemma_bank/persistent_bank.py — SQLite 持久化引理银行

与内存版 LemmaBank 的对比:
  内存版: list[ProvedLemma], 进程退出即丢失, 仅单问题内复用
  持久版: SQLite + BM25 检索, 跨问题/跨会话复用已验证引理

核心能力:
  1. 跨问题复用: 问题 A 证明的辅助引理, 问题 B 可直接检索使用
  2. 语义检索:   按定理类型 / 关键词 / 标签搜索相关引理
  3. 增量写入:   每条引理独立持久化, 不需要全量序列化
  4. 版本感知:   记录 Lean4/Mathlib 版本, 升级后自动标记需重验

用法::

    bank = PersistentLemmaBank("~/.ai4math/lemma_bank.db")
    bank.add(ProvedLemma(name="h1", statement="lemma h1 ...", proof=":= by ..."))

    # 检索与当前问题相关的引理
    relevant = bank.search("Nat.add_comm", top_k=5)
    preamble = bank.to_lean_preamble(relevant)
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from prover.lemma_bank.bank import ProvedLemma

logger = logging.getLogger(__name__)


@dataclass
class LemmaRecord:
    """持久化引理记录 (扩展 ProvedLemma)"""
    id: int = 0
    name: str = ""
    statement: str = ""
    proof: str = ""
    verified: bool = False

    # 元信息
    source_problem: str = ""          # 来源问题 ID
    source_direction: str = ""        # 来源方向 (automation/structured/...)
    tags: list[str] = field(default_factory=list)
    lean_version: str = ""
    mathlib_rev: str = ""

    # 统计
    times_used: int = 0               # 被其他证明引用的次数
    created_at: float = 0.0
    last_used_at: float = 0.0

    # 检索辅助
    statement_hash: str = ""          # 用于去重
    keywords: list[str] = field(default_factory=list)

    def to_proved_lemma(self) -> ProvedLemma:
        return ProvedLemma(
            name=self.name, statement=self.statement,
            proof=self.proof, verified=self.verified)

    def to_lean(self) -> str:
        return f"{self.statement} {self.proof}"


class PersistentLemmaBank:
    """SQLite 持久化引理银行

    线程安全: 所有数据库操作通过 _lock 保护。
    SQLite 本身支持 WAL 模式下的并发读, 但写仍需串行化。
    """

    def __init__(self, db_path: str = "",
                 lean_version: str = "",
                 mathlib_rev: str = ""):
        if not db_path:
            db_path = os.path.join(
                str(Path.home()), ".ai4math", "lemma_bank.db")

        self._db_path = db_path
        self._lean_version = lean_version
        self._mathlib_rev = mathlib_rev
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

        # 内存缓存 (避免频繁 SQL 查询)
        self._statement_hashes: set[str] = set()

        self._init_db()

    def _init_db(self):
        """创建表结构"""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS lemmas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                statement TEXT NOT NULL,
                proof TEXT NOT NULL,
                verified INTEGER DEFAULT 0,
                source_problem TEXT DEFAULT '',
                source_direction TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                lean_version TEXT DEFAULT '',
                mathlib_rev TEXT DEFAULT '',
                times_used INTEGER DEFAULT 0,
                created_at REAL DEFAULT 0,
                last_used_at REAL DEFAULT 0,
                statement_hash TEXT NOT NULL,
                keywords TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_lemmas_hash
                ON lemmas(statement_hash);
            CREATE INDEX IF NOT EXISTS idx_lemmas_verified
                ON lemmas(verified);
            CREATE INDEX IF NOT EXISTS idx_lemmas_source
                ON lemmas(source_problem);
        """)
        self._conn.commit()

        # 加载已有 hash 到内存
        cursor = self._conn.execute(
            "SELECT statement_hash FROM lemmas")
        for row in cursor:
            self._statement_hashes.add(row[0])

    # ── 写入 ──

    def add(self, lemma: ProvedLemma,
            source_problem: str = "",
            source_direction: str = "",
            tags: list[str] = None) -> Optional[int]:
        """添加一条已验证引理 (去重)

        Returns:
            引理 ID, 如果是重复则返回 None
        """
        stmt_hash = self._hash_statement(lemma.statement)
        if stmt_hash in self._statement_hashes:
            return None

        keywords = self._extract_keywords(lemma.statement)
        now = time.time()

        with self._lock:
            cursor = self._conn.execute("""
                INSERT INTO lemmas (
                    name, statement, proof, verified,
                    source_problem, source_direction, tags,
                    lean_version, mathlib_rev,
                    created_at, statement_hash, keywords
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lemma.name, lemma.statement, lemma.proof,
                int(lemma.verified),
                source_problem, source_direction,
                json.dumps(tags or []),
                self._lean_version, self._mathlib_rev,
                now, stmt_hash, json.dumps(keywords),
            ))
            self._conn.commit()
            self._statement_hashes.add(stmt_hash)
            lemma_id = cursor.lastrowid
            logger.debug(f"LemmaBank: added '{lemma.name}' (id={lemma_id})")
            return lemma_id

    def add_batch(self, lemmas: list[ProvedLemma],
                  source_problem: str = "") -> int:
        """批量添加 (返回实际新增数量)"""
        added = 0
        for lemma in lemmas:
            if self.add(lemma, source_problem=source_problem) is not None:
                added += 1
        return added

    # ── 检索 ──

    def search(self, query: str, top_k: int = 10,
               verified_only: bool = True) -> list[LemmaRecord]:
        """按关键词搜索相关引理 (简易 BM25)

        搜索策略:
          1. 从 query 中提取关键词
          2. 在 keywords 字段中匹配
          3. 按匹配数量 + times_used 排序
        """
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return self.get_recent(top_k)

        where_clause = "WHERE verified = 1" if verified_only else "WHERE 1=1"
        with self._lock:
            cursor = self._conn.execute(f"""
                SELECT * FROM lemmas {where_clause}
                ORDER BY times_used DESC, created_at DESC
                LIMIT 200
            """)
            rows = cursor.fetchall()

        # 在 Python 侧做关键词匹配排序
        scored = []
        for row in rows:
            record = self._row_to_record(row)
            score = self._score_match(query_keywords, record)
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k]]

    def search_by_type(self, target_type: str,
                       top_k: int = 5) -> list[LemmaRecord]:
        """按目标类型搜索引理

        例: search_by_type("Nat → Nat → Nat") 找所有返回类型匹配的引理
        """
        return self.search(target_type, top_k=top_k)

    def get_for_problem(self, problem_id: str) -> list[LemmaRecord]:
        """获取特定问题产生的所有引理"""
        with self._lock:
            cursor = self._conn.execute("""
                SELECT * FROM lemmas
                WHERE source_problem = ?
                ORDER BY created_at
            """, (problem_id,))
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def get_recent(self, n: int = 10) -> list[LemmaRecord]:
        """获取最近添加的 N 条引理"""
        with self._lock:
            cursor = self._conn.execute("""
                SELECT * FROM lemmas
                WHERE verified = 1
                ORDER BY created_at DESC
                LIMIT ?
            """, (n,))
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def get_most_used(self, n: int = 10) -> list[LemmaRecord]:
        """获取被引用最多的引理"""
        with self._lock:
            cursor = self._conn.execute("""
                SELECT * FROM lemmas
                WHERE verified = 1
                ORDER BY times_used DESC
                LIMIT ?
            """, (n,))
            return [self._row_to_record(r) for r in cursor.fetchall()]

    # ── 使用记录 ──

    def mark_used(self, lemma_id: int):
        """标记引理被使用 (更新使用次数和最近使用时间)"""
        with self._lock:
            self._conn.execute("""
                UPDATE lemmas
                SET times_used = times_used + 1,
                    last_used_at = ?
                WHERE id = ?
            """, (time.time(), lemma_id))
            self._conn.commit()

    # ── 版本管理 ──

    def mark_stale_on_upgrade(self, new_lean_version: str = "",
                              new_mathlib_rev: str = ""):
        """升级 Lean/Mathlib 后, 标记旧版本引理为未验证"""
        with self._lock:
            conditions = []
            params = []
            if new_lean_version:
                conditions.append("lean_version != ?")
                params.append(new_lean_version)
            if new_mathlib_rev:
                conditions.append("mathlib_rev != ?")
                params.append(new_mathlib_rev)
            if conditions:
                where = " OR ".join(conditions)
                cursor = self._conn.execute(
                    f"UPDATE lemmas SET verified = 0 WHERE {where}",
                    params)
                self._conn.commit()
                count = cursor.rowcount
                if count:
                    logger.info(
                        f"LemmaBank: marked {count} lemmas as stale "
                        f"after upgrade")

    # ── 输出 ──

    def to_lean_preamble(self, records: list[LemmaRecord] = None,
                         max_lemmas: int = 20) -> str:
        """输出为 Lean4 preamble 代码"""
        if records is None:
            records = self.get_recent(max_lemmas)
        return "\n".join(r.to_lean() for r in records[:max_lemmas])

    def to_prompt_context(self, records: list[LemmaRecord] = None,
                          max_lemmas: int = 10) -> str:
        """输出为 LLM prompt 上下文"""
        if records is None:
            records = self.get_recent(max_lemmas)
        if not records:
            return ""
        parts = ["## Verified lemmas from previous proofs\n"]
        for r in records[:max_lemmas]:
            parts.append(f"```lean\n{r.to_lean()}\n```\n")
        return "\n".join(parts)

    # ── 统计 ──

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM lemmas").fetchone()[0]
            verified = self._conn.execute(
                "SELECT COUNT(*) FROM lemmas WHERE verified=1"
            ).fetchone()[0]
            return {
                "total": total,
                "verified": verified,
                "stale": total - verified,
                "db_path": self._db_path,
            }

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 内部方法 ──

    @staticmethod
    def _hash_statement(statement: str) -> str:
        normalized = re.sub(r'\s+', ' ', statement.strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()[:24]

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """提取关键词用于搜索"""
        # 提取标识符 (Lean4 命名风格)
        tokens = re.findall(r'[A-Z][a-zA-Z0-9]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*',
                            text)
        # 提取类型关键词
        type_tokens = re.findall(r'\b(Nat|Int|Real|List|Fin|Set|'
                                 r'Prop|Type|Bool|Option|Prod|Sum)\b', text)
        # 提取 tactic 名
        tactic_tokens = re.findall(r'\b(simp|ring|omega|norm_num|'
                                   r'linarith|induction|cases)\b', text)
        all_tokens = set(tokens + type_tokens + tactic_tokens)
        return sorted(t for t in all_tokens if len(t) >= 2)[:30]

    @staticmethod
    def _score_match(query_keywords: list[str],
                     record: LemmaRecord) -> float:
        """计算查询关键词与引理的匹配分数"""
        if not query_keywords:
            return 0.0
        record_text = (record.statement + " " +
                       " ".join(record.keywords)).lower()
        matches = sum(1 for kw in query_keywords
                      if kw.lower() in record_text)
        score = matches / len(query_keywords)
        # 加分: 被多次使用的引理更可能有用
        score += min(0.3, record.times_used * 0.05)
        return score

    def _row_to_record(self, row: tuple) -> LemmaRecord:
        return LemmaRecord(
            id=row[0], name=row[1], statement=row[2], proof=row[3],
            verified=bool(row[4]),
            source_problem=row[5], source_direction=row[6],
            tags=json.loads(row[7]) if row[7] else [],
            lean_version=row[8], mathlib_rev=row[9],
            times_used=row[10],
            created_at=row[11], last_used_at=row[12],
            statement_hash=row[13],
            keywords=json.loads(row[14]) if row[14] else [])
