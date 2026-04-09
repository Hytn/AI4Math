# 前提检索 Embedding 设置指南 (Fix #10)

## 当前状态

默认的前提检索使用 **BM25 + 字符 n-gram TF-IDF** 混合方案，无需额外依赖。
这对 Lean4 标识符匹配已经相当有效，但在语义级别的检索（如"交换律" → `Nat.add_comm`）上表现有限。

## 启用 Dense Embedding 检索

### 1. 安装依赖

```bash
pip install sentence-transformers>=2.0 torch>=2.0
```

### 2. 下载推荐模型

对于数学定理检索，推荐以下模型（按效果/速度权衡排序）：

| 模型 | 维度 | 速度 | 数学效果 | 说明 |
|------|------|------|----------|------|
| `BAAI/bge-base-en-v1.5` | 768 | 中 | ★★★★ | 通用检索，数学效果好 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 快 | ★★★ | 轻量快速 |
| `intfloat/e5-large-v2` | 1024 | 慢 | ★★★★★ | 最高质量 |

```bash
# 预下载模型（可选，首次使用时会自动下载）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
```

### 3. 配置

在 `config/default.yaml` 中：

```yaml
prover:
  premise:
    mode: hybrid           # bm25 / embedding / hybrid
    embedding_model: "BAAI/bge-base-en-v1.5"
    embedding_weight: 0.6  # hybrid 模式下 embedding 的权重
    bm25_weight: 0.4       # hybrid 模式下 BM25 的权重
```

或通过环境变量：

```bash
export APE_PROVER__PREMISE__MODE=hybrid
export APE_PROVER__PREMISE__EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
```

### 4. 构建前提索引

首次使用 embedding 模式时，需要对 Mathlib 前提库建立向量索引：

```bash
python scripts/export_mathlib_premises.py --build-embeddings
```

这会在 `data/premises/` 下生成 `.npy` 向量文件，后续启动直接加载。

### 5. 自定义前提库

如果需要添加自定义引理到检索库：

```python
from prover.premise.selector import PremiseSelector

selector = PremiseSelector({
    "mode": "hybrid",
    "embedding_model": "BAAI/bge-base-en-v1.5"
})

# 添加自定义前提
selector.add_premises([
    {"name": "my_lemma", "statement": "theorem my_lemma : ..."},
])
```

## 不使用 Dense Embedding 的替代方案

如果不想安装 PyTorch（体积较大），默认的 n-gram TF-IDF 混合检索
对于大部分场景已经足够。它的优势：

- 零额外依赖
- 毫秒级检索速度
- 对 Lean4 的 CamelCase / dot-notation 做了专门优化
- 在 miniF2F 级别的题目上，前提召回率与 dense embedding 差距不大

主要劣势是无法做语义级别的匹配（如自然语言描述 → 形式化引理）。
