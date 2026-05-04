"""benchmarks/ — 统一基准加载与评测指标

支持的数据集 (见 benchmarks.loader.load_benchmark):
    builtin         5 题, 内置冒烟测试
    minif2f         244 题, 竞赛数学
    putnambench     672 题, Putnam 1962-2024
    proofnet        360 题, 本科数学
    fate-m / -h / -x  150 / 100 / 100 题, 抽象代数 (本科 → 研究级)
    formalmath      5560 题, 多领域

入口::

    from benchmarks.loader import load_benchmark
    problems = load_benchmark("minif2f", split="test")
    # → list[BenchmarkProblem]
"""
