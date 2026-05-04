"""tests/conftest.py — 共享测试配置

只做 sys.path 注入。legacy CIC 内核相关的 fixture (mk_standard_env /
mk_standard_state) 已随 v8 清理移除;那些 fixture 唯一服务的测试都
已删除 (依赖被删的 engine.core/state/kernel/tactic)。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
