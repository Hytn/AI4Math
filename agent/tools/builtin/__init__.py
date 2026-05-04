"""agent/tools/builtin/ — Built-in tool implementations.

Tools are registered selectively per-profile by
``prover.unified.tool_kits.build_tool_registry``, which imports each
tool class directly. There is no "register everything at once" entry
point — that pattern was removed in v10 (zero callers, and the
selective per-profile registration is the actual contract).

To add a new built-in tool:
    1. Implement it as a subclass of ``agent.tools.base.Tool`` in this dir.
    2. Add a ``ToolKit`` enum entry in ``prover/unified/profiles.py``.
    3. Register it in ``prover/unified/tool_kits.py::_build_tool``.
"""
