"""agent/executor/sandbox.py — 安全沙箱执行"""
from __future__ import annotations
import subprocess

class Sandbox:
    def __init__(self, mode: str = "subprocess"):
        self.mode = mode

    def run(self, cmd: list[str], timeout: int = 120, cwd: str = None) -> tuple[int, str, str]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Timeout after {timeout}s"
