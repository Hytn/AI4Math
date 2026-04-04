"""agent/executor/sandbox.py — 安全沙箱执行

在受限环境中执行 Lean4 编译和外部命令。
支持 subprocess 和 Docker 模式。
"""
from __future__ import annotations
import subprocess
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_ms: int = 0


class Sandbox:
    """Safe execution sandbox for external commands.

    Modes:
        'subprocess': Direct subprocess execution (basic isolation)
        'docker':     Docker container execution (strong isolation)
    """

    def __init__(self, mode: str = "subprocess", docker_image: str = "",
                 allowed_commands: list[str] = None):
        self.mode = mode
        self.docker_image = docker_image
        self.allowed_commands = set(allowed_commands or [
            "lean", "lake", "sage", "python3", "cat", "echo"])

    def run(self, cmd: list[str], timeout: int = 120,
            cwd: str = None, input_data: str = None) -> SandboxResult:
        """Execute a command in the sandbox.

        Args:
            cmd: Command and arguments.
            timeout: Timeout in seconds.
            cwd: Working directory.
            input_data: Stdin data.

        Returns:
            SandboxResult with output and status.
        """
        import time

        # Validate command
        if cmd and cmd[0] not in self.allowed_commands:
            return SandboxResult(
                returncode=-1, stdout="",
                stderr=f"Command '{cmd[0]}' not in allowed list: {self.allowed_commands}")

        if self.mode == "docker" and self.docker_image:
            return self._run_docker(cmd, timeout, cwd, input_data)

        return self._run_subprocess(cmd, timeout, cwd, input_data)

    def _run_subprocess(self, cmd: list[str], timeout: int,
                        cwd: str = None, input_data: str = None) -> SandboxResult:
        import time
        start = time.time()
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=cwd, input=input_data,
                env={**os.environ, "LANG": "en_US.UTF-8"})
            duration = int((time.time() - start) * 1000)
            return SandboxResult(
                returncode=r.returncode, stdout=r.stdout, stderr=r.stderr,
                duration_ms=duration)
        except subprocess.TimeoutExpired:
            duration = int((time.time() - start) * 1000)
            return SandboxResult(
                returncode=-1, stdout="",
                stderr=f"Timeout after {timeout}s",
                timed_out=True, duration_ms=duration)
        except FileNotFoundError:
            return SandboxResult(
                returncode=-1, stdout="",
                stderr=f"Command not found: {cmd[0] if cmd else '?'}")

    def _run_docker(self, cmd: list[str], timeout: int,
                    cwd: str = None, input_data: str = None) -> SandboxResult:
        docker_cmd = ["docker", "run", "--rm", "--network=none",
                       f"--memory=4g", f"--cpus=2",
                       self.docker_image] + cmd
        return self._run_subprocess(docker_cmd, timeout, cwd, input_data)
