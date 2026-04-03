"""agent/executor/lean_env.py — Lean 4 执行环境管理"""
from __future__ import annotations
import subprocess, tempfile, os, logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LeanEnvironment:
    def __init__(self, mode: str = "docker", docker_image: str = "ai4math-lean",
                 docker_container: str = "", lean_project_dir: str = "/workspace/lean-project",
                 local_project_dir: str = "", timeout: int = 120):
        self.mode = mode
        self.docker_image = docker_image
        self.docker_container = docker_container
        self.lean_project_dir = lean_project_dir
        self.local_project_dir = local_project_dir
        self.timeout = timeout

    def compile(self, lean_code: str) -> tuple[int, str, str]:
        if self.mode == "docker":
            return self._compile_docker(lean_code)
        return self._compile_local(lean_code)

    def _compile_local(self, code: str) -> tuple[int, str, str]:
        project = self.local_project_dir or "."
        check_file = Path(project) / "AI4MathCheck.lean"
        try:
            check_file.write_text(code, encoding="utf-8")
            r = subprocess.run(["lake", "env", "lean", str(check_file)],
                               cwd=project, capture_output=True, text=True, timeout=self.timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Timeout after {self.timeout}s"
        except Exception as e:
            return 1, "", str(e)
        finally:
            if check_file.exists(): check_file.unlink()

    def _compile_docker(self, code: str) -> tuple[int, str, str]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
            f.write(code); tmp = f.name
        container_file = f"{self.lean_project_dir}/AI4MathCheck.lean"
        try:
            if self.docker_container:
                subprocess.run(["docker", "cp", tmp, f"{self.docker_container}:{container_file}"],
                               check=True, capture_output=True, timeout=10)
                r = subprocess.run(["docker", "exec", self.docker_container,
                                    "lake", "env", "lean", container_file],
                                   capture_output=True, text=True, timeout=self.timeout)
            else:
                r = subprocess.run(["docker", "run", "--rm", "-v", f"{tmp}:{container_file}",
                                    self.docker_image, "lake", "env", "lean", container_file],
                                   capture_output=True, text=True, timeout=self.timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Timeout after {self.timeout}s"
        except Exception as e:
            return 1, "", str(e)
        finally:
            os.unlink(tmp)
