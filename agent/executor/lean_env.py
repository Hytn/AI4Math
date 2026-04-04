"""agent/executor/lean_env.py — Real Lean 4 execution environment.

Handles:
  1. Detecting elan / lean4 / lake installation
  2. Initializing a Lake project with Mathlib
  3. Building and caching Mathlib oleans
  4. Compiling single Lean4 files for verification
  5. Health checks and status reporting

Usage:
    env = LeanEnvironment.create()     # auto-detect best mode
    env.ensure_ready()                 # install/build if needed
    rc, stdout, stderr = env.compile("import Mathlib\\ntheorem t : True := trivial")
"""
from __future__ import annotations
import subprocess
import tempfile
import os
import shutil
import logging
import json
import time
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LeanStatus:
    """Status of the Lean4 environment."""
    elan_installed: bool = False
    lean_installed: bool = False
    lean_version: str = ""
    lake_installed: bool = False
    project_initialized: bool = False
    mathlib_available: bool = False
    cache_built: bool = False
    mode: str = "unavailable"  # local, docker, unavailable
    project_dir: str = ""
    issues: list[str] = field(default_factory=list)


class LeanEnvironment:
    """Lean 4 execution environment with real compilation support."""

    def __init__(self, mode: str = "auto", project_dir: str = "",
                 docker_image: str = "ai4math-lean",
                 docker_container: str = "",
                 timeout: int = 120,
                 mathlib: bool = True):
        self.mode = mode
        self.project_dir = project_dir or str(Path.home() / ".ai4math" / "lean-project")
        self.docker_image = docker_image
        self.docker_container = docker_container
        self.timeout = timeout
        self.mathlib = mathlib
        self._status: LeanStatus | None = None

    @staticmethod
    def create(project_dir: str = "", mathlib: bool = True) -> LeanEnvironment:
        """Auto-detect and create the best available environment."""
        env = LeanEnvironment(mode="auto", project_dir=project_dir, mathlib=mathlib)
        env._detect_mode()
        return env

    # ── Status & Health ──

    def status(self) -> LeanStatus:
        """Get current environment status."""
        if self._status is None:
            self._status = self._check_status()
        return self._status

    def _check_status(self) -> LeanStatus:
        s = LeanStatus()

        # Check elan
        s.elan_installed = shutil.which("elan") is not None

        # Check lean
        lean_path = shutil.which("lean")
        s.lean_installed = lean_path is not None
        if s.lean_installed:
            try:
                r = subprocess.run(["lean", "--version"], capture_output=True,
                                    text=True, timeout=10)
                if r.returncode == 0:
                    s.lean_version = r.stdout.strip().split("\n")[0]
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass

        # Check lake
        s.lake_installed = shutil.which("lake") is not None

        # Check project
        proj = Path(self.project_dir)
        s.project_dir = str(proj)
        s.project_initialized = (proj / "lakefile.lean").exists()
        s.mathlib_available = (proj / "lake-packages").exists() or (proj / ".lake" / "packages").exists()
        s.cache_built = (proj / ".lake" / "build").exists()

        # Determine mode
        if s.lean_installed and s.lake_installed:
            s.mode = "local"
        elif shutil.which("docker") is not None:
            s.mode = "docker"
        else:
            s.mode = "unavailable"
            s.issues.append(
                "Neither lean4 nor docker found. "
                "Install lean4 via: curl https://elan.lean-lang.org/elan-init.sh -sSf | sh")

        self.mode = s.mode
        return s

    def _detect_mode(self):
        s = self._check_status()
        self._status = s
        self.mode = s.mode

    def is_available(self) -> bool:
        return self.status().mode != "unavailable"

    # ── Project Setup ──

    def ensure_ready(self) -> bool:
        """Ensure the Lean4 environment is ready for compilation.

        Creates project, downloads Mathlib cache if needed.
        Returns True if ready.
        """
        s = self.status()

        if s.mode == "unavailable":
            logger.error("Lean4 not available. Install via elan.")
            return False

        if s.mode == "docker":
            logger.info("Using Docker mode — project setup is inside container")
            return True

        # Local mode: ensure project exists
        proj = Path(self.project_dir)
        if not s.project_initialized:
            logger.info(f"Initializing Lean4 project at {proj}...")
            if not self._init_project(proj):
                return False

        if self.mathlib and not s.cache_built:
            logger.info("Fetching Mathlib cache (this may take a few minutes)...")
            self._fetch_cache(proj)

        # Refresh status
        self._status = None
        return True

    def _init_project(self, proj: Path) -> bool:
        """Initialize a new Lake project with Mathlib."""
        try:
            proj.mkdir(parents=True, exist_ok=True)

            # Create lakefile
            mathlib_dep = ""
            if self.mathlib:
                mathlib_dep = """
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
"""
            lakefile_content = f"""import Lake
open Lake DSL

package «ai4math» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]
{mathlib_dep}
@[default_target]
lean_lib «AI4MathCheck» where
  srcDir := "."
"""
            (proj / "lakefile.lean").write_text(lakefile_content)

            # Create lean-toolchain
            # Get current lean version
            try:
                r = subprocess.run(["lean", "--version"], capture_output=True,
                                    text=True, timeout=10)
                version_line = r.stdout.strip().split("\n")[0]
                # Extract version like "leanprover/lean4:v4.x.y"
                import re
                m = re.search(r'v[\d.]+(-\w+)?', version_line)
                toolchain = f"leanprover/lean4:{m.group()}" if m else "leanprover/lean4:stable"
            except (FileNotFoundError, subprocess.TimeoutExpired,
                    AttributeError, OSError):
                toolchain = "leanprover/lean4:stable"

            (proj / "lean-toolchain").write_text(toolchain + "\n")

            # Run lake update
            r = subprocess.run(["lake", "update"], cwd=str(proj),
                                capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                logger.warning(f"lake update warning: {r.stderr[:200]}")

            return True

        except Exception as e:
            logger.error(f"Failed to init project: {e}")
            return False

    def _fetch_cache(self, proj: Path):
        """Fetch pre-built Mathlib oleans."""
        try:
            r = subprocess.run(["lake", "exe", "cache", "get"],
                                cwd=str(proj), capture_output=True, text=True,
                                timeout=600)
            if r.returncode == 0:
                logger.info("Mathlib cache fetched successfully")
            else:
                logger.warning(f"Cache fetch failed, will build from source: {r.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("Cache fetch timed out")
        except FileNotFoundError:
            logger.warning("lake not found, skipping cache")

    # ── Compilation ──

    def compile(self, lean_code: str) -> tuple[int, str, str]:
        """Compile a Lean4 source string.

        Returns: (returncode, stdout, stderr)
        """
        if self.mode == "docker":
            return self._compile_docker(lean_code)
        elif self.mode == "local":
            return self._compile_local(lean_code)
        else:
            return 1, "", "Lean4 environment not available"

    def _compile_local(self, code: str) -> tuple[int, str, str]:
        """Compile using local lean4 installation."""
        proj = Path(self.project_dir)
        check_file = proj / "AI4MathCheck.lean"

        try:
            check_file.write_text(code, encoding="utf-8")

            # Use lake env lean for proper environment setup
            cmd = ["lake", "env", "lean", str(check_file)]
            r = subprocess.run(cmd, cwd=str(proj), capture_output=True,
                                text=True, timeout=self.timeout)
            return r.returncode, r.stdout, r.stderr

        except subprocess.TimeoutExpired:
            return 1, "", f"Compilation timed out after {self.timeout}s"
        except FileNotFoundError:
            # Fallback: try lean directly without lake
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".lean",
                                                   delete=False) as f:
                    f.write(code)
                    tmp = f.name
                r = subprocess.run(["lean", tmp], capture_output=True,
                                    text=True, timeout=self.timeout)
                os.unlink(tmp)
                return r.returncode, r.stdout, r.stderr
            except Exception as e:
                return 1, "", f"Compilation failed: {e}"
        except Exception as e:
            return 1, "", str(e)
        finally:
            if check_file.exists():
                try:
                    check_file.unlink()
                except OSError:
                    pass

    def _compile_docker(self, code: str) -> tuple[int, str, str]:
        """Compile using Docker container."""
        container_file = "/workspace/lean-project/AI4MathCheck.lean"
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".lean",
                                               delete=False) as f:
                f.write(code)
                tmp = f.name

            if self.docker_container:
                # Use existing container
                subprocess.run(
                    ["docker", "cp", tmp, f"{self.docker_container}:{container_file}"],
                    check=True, capture_output=True, timeout=10)
                r = subprocess.run(
                    ["docker", "exec", self.docker_container,
                     "lake", "env", "lean", container_file],
                    capture_output=True, text=True, timeout=self.timeout)
            else:
                # Run fresh container
                r = subprocess.run(
                    ["docker", "run", "--rm",
                     "--network=none", "--memory=4g", "--cpus=2",
                     "-v", f"{tmp}:{container_file}",
                     self.docker_image,
                     "lake", "env", "lean", container_file],
                    capture_output=True, text=True, timeout=self.timeout)

            return r.returncode, r.stdout, r.stderr

        except subprocess.TimeoutExpired:
            return 1, "", f"Docker compilation timed out after {self.timeout}s"
        except Exception as e:
            return 1, "", str(e)
        finally:
            try:
                os.unlink(tmp)
            except (OSError, UnboundLocalError):
                pass

    # ── Convenience ──

    def check_theorem(self, statement: str, proof: str,
                      imports: str = "import Mathlib") -> tuple[bool, str]:
        """High-level: check if a proof is valid.

        Returns (success, error_message).
        """
        code = f"{imports}\n\n{statement} {proof}\n"
        rc, stdout, stderr = self.compile(code)
        if rc == 0 and "error" not in stderr.lower():
            return True, ""
        return False, stderr[:500]

    def __repr__(self):
        s = self.status()
        return (f"LeanEnvironment(mode={s.mode}, lean={s.lean_version}, "
                f"project={s.project_dir})")
