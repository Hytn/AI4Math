"""tests/test_yaml_profile_templates.py — V5 YAML profile templates.

Pins the contract that every YAML in ``config/profiles/`` round-trips
through :func:`load_profile_from_yaml` to a Profile that is structurally
equal to the corresponding ``PRESETS[name]`` (when one exists).

Closes ``REFACTOR_REPORT.md`` § 九.3 / ``INFRA_MERGE_V3_REPORT.md``
"YAML profile templates" V5+ item.
"""
from __future__ import annotations

import os
from dataclasses import asdict

import pytest

from prover.unified.profiles import (
    PRESETS, load_profile_from_yaml,
)


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(HERE, "config", "profiles")


def _yaml_files() -> list[str]:
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(
        os.path.join(PROFILE_DIR, f)
        for f in os.listdir(PROFILE_DIR)
        if f.endswith(".yaml"))


# ─────────────────────────────────────────────────────────────────────
# 1. Coverage: every shipped PRESET has a YAML
# ─────────────────────────────────────────────────────────────────────


class TestCoverage:
    def test_every_shipped_preset_has_a_yaml(self):
        """If a profile name is in ``PRESETS``, there must be a
        ``config/profiles/<name>.yaml`` on disk so users can copy
        and customize without reading Python."""
        files = {os.path.basename(p)[:-5] for p in _yaml_files()}
        for name in PRESETS:
            assert name in files, (
                f"PRESETS[{name!r}] has no ``config/profiles/{name}.yaml`` "
                f"template — regenerate via "
                f"``python scripts/dump_profile_yamls.py``")


# ─────────────────────────────────────────────────────────────────────
# 2. Structural round-trip
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "yaml_path", _yaml_files(),
    ids=[os.path.basename(p) for p in _yaml_files()])
class TestRoundTrip:
    def test_loads_without_error(self, yaml_path):
        prof = load_profile_from_yaml(yaml_path)
        assert prof.name, f"{yaml_path}: empty name"

    def test_round_trips_to_matching_preset(self, yaml_path):
        prof = load_profile_from_yaml(yaml_path)
        if prof.name not in PRESETS:
            pytest.skip(
                f"{prof.name!r} is a user-defined YAML, not a shipped preset")

        loaded = asdict(prof)
        canonical = asdict(PRESETS[prof.name])
        # The dump uses ToolKit string values; canonical retains the
        # ToolKit enum. Compare on tool .value strings.
        loaded["tools"] = [
            t if isinstance(t, str) else t.value for t in loaded["tools"]]
        canonical["tools"] = [
            t if isinstance(t, str) else t.value for t in canonical["tools"]]
        assert loaded == canonical, (
            f"{yaml_path}: drift between YAML and Python preset. "
            f"Re-run ``python scripts/dump_profile_yamls.py`` to refresh.")


# ─────────────────────────────────────────────────────────────────────
# 3. Required fields
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "yaml_path", _yaml_files(),
    ids=[os.path.basename(p) for p in _yaml_files()])
class TestStructuralInvariants:
    def test_has_required_top_level_fields(self, yaml_path):
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "name" in data
        assert "tools" in data
        assert "max_turns" in data
        assert "framing" in data

    def test_tools_are_string_enum_values(self, yaml_path):
        from prover.unified.profiles import ToolKit
        valid = {t.value for t in ToolKit}
        prof = load_profile_from_yaml(yaml_path)
        for t in prof.tools:
            assert t.value in valid, (
                f"{yaml_path}: unknown ToolKit value {t.value!r}")


# ─────────────────────────────────────────────────────────────────────
# 4. CLI-style use
# ─────────────────────────────────────────────────────────────────────


class TestRegisterFlow:
    def test_load_then_register_then_get(self):
        """The README-advertised pattern must work end-to-end."""
        from prover.unified.profiles import register_profile, get_profile
        path = os.path.join(PROFILE_DIR, "whole_proof_repair.yaml")
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")
        # Load + re-register under a fresh name to avoid clobbering.
        prof = load_profile_from_yaml(path)
        from dataclasses import replace
        prof = replace(prof, name="_v5_test_yaml")
        register_profile(prof)
        try:
            got = get_profile("_v5_test_yaml")
            assert got is prof
        finally:
            # Best-effort cleanup so we don't leak into other tests.
            from prover.unified.profiles import PRESETS as _P
            _P.pop("_v5_test_yaml", None)
