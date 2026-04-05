"""agent/plugins/loader.py — 证明策略插件的发现、加载与匹配

将证明策略、领域前提库、few-shot示例、钩子规则
从硬编码的 Python 代码变为可声明的 YAML 配置文件。

插件目录结构::

    plugins/strategies/number-theory/
    ├── plugin.yaml      # 元数据 + 匹配条件 + 参数覆盖
    ├── premises.jsonl    # 领域专用引理库
    ├── few_shot.md       # 领域特定的证明示例
    └── tactics.yaml      # 推荐 tactic 列表
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class StrategyPlugin:
    """一个已加载的策略插件"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    domains: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    difficulty: list[str] = field(default_factory=list)

    # 策略参数覆盖
    inherits: str = "medium"
    temperature: float = 0.7
    samples_per_round: int = 8
    custom_params: dict = field(default_factory=dict)

    # 领域知识
    extra_premises: list[dict] = field(default_factory=list)
    few_shot_examples: str = ""
    recommended_tactics: list[str] = field(default_factory=list)
    strategic_hint: str = ""

    # 钩子规则 (传给 HookManager.register_from_plugin)
    hooks: dict = field(default_factory=dict)

    # 匹配缓存
    _keyword_patterns: list = field(default_factory=list, repr=False)

    def match_score(self, theorem_statement: str) -> float:
        """计算本插件与给定定理的匹配分数 (0.0 = 不匹配)"""
        if not self._keyword_patterns:
            self._keyword_patterns = [
                re.compile(kw, re.IGNORECASE) for kw in self.keywords
            ]

        score = 0.0
        for pat in self._keyword_patterns:
            if pat.search(theorem_statement):
                score += 1.0

        # 归一化到 0-1
        if self._keyword_patterns:
            score /= len(self._keyword_patterns)

        return score

    def get_temperature(self) -> float:
        return self.custom_params.get("temperature", self.temperature)

    def get_strategic_hint(self) -> str:
        return self.strategic_hint


class PluginLoader:
    """插件发现与加载器"""

    def __init__(self, plugin_dirs: list[str] = None):
        self.plugin_dirs = plugin_dirs or ["plugins/strategies"]
        self._registry: dict[str, StrategyPlugin] = {}
        self._discovered = False

    def discover(self):
        """扫描插件目录, 加载所有有效插件"""
        for base_dir in self.plugin_dirs:
            base = Path(base_dir)
            if not base.exists():
                continue
            for plugin_dir in base.iterdir():
                if plugin_dir.is_dir():
                    manifest_path = plugin_dir / "plugin.yaml"
                    if manifest_path.exists():
                        self._load_plugin(plugin_dir, manifest_path)
        self._discovered = True
        logger.info(f"Loaded {len(self._registry)} strategy plugins: "
                    f"{list(self._registry.keys())}")

    def match(self, theorem_statement: str,
              classification: dict = None) -> list[StrategyPlugin]:
        """根据定理和分类信息匹配插件, 按匹配度排序"""
        if not self._discovered:
            self.discover()

        matches = []
        for plugin in self._registry.values():
            score = plugin.match_score(theorem_statement)

            # 如果有钩子分类结果, 用领域匹配加分
            if classification:
                domains = classification.get("domains", [])
                for d in domains:
                    if d in plugin.domains:
                        score += 0.5

            if score > 0:
                matches.append((score, plugin))

        matches.sort(key=lambda x: -x[0])
        return [p for _, p in matches]

    def get(self, name: str) -> Optional[StrategyPlugin]:
        return self._registry.get(name)

    def list_plugins(self) -> list[str]:
        if not self._discovered:
            self.discover()
        return list(self._registry.keys())

    def _load_plugin(self, plugin_dir: Path, manifest_path: Path):
        """加载单个插件"""
        try:
            if yaml:
                manifest = yaml.safe_load(manifest_path.read_text())
            else:
                # fallback: 简单解析 YAML
                manifest = self._simple_yaml_parse(
                    manifest_path.read_text())

            if not manifest or "name" not in manifest:
                return

            plugin = StrategyPlugin(
                name=manifest["name"],
                version=manifest.get("version", "1.0.0"),
                description=manifest.get("description", ""),
                domains=manifest.get("domain", {}).get("branches", []),
                keywords=manifest.get("domain", {}).get("keywords", []),
                difficulty=manifest.get("domain", {}).get("difficulty", []),
            )

            # 策略参数
            strategy = manifest.get("strategy", {})
            plugin.inherits = strategy.get("inherits", "medium")
            overrides = strategy.get("overrides", {})
            plugin.temperature = overrides.get("temperature", 0.7)
            plugin.samples_per_round = overrides.get(
                "samples_per_round", 8)
            plugin.custom_params = strategy.get("custom_params", {})

            # 领域知识文件
            premises_file = manifest.get("premises", {}).get("file", "")
            if premises_file:
                pf = plugin_dir / premises_file
                if pf.exists():
                    plugin.extra_premises = self._load_premises(pf)

            few_shot_file = manifest.get("few_shot", {}).get("file", "")
            if few_shot_file:
                ff = plugin_dir / few_shot_file
                if ff.exists():
                    plugin.few_shot_examples = ff.read_text()

            tactics_file = manifest.get("tactics", {}).get("file", "")
            if tactics_file:
                tf = plugin_dir / tactics_file
                if tf.exists():
                    plugin.recommended_tactics = self._load_tactics(tf)

            # 钩子规则
            plugin.hooks = manifest.get("hooks", {})

            # 战略提示
            if plugin.custom_params.get("strategic_hint"):
                plugin.strategic_hint = plugin.custom_params["strategic_hint"]

            self._registry[plugin.name] = plugin
            logger.debug(f"Loaded plugin: {plugin.name}")

        except Exception as e:
            logger.warning(f"Failed to load plugin from {plugin_dir}: {e}")

    @staticmethod
    def _load_premises(path: Path) -> list[dict]:
        premises = []
        for line in path.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    premises.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return premises

    @staticmethod
    def _load_tactics(path: Path) -> list[str]:
        if yaml:
            data = yaml.safe_load(path.read_text())
            return data if isinstance(data, list) else []
        return []

    @staticmethod
    def _simple_yaml_parse(text: str) -> dict:
        """极简 YAML 解析 (仅在 pyyaml 不可用时使用)"""
        result = {}
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, _, val = line.partition(":")
                key = key.strip().strip('"')
                val = val.strip().strip('"')
                if val:
                    result[key] = val
        return result
