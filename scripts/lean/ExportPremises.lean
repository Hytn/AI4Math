/- scripts/lean/ExportPremises.lean — 从编译后的环境导出全量声明

   用法 (在任何依赖 Mathlib 且已 lake build 的项目里, 例如 data/miniF2F):

       cd data/miniF2F
       cp ../../scripts/lean/ExportPremises.lean .
       lake env lean ExportPremises.lean
       # → 当前目录生成 mathlib_premises_raw.jsonl

   然后用 scripts/export_mathlib_premises_full.py 做后处理 (去重 /
   domain 标注 / 写入 data/premises/)。

   为什么不用正则抽源码 (scripts/export_mathlib_premises.py 的旧路径):
   正则会漏多行声明 / instance / 结构体投影, 且拿不到 pp 后的完整类型。
   从 Environment 导出是唯一能保证 ~10^5 量级全覆盖的方式。

   导出范围: theorem (thmInfo)。如需 def/instance, 改 `keep` 谓词。
   过滤: 内部名 (含 ._ / 以 _ 开头的组件)、自动生成的 eq/match/proof 辅助。
-/
import Mathlib
import Lean

open Lean Meta

def isInternalish (n : Name) : Bool :=
  n.isInternal
  || n.components.any (fun c =>
       match c with
       | .str _ s =>
           s.startsWith "_" || s.startsWith "match_" || s.startsWith "proof_"
           || s.startsWith "eq_def" || s == "eq_1" || s.startsWith "injEq"
       | _ => false)

/-- 极简 JSON 字符串转义 (足够覆盖 pp 输出)。 -/
def jsonEscape (s : String) : String :=
  s.foldl (fun acc c =>
    acc ++ match c with
      | '"'  => "\\\""
      | '\\' => "\\\\"
      | '\n' => "\\n"
      | '\t' => "\\t"
      | '\r' => "\\r"
      | c    => String.singleton c) ""

def exportPremises (outPath : System.FilePath) : MetaM Unit := do
  let env ← getEnv
  let h ← IO.FS.Handle.mk outPath .write
  let mut count := 0
  for (name, ci) in env.constants.toList do
    -- 只导出定理; 跳过内部/自动生成名。
    let keep := match ci with
      | .thmInfo _ => true
      | _ => false
    if keep && !isInternalish name then
      try
        -- pp 类型; maxHeartbeats 0 仅用于 pp (不涉及证明检查)。
        let ty ← withOptions (fun o => o.setNat `maxHeartbeats 400000) do
          ppExpr ci.type
        let modIdx := env.getModuleIdxFor? name
        let modName := match modIdx with
          | some i => toString (env.header.moduleNames[i.toNat]!)
          | none   => ""
        h.putStrLn s!"\{\"name\": \"{jsonEscape (toString name)}\", \"statement\": \"{jsonEscape (toString ty)}\", \"module\": \"{jsonEscape modName}\", \"kind\": \"theorem\"}"
        count := count + 1
        if count % 10000 == 0 then
          IO.println s!"  ... {count} declarations exported"
      catch _ =>
        pure ()  -- 个别 pp 失败 (宇宙多态边角) 直接跳过
  IO.println s!"Done: {count} theorem declarations → {outPath}"

#eval (exportPremises "mathlib_premises_raw.jsonl" : MetaM Unit)
