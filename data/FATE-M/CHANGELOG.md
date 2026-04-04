# Changelog

## 1.2.0 - 2026-02-23

### Added
- Added support for all Lean minor versions from [v4.16.0] through [v4.28.0].

## [1.1.0] - 2026-02-23

### Added
- This CHANGELOG file.
- Added an entry to the JSON file recording the compatible Lean version.

### Fixed
- Resolved misalignments between formal and natural language statements in the following problems:
  - **M-15**
  - **M-134**

### Changed
- Removed void references from the informal statements of M-79 and M-150. The formal statements are unaffected.
- Corrected minor formatting issues in comments and standardized all comments to use `/-- ... -/`.

## [1.0.0] - 2025-08-06

### Added
- Refactored the [original benchmark] to create 150 FATE-M benchmark questions.
- A PDF file containing both natural language and formalization for human reading.
- A JSON file for machine reading.
- CI to automatically run Lean builds on push.

[v4.28.0]: https://github.com/frenzymath/FATE-M/tree/v4.28.0
[v4.16.0]: https://github.com/frenzymath/FATE-M/tree/v4.16.0
[1.1.0]: https://github.com/frenzymath/FATE-M/tree/v4.16.0
[1.0.0]: https://github.com/frenzymath/FATE-M/tree/initial-release
[original benchmark]: https://github.com/frenzymath/REAL-Prover/blob/main/Realprover/data/fate_m.jsonl
