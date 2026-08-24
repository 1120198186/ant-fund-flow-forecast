# Tianchi 231573 — Purchase Redemption Forecast

This repository is the clean workspace for Tianchi competition `231573`. The initial setup separates immutable competition data, self-built validation data, and every major modeling version.

Project root: `/Users/zyy/Documents/tianchi_231573_purchase_redemption`. It is an independent Git repository rather than a subdirectory of the pre-existing `movir-command` repository.

## Workspace map

```text
.
├── data/                         # the only competition/derived data zone
│   ├── raw/official/             # immutable supplied CSV files + manifest
│   └── derived/                  # reproducible generated datasets
├── validation/                   # the only self-built test zone
│   ├── splits/                   # split definitions, no labels
│   ├── locked/                   # generated holdout labels, ignored by Git
│   └── reports/                  # evaluation results, ignored by Git
├── versions/
│   ├── REGISTRY.md               # major-version status and lineage
│   └── v001_baseline/            # first major version
│       ├── src/
│       ├── configs/
│       ├── notebooks/
│       └── artifacts/             # models, metrics, predictions, submissions
├── scripts/                       # project-level setup and integrity tools
└── docs/                          # competition contract and decision records
```

## Non-negotiable rules

- Treat `data/raw/official/` as read-only. Verify it with `python3 scripts/verify_official_data.py`.
- Keep self-built test definitions and their labels under `validation/`; training code must exclude the locked interval.
- Start every major change with `python3 scripts/new_version.py vNNN_short_name --title "..."`.
- Keep every version-specific output under that version's `artifacts/` directory.
- Record the version lineage and status in `versions/REGISTRY.md`; never silently replace an older result.

## Current state

- Official data: copied from `/Users/zyy/Downloads/Purchase Redemption Data` without modifying the source.
- Validation: the first time-based holdout definition is documented under `validation/splits/`.
- Modeling: `v001_baseline` is scaffolded but no model has been trained yet.
- Competition facts and current evidence limits: see `docs/competition.md`.
