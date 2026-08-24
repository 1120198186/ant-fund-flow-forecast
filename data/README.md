# Data zone

`data/` is the only place for competition data and derived datasets.

## `raw/official/`

- Contains byte-for-byte copies of the five supplied CSV files.
- CSV files are intentionally ignored by Git because the largest file is about 158 MB.
- `manifest.sha256` and `inventory.csv` are tracked so the local copy can be verified and reconstructed.
- Never edit, rename, clean, or resave these CSV files in place.

## `derived/`

Create one subfolder per reproducible dataset, for example `data/derived/d001_daily_aggregate/`. Each dataset folder must contain a small manifest with its generator command, source hashes, schema, and creation time. Do not put model outputs here.

