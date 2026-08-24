# Project operating boundaries

This project is organized around three hard partitions.

The project root is also an independent Git repository. Do not absorb it into a parent competition repository or point its history at an unrelated project's remote.

1. `data/` is the only data zone. Files under `data/raw/official/` are immutable byte-for-byte copies of the supplied competition data. Generated datasets go under `data/derived/<dataset_version>/` and must include a manifest describing their source and generator.
2. `validation/` is the only self-built test zone. Split definitions belong in `validation/splits/`; locked labels belong in `validation/locked/`; evaluation reports belong in `validation/reports/`. Do not tune on a locked holdout after reading its result.
3. `versions/vNNN_name/` is the only major-version zone. Every major modeling direction gets its own numbered folder and keeps its code, configs, notebooks, models, metrics, predictions, and submissions inside that folder. Do not place version-specific files in the project root.

Keep a major version reproducible from its own folder plus `data/` and `validation/`. Reusable utilities may be promoted to a future shared package only after at least two versions need the same stable behavior; until then, keep implementation local to the version.

Never overwrite official raw data, a locked validation split, or an existing submission artifact. Create a new named artifact or a new major version instead.

Competition features and forecasts must not use public data after 2014-08-31. Treat this cutoff as part of the competition contract, not as an optional modeling choice.
