# Self-built test zone

This zone is deliberately separate from both official data and model versions.

- `splits/`: label-free, reviewable definitions of training and holdout windows.
- `locked/`: generated ground-truth files. These are ignored by Git and must not be imported by training code.
- `reports/`: evaluation output. These are ignored by Git to reduce accidental tuning to the holdout.

The initial split mirrors the exact 30-day forecast horizon: train through 2014-08-01 and hold out 2014-08-02 through 2014-08-31. Generate it with:

```bash
python3 scripts/build_holdout.py --split validation/splits/holdout_2014_08_v1.json
```

Reading or using holdout labels for feature engineering is leakage. The official score cannot be reproduced exactly because its error-to-score mapping is not public. If the validation design changes materially, create a new split file rather than editing the old one.
