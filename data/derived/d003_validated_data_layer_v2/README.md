# Validated derived data layer

Typed, date-sorted Parquet tables derived only from frozen official data through
2014-08-31. Raw values are preserved. Structural category nulls remain nullable;
explicit `_filled` columns provide zero-filled values for computation.

`shibor_daily.parquet` uses historical forward fill only and includes observation
flags. `balance_reconciliation_delta` exposes source inconsistencies without
silently changing official balances.

Treat this directory as read-only. Rebuild with `scripts/build_data_foundation.py`
into a new dataset version when transformation logic changes.
