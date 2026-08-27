# Daily flow and rate feature table

One row per calendar day from 2013-07-01 through 2014-08-31 (427 rows).

Core definitions:

- `direct_inflow_amt`: `direct_purchase_amt`; active user transfer-in, excluding profit share.
- `profit_share_inflow_amt`: `share_amt`; passive profit/share credit.
- `profit_share_per_10000_opening_balance`: passive credit normalized by opening balance.
- `total_outflow_amt`: `total_redeem_amt`; consumption plus transfer-out.
- `transfer_outflow_amt`: `transfer_amt`; transfer-out behavior excluding consumption.

The table also retains payment-channel decompositions, user counts, fund yields,
all SHIBOR tenors, observation flags, and staleness. Same-day rate correlations
are descriptive only; predictive features must use values available before the
forecast date.
