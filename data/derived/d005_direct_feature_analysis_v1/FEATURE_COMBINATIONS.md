# Feature combination catalog

Predictive features must use information available strictly before the forecast date.
Same-day component ratios are useful for descriptive analysis only; use lagged forms in models.

## P0

### raw_demographic

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `sex` | `sex` | user | Safe as static input; fit categorical encoding inside each fold. |
| `city` | `city` | user | Safe as static input; fit categorical encoding inside each fold. |
| `constellation` | `constellation` | user | Safe as static input; fit categorical encoding inside each fold. |

### demographic_cross

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `sex_x_city` | `sex + city` | user | Pool cells below 50 training users; create mapping within each fold. |
| `sex_x_constellation` | `sex + constellation` | user | Pool cells below 50 training users; create mapping within each fold. |
| `city_x_constellation` | `city + constellation` | user | Pool cells below 50 training users; create mapping within each fold. |

### rate_lag

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `mfd_daily_yield_lag1` | `lag(mfd_daily_yield, 1)` | date | Only values published by forecast cutoff may be used. |
| `mfd_daily_yield_lag7` | `lag(mfd_daily_yield, 7)` | date | Only values published by forecast cutoff may be used. |
| `mfd_7daily_yield_lag1` | `lag(mfd_7daily_yield, 1)` | date | Only values published by forecast cutoff may be used. |
| `mfd_7daily_yield_lag7` | `lag(mfd_7daily_yield, 7)` | date | Only values published by forecast cutoff may be used. |
| `Interest_O_N_lag1` | `lag(Interest_O_N, 1)` | date | Only values published by forecast cutoff may be used. |
| `Interest_O_N_lag7` | `lag(Interest_O_N, 7)` | date | Only values published by forecast cutoff may be used. |
| `Interest_1_W_lag1` | `lag(Interest_1_W, 1)` | date | Only values published by forecast cutoff may be used. |
| `Interest_1_W_lag7` | `lag(Interest_1_W, 7)` | date | Only values published by forecast cutoff may be used. |
| `Interest_2_W_lag1` | `lag(Interest_2_W, 1)` | date | Only values published by forecast cutoff may be used. |
| `Interest_2_W_lag7` | `lag(Interest_2_W, 7)` | date | Only values published by forecast cutoff may be used. |
| `Interest_1_M_lag1` | `lag(Interest_1_M, 1)` | date | Only values published by forecast cutoff may be used. |
| `Interest_1_M_lag7` | `lag(Interest_1_M, 7)` | date | Only values published by forecast cutoff may be used. |
| `Interest_3_M_lag1` | `lag(Interest_3_M, 1)` | date | Only values published by forecast cutoff may be used. |
| `Interest_3_M_lag7` | `lag(Interest_3_M, 7)` | date | Only values published by forecast cutoff may be used. |
| `Interest_6_M_lag1` | `lag(Interest_6_M, 1)` | date | Only values published by forecast cutoff may be used. |
| `Interest_6_M_lag7` | `lag(Interest_6_M, 7)` | date | Only values published by forecast cutoff may be used. |
| `Interest_9_M_lag1` | `lag(Interest_9_M, 1)` | date | Only values published by forecast cutoff may be used. |
| `Interest_9_M_lag7` | `lag(Interest_9_M, 7)` | date | Only values published by forecast cutoff may be used. |
| `Interest_1_Y_lag1` | `lag(Interest_1_Y, 1)` | date | Only values published by forecast cutoff may be used. |
| `Interest_1_Y_lag7` | `lag(Interest_1_Y, 7)` | date | Only values published by forecast cutoff may be used. |

### rate_change

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `mfd_daily_yield_delta1_lag1` | `lag(mfd_daily_yield - lag(mfd_daily_yield,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `mfd_daily_yield_delta7_lag1` | `lag(mfd_daily_yield - lag(mfd_daily_yield,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `mfd_7daily_yield_delta1_lag1` | `lag(mfd_7daily_yield - lag(mfd_7daily_yield,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `mfd_7daily_yield_delta7_lag1` | `lag(mfd_7daily_yield - lag(mfd_7daily_yield,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_O_N_delta1_lag1` | `lag(Interest_O_N - lag(Interest_O_N,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_O_N_delta7_lag1` | `lag(Interest_O_N - lag(Interest_O_N,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_1_W_delta1_lag1` | `lag(Interest_1_W - lag(Interest_1_W,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_1_W_delta7_lag1` | `lag(Interest_1_W - lag(Interest_1_W,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_2_W_delta1_lag1` | `lag(Interest_2_W - lag(Interest_2_W,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_2_W_delta7_lag1` | `lag(Interest_2_W - lag(Interest_2_W,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_1_M_delta1_lag1` | `lag(Interest_1_M - lag(Interest_1_M,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_1_M_delta7_lag1` | `lag(Interest_1_M - lag(Interest_1_M,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_3_M_delta1_lag1` | `lag(Interest_3_M - lag(Interest_3_M,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_3_M_delta7_lag1` | `lag(Interest_3_M - lag(Interest_3_M,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_6_M_delta1_lag1` | `lag(Interest_6_M - lag(Interest_6_M,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_6_M_delta7_lag1` | `lag(Interest_6_M - lag(Interest_6_M,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_9_M_delta1_lag1` | `lag(Interest_9_M - lag(Interest_9_M,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_9_M_delta7_lag1` | `lag(Interest_9_M - lag(Interest_9_M,7), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_1_Y_delta1_lag1` | `lag(Interest_1_Y - lag(Interest_1_Y,1), 1)` | date | Difference first, then lag the result before forecast use. |
| `Interest_1_Y_delta7_lag1` | `lag(Interest_1_Y - lag(Interest_1_Y,7), 1)` | date | Difference first, then lag the result before forecast use. |

### rate_spread_curve

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `alipay_vs_shibor_on` | `lag(mfd_7daily_yield - Interest_O_N, 1)` | date | Use lagged published rates; same-day version is descriptive only. |
| `alipay_vs_shibor_1w` | `lag(mfd_7daily_yield - Interest_1_W, 1)` | date | Use lagged published rates; same-day version is descriptive only. |
| `alipay_vs_shibor_1m` | `lag(mfd_7daily_yield - Interest_1_M, 1)` | date | Use lagged published rates; same-day version is descriptive only. |
| `alipay_vs_shibor_3m` | `lag(mfd_7daily_yield - Interest_3_M, 1)` | date | Use lagged published rates; same-day version is descriptive only. |
| `shibor_slope_1y_on` | `lag(Interest_1_Y - Interest_O_N, 1)` | date | Use lagged published rates; same-day version is descriptive only. |
| `shibor_slope_3m_on` | `lag(Interest_3_M - Interest_O_N, 1)` | date | Use lagged published rates; same-day version is descriptive only. |
| `shibor_slope_1m_1w` | `lag(Interest_1_M - Interest_1_W, 1)` | date | Use lagged published rates; same-day version is descriptive only. |
| `shibor_curve_curvature` | `lag(2 * Interest_3_M - Interest_O_N - Interest_1_Y, 1)` | date | Use lagged published rates; same-day version is descriptive only. |

### target_history

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `total_purchase_amt_lag1` | `lag(daily_sum(total_purchase_amt), 1)` | date | Compute independently inside each rolling split. |
| `total_purchase_amt_lag7` | `lag(daily_sum(total_purchase_amt), 7)` | date | Compute independently inside each rolling split. |
| `total_purchase_amt_lag14` | `lag(daily_sum(total_purchase_amt), 14)` | date | Compute independently inside each rolling split. |
| `total_purchase_amt_lag28` | `lag(daily_sum(total_purchase_amt), 28)` | date | Compute independently inside each rolling split. |
| `total_purchase_amt_rolling_mean7` | `rolling_mean(lag(daily_sum(total_purchase_amt),1), 7)` | date | Shift one day before rolling. |
| `total_purchase_amt_rolling_mean14` | `rolling_mean(lag(daily_sum(total_purchase_amt),1), 14)` | date | Shift one day before rolling. |
| `total_purchase_amt_rolling_mean28` | `rolling_mean(lag(daily_sum(total_purchase_amt),1), 28)` | date | Shift one day before rolling. |
| `total_redeem_amt_lag1` | `lag(daily_sum(total_redeem_amt), 1)` | date | Compute independently inside each rolling split. |
| `total_redeem_amt_lag7` | `lag(daily_sum(total_redeem_amt), 7)` | date | Compute independently inside each rolling split. |
| `total_redeem_amt_lag14` | `lag(daily_sum(total_redeem_amt), 14)` | date | Compute independently inside each rolling split. |
| `total_redeem_amt_lag28` | `lag(daily_sum(total_redeem_amt), 28)` | date | Compute independently inside each rolling split. |
| `total_redeem_amt_rolling_mean7` | `rolling_mean(lag(daily_sum(total_redeem_amt),1), 7)` | date | Shift one day before rolling. |
| `total_redeem_amt_rolling_mean14` | `rolling_mean(lag(daily_sum(total_redeem_amt),1), 14)` | date | Shift one day before rolling. |
| `total_redeem_amt_rolling_mean28` | `rolling_mean(lag(daily_sum(total_redeem_amt),1), 28)` | date | Shift one day before rolling. |
| `direct_purchase_amt_lag1` | `lag(daily_sum(direct_purchase_amt), 1)` | date | Compute independently inside each rolling split. |
| `direct_purchase_amt_lag7` | `lag(daily_sum(direct_purchase_amt), 7)` | date | Compute independently inside each rolling split. |
| `direct_purchase_amt_lag14` | `lag(daily_sum(direct_purchase_amt), 14)` | date | Compute independently inside each rolling split. |
| `direct_purchase_amt_lag28` | `lag(daily_sum(direct_purchase_amt), 28)` | date | Compute independently inside each rolling split. |
| `direct_purchase_amt_rolling_mean7` | `rolling_mean(lag(daily_sum(direct_purchase_amt),1), 7)` | date | Shift one day before rolling. |
| `direct_purchase_amt_rolling_mean14` | `rolling_mean(lag(daily_sum(direct_purchase_amt),1), 14)` | date | Shift one day before rolling. |
| `direct_purchase_amt_rolling_mean28` | `rolling_mean(lag(daily_sum(direct_purchase_amt),1), 28)` | date | Shift one day before rolling. |
| `purchase_bank_amt_lag1` | `lag(daily_sum(purchase_bank_amt), 1)` | date | Compute independently inside each rolling split. |
| `purchase_bank_amt_lag7` | `lag(daily_sum(purchase_bank_amt), 7)` | date | Compute independently inside each rolling split. |
| `purchase_bank_amt_lag14` | `lag(daily_sum(purchase_bank_amt), 14)` | date | Compute independently inside each rolling split. |
| `purchase_bank_amt_lag28` | `lag(daily_sum(purchase_bank_amt), 28)` | date | Compute independently inside each rolling split. |
| `purchase_bank_amt_rolling_mean7` | `rolling_mean(lag(daily_sum(purchase_bank_amt),1), 7)` | date | Shift one day before rolling. |
| `purchase_bank_amt_rolling_mean14` | `rolling_mean(lag(daily_sum(purchase_bank_amt),1), 14)` | date | Shift one day before rolling. |
| `purchase_bank_amt_rolling_mean28` | `rolling_mean(lag(daily_sum(purchase_bank_amt),1), 28)` | date | Shift one day before rolling. |
| `transfer_amt_lag1` | `lag(daily_sum(transfer_amt), 1)` | date | Compute independently inside each rolling split. |
| `transfer_amt_lag7` | `lag(daily_sum(transfer_amt), 7)` | date | Compute independently inside each rolling split. |
| `transfer_amt_lag14` | `lag(daily_sum(transfer_amt), 14)` | date | Compute independently inside each rolling split. |
| `transfer_amt_lag28` | `lag(daily_sum(transfer_amt), 28)` | date | Compute independently inside each rolling split. |
| `transfer_amt_rolling_mean7` | `rolling_mean(lag(daily_sum(transfer_amt),1), 7)` | date | Shift one day before rolling. |
| `transfer_amt_rolling_mean14` | `rolling_mean(lag(daily_sum(transfer_amt),1), 14)` | date | Shift one day before rolling. |
| `transfer_amt_rolling_mean28` | `rolling_mean(lag(daily_sum(transfer_amt),1), 28)` | date | Shift one day before rolling. |

### behavior_composition

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `bank_purchase_share_lag1` | `lag(daily(purchase_bank_amt / direct_purchase_amt), 1)` | date | Same-day formula is target leakage; only lagged/rolling forms are predictive. |
| `balance_purchase_share_lag1` | `lag(daily(purchase_bal_amt / direct_purchase_amt), 1)` | date | Same-day formula is target leakage; only lagged/rolling forms are predictive. |
| `share_purchase_share_lag1` | `lag(daily(share_amt / total_purchase_amt), 1)` | date | Same-day formula is target leakage; only lagged/rolling forms are predictive. |
| `transfer_redeem_share_lag1` | `lag(daily(transfer_amt / total_redeem_amt), 1)` | date | Same-day formula is target leakage; only lagged/rolling forms are predictive. |
| `consume_redeem_share_lag1` | `lag(daily(consume_amt / total_redeem_amt), 1)` | date | Same-day formula is target leakage; only lagged/rolling forms are predictive. |
| `transfer_to_card_share_lag1` | `lag(daily(tftocard_amt / transfer_amt), 1)` | date | Same-day formula is target leakage; only lagged/rolling forms are predictive. |
| `transfer_to_balance_share_lag1` | `lag(daily(tftobal_amt / transfer_amt), 1)` | date | Same-day formula is target leakage; only lagged/rolling forms are predictive. |
| `net_flow_lag1` | `lag(daily(total_purchase_amt - total_redeem_amt), 1)` | date | Same-day formula is target leakage; only lagged/rolling forms are predictive. |

## P1

### demographic_cross

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `sex_x_city_x_constellation` | `sex + city + constellation` | user | Pool cells below 50 training users; create mapping within each fold. |

### rate_rolling

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `mfd_daily_yield_mean7` | `rolling_mean(lag(mfd_daily_yield,1), 7)` | date | Shift before rolling; never center the window. |
| `mfd_daily_yield_mean14` | `rolling_mean(lag(mfd_daily_yield,1), 14)` | date | Shift before rolling; never center the window. |
| `mfd_daily_yield_mean28` | `rolling_mean(lag(mfd_daily_yield,1), 28)` | date | Shift before rolling; never center the window. |
| `mfd_7daily_yield_mean7` | `rolling_mean(lag(mfd_7daily_yield,1), 7)` | date | Shift before rolling; never center the window. |
| `mfd_7daily_yield_mean14` | `rolling_mean(lag(mfd_7daily_yield,1), 14)` | date | Shift before rolling; never center the window. |
| `mfd_7daily_yield_mean28` | `rolling_mean(lag(mfd_7daily_yield,1), 28)` | date | Shift before rolling; never center the window. |
| `Interest_O_N_mean7` | `rolling_mean(lag(Interest_O_N,1), 7)` | date | Shift before rolling; never center the window. |
| `Interest_O_N_mean14` | `rolling_mean(lag(Interest_O_N,1), 14)` | date | Shift before rolling; never center the window. |
| `Interest_O_N_mean28` | `rolling_mean(lag(Interest_O_N,1), 28)` | date | Shift before rolling; never center the window. |
| `Interest_1_W_mean7` | `rolling_mean(lag(Interest_1_W,1), 7)` | date | Shift before rolling; never center the window. |
| `Interest_1_W_mean14` | `rolling_mean(lag(Interest_1_W,1), 14)` | date | Shift before rolling; never center the window. |
| `Interest_1_W_mean28` | `rolling_mean(lag(Interest_1_W,1), 28)` | date | Shift before rolling; never center the window. |
| `Interest_2_W_mean7` | `rolling_mean(lag(Interest_2_W,1), 7)` | date | Shift before rolling; never center the window. |
| `Interest_2_W_mean14` | `rolling_mean(lag(Interest_2_W,1), 14)` | date | Shift before rolling; never center the window. |
| `Interest_2_W_mean28` | `rolling_mean(lag(Interest_2_W,1), 28)` | date | Shift before rolling; never center the window. |
| `Interest_1_M_mean7` | `rolling_mean(lag(Interest_1_M,1), 7)` | date | Shift before rolling; never center the window. |
| `Interest_1_M_mean14` | `rolling_mean(lag(Interest_1_M,1), 14)` | date | Shift before rolling; never center the window. |
| `Interest_1_M_mean28` | `rolling_mean(lag(Interest_1_M,1), 28)` | date | Shift before rolling; never center the window. |
| `Interest_3_M_mean7` | `rolling_mean(lag(Interest_3_M,1), 7)` | date | Shift before rolling; never center the window. |
| `Interest_3_M_mean14` | `rolling_mean(lag(Interest_3_M,1), 14)` | date | Shift before rolling; never center the window. |
| `Interest_3_M_mean28` | `rolling_mean(lag(Interest_3_M,1), 28)` | date | Shift before rolling; never center the window. |
| `Interest_6_M_mean7` | `rolling_mean(lag(Interest_6_M,1), 7)` | date | Shift before rolling; never center the window. |
| `Interest_6_M_mean14` | `rolling_mean(lag(Interest_6_M,1), 14)` | date | Shift before rolling; never center the window. |
| `Interest_6_M_mean28` | `rolling_mean(lag(Interest_6_M,1), 28)` | date | Shift before rolling; never center the window. |
| `Interest_9_M_mean7` | `rolling_mean(lag(Interest_9_M,1), 7)` | date | Shift before rolling; never center the window. |
| `Interest_9_M_mean14` | `rolling_mean(lag(Interest_9_M,1), 14)` | date | Shift before rolling; never center the window. |
| `Interest_9_M_mean28` | `rolling_mean(lag(Interest_9_M,1), 28)` | date | Shift before rolling; never center the window. |
| `Interest_1_Y_mean7` | `rolling_mean(lag(Interest_1_Y,1), 7)` | date | Shift before rolling; never center the window. |
| `Interest_1_Y_mean14` | `rolling_mean(lag(Interest_1_Y,1), 14)` | date | Shift before rolling; never center the window. |
| `Interest_1_Y_mean28` | `rolling_mean(lag(Interest_1_Y,1), 28)` | date | Shift before rolling; never center the window. |

### rate_volatility

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `mfd_daily_yield_std7` | `rolling_std(lag(mfd_daily_yield,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `mfd_daily_yield_std28` | `rolling_std(lag(mfd_daily_yield,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `mfd_7daily_yield_std7` | `rolling_std(lag(mfd_7daily_yield,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `mfd_7daily_yield_std28` | `rolling_std(lag(mfd_7daily_yield,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_O_N_std7` | `rolling_std(lag(Interest_O_N,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_O_N_std28` | `rolling_std(lag(Interest_O_N,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_1_W_std7` | `rolling_std(lag(Interest_1_W,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_1_W_std28` | `rolling_std(lag(Interest_1_W,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_2_W_std7` | `rolling_std(lag(Interest_2_W,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_2_W_std28` | `rolling_std(lag(Interest_2_W,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_1_M_std7` | `rolling_std(lag(Interest_1_M,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_1_M_std28` | `rolling_std(lag(Interest_1_M,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_3_M_std7` | `rolling_std(lag(Interest_3_M,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_3_M_std28` | `rolling_std(lag(Interest_3_M,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_6_M_std7` | `rolling_std(lag(Interest_6_M,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_6_M_std28` | `rolling_std(lag(Interest_6_M,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_9_M_std7` | `rolling_std(lag(Interest_9_M,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_9_M_std28` | `rolling_std(lag(Interest_9_M,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_1_Y_std7` | `rolling_std(lag(Interest_1_Y,1), 7)` | date | Shift before rolling and fit any normalization inside the fold. |
| `Interest_1_Y_std28` | `rolling_std(lag(Interest_1_Y,1), 28)` | date | Shift before rolling and fit any normalization inside the fold. |

### segment_history

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `sex_total_purchase_amt_rolling7` | `rolling_mean(lag(daily_sum(total_purchase_amt) by sex,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `sex_total_redeem_amt_rolling7` | `rolling_mean(lag(daily_sum(total_redeem_amt) by sex,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `sex_direct_purchase_amt_rolling7` | `rolling_mean(lag(daily_sum(direct_purchase_amt) by sex,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `sex_purchase_bank_amt_rolling7` | `rolling_mean(lag(daily_sum(purchase_bank_amt) by sex,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `sex_transfer_amt_rolling7` | `rolling_mean(lag(daily_sum(transfer_amt) by sex,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `city_total_purchase_amt_rolling7` | `rolling_mean(lag(daily_sum(total_purchase_amt) by city,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `city_total_redeem_amt_rolling7` | `rolling_mean(lag(daily_sum(total_redeem_amt) by city,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `city_direct_purchase_amt_rolling7` | `rolling_mean(lag(daily_sum(direct_purchase_amt) by city,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `city_purchase_bank_amt_rolling7` | `rolling_mean(lag(daily_sum(purchase_bank_amt) by city,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `city_transfer_amt_rolling7` | `rolling_mean(lag(daily_sum(transfer_amt) by city,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `constellation_total_purchase_amt_rolling7` | `rolling_mean(lag(daily_sum(total_purchase_amt) by constellation,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `constellation_total_redeem_amt_rolling7` | `rolling_mean(lag(daily_sum(total_redeem_amt) by constellation,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `constellation_direct_purchase_amt_rolling7` | `rolling_mean(lag(daily_sum(direct_purchase_amt) by constellation,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `constellation_purchase_bank_amt_rolling7` | `rolling_mean(lag(daily_sum(purchase_bank_amt) by constellation,1),7)` | date x segment | Aggregate and shift before joining forecast date. |
| `constellation_transfer_amt_rolling7` | `rolling_mean(lag(daily_sum(transfer_amt) by constellation,1),7)` | date x segment | Aggregate and shift before joining forecast date. |

### cross_domain_interaction

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `rate_spread_x_bank_share` | `alipay_vs_shibor_1w_lag1 * bank_purchase_share_lag1` | date | All continuous inputs must be lagged before interaction. |
| `rate_change_x_transfer_share` | `delta7(mfd_7daily_yield)_lag1 * transfer_redeem_share_lag1` | date | All continuous inputs must be lagged before interaction. |
| `weekday_x_rate_spread` | `weekday * alipay_vs_shibor_1w_lag1` | date | All continuous inputs must be lagged before interaction. |
| `new_user_share_x_rate` | `new_user_share_lag1 * mfd_7daily_yield_lag1` | date | All continuous inputs must be lagged before interaction. |

## P2_NOT_NOW

### defer_or_reject

| Feature | Formula | Grain | Leakage rule |
|---|---|---|---|
| `all_pairwise_polynomials` | `every numeric pair and square` | mixed | Do not use until a fold-safe pipeline and incremental-value evidence exist. |
| `high_cardinality_user_id_cross` | `user_id x demographic x rate` | mixed | Do not use until a fold-safe pipeline and incremental-value evidence exist. |
| `full_period_target_encoding` | `category mean target on all dates` | mixed | Do not use until a fold-safe pipeline and incremental-value evidence exist. |
| `same_day_component_ratios` | `ratios built from forecast-day outcomes` | mixed | Do not use until a fold-safe pipeline and incremental-value evidence exist. |
