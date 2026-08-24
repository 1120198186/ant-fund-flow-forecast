# Competition contract

Checked on 2026-08-22.

## Verified task

The official competition is `资金流入流出预测-挑战Baseline`. Using data through 2014-08-31, predict the aggregate purchase and redeem amounts for all users on each day from 2014-09-01 through 2014-09-30.

Official pages:

- [Introduction](https://tianchi.aliyun.com/competition/entrance/231573/introduction)
- [Data and rules](https://tianchi.aliyun.com/competition/entrance/231573/information)
- [Ranking list](https://tianchi.aliyun.com/competition/entrance/231573/rankingList)
- [Current official detail API](https://tianchi.aliyun.com/v3/proxy/competition/api/race/getDetail?raceId=231573)

The current official API reports a long-running season ending at 2027-05-06 23:59:59 and `seasonEnd=false`. The anonymous session was not signed in and returned `showSubmit=false`, so actual submission availability and daily limits must be rechecked after login. The older page text says the race has no time limit; for operations, use the current API date and verify again immediately before submitting.

## Data

The task uses four source tables:

- `user_balance_table.csv`: user-level balance, purchase, redeem, transfer, consumption, and category data from 2013-07-01 through 2014-08-31.
- `user_profile_table.csv`: sex, city, and constellation.
- `mfd_day_share_interest.csv`: daily and seven-day fund yields.
- `mfd_bank_shibor.csv`: SHIBOR tenors.

Amounts are in cents. Official rules prohibit using public data after 2014-08-31 for prediction. The actual supplied CSV headers use lowercase `sex` and `city` even though the official page displays `Sex` and `City`; code must follow the files recorded in the inventory.

## Submission contract

- Exactly 30 rows, one per date from 20140901 through 20140930.
- Three comma-separated columns in order: `report_date,purchase,redeem`.
- No header.
- Dates unique and ascending.
- Purchase and redeem are finite, non-negative integers in cents.

The supplied `comp_predict_table.csv` is only a three-row fragment. Keep it immutable as source evidence, but never use it as the final structural validator. Run `python3 scripts/validate_submission.py <file>` instead.

## Evaluation boundary

Official scoring starts from daily absolute relative error for purchase and redeem. An error of zero earns 10 points for that day, and an error above 0.3 earns zero. Purchase is weighted 45% and redeem 55%.

The intermediate error-to-score mapping and zero-denominator handling are not public. Local validation may report daily relative errors, the count above 0.3, and a 45/55 weighted proxy, but it must not claim to reproduce the exact leaderboard score.

