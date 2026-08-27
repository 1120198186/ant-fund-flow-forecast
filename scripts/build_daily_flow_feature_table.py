#!/usr/bin/env python3
"""Build the canonical daily flow/rate table and rate association evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "d006_daily_flow_feature_table_v1"
LAYER_VERSION = "d003_validated_data_layer_v2"
RATE_FEATURES = [
    "mfd_daily_yield",
    "mfd_7daily_yield",
    "Interest_O_N",
    "Interest_1_W",
    "Interest_2_W",
    "Interest_1_M",
    "Interest_3_M",
    "Interest_6_M",
    "Interest_9_M",
    "Interest_1_Y",
]
FLOW_COLUMNS = [
    "direct_inflow_amt",
    "profit_share_inflow_amt",
    "profit_share_per_10000_opening_balance",
    "total_outflow_amt",
    "transfer_outflow_amt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": left.astype(float), "right": right.astype(float)}).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame["left"].rank().corr(frame["right"].rank()))


def build_daily(balance: pd.DataFrame, fund: pd.DataFrame, shibor: pd.DataFrame) -> pd.DataFrame:
    grouped = balance.groupby("report_date", observed=True)
    daily = grouped.agg(
        observed_users=("user_id", "nunique"),
        opening_balance_amt=("yBalance", "sum"),
        closing_balance_amt=("tBalance", "sum"),
        direct_inflow_amt=("direct_purchase_amt", "sum"),
        balance_purchase_inflow_amt=("purchase_bal_amt", "sum"),
        bankcard_purchase_inflow_amt=("purchase_bank_amt", "sum"),
        profit_share_inflow_amt=("share_amt", "sum"),
        total_inflow_amt=("total_purchase_amt", "sum"),
        total_outflow_amt=("total_redeem_amt", "sum"),
        transfer_outflow_amt=("transfer_amt", "sum"),
        consume_outflow_amt=("consume_amt", "sum"),
        transfer_to_balance_amt=("tftobal_amt", "sum"),
        transfer_to_card_amt=("tftocard_amt", "sum"),
        direct_inflow_users=("direct_purchase_amt", lambda x: int((x > 0).sum())),
        profit_share_users=("share_amt", lambda x: int((x > 0).sum())),
        total_outflow_users=("total_redeem_amt", lambda x: int((x > 0).sum())),
        transfer_outflow_users=("transfer_amt", lambda x: int((x > 0).sum())),
    ).reset_index()
    daily["net_flow_amt"] = daily["total_inflow_amt"] - daily["total_outflow_amt"]
    daily["profit_share_per_10000_opening_balance"] = np.where(
        daily["opening_balance_amt"].eq(0),
        0.0,
        daily["profit_share_inflow_amt"] / daily["opening_balance_amt"] * 10_000,
    )
    daily["direct_inflow_share"] = np.where(
        daily["total_inflow_amt"].eq(0),
        0.0,
        daily["direct_inflow_amt"] / daily["total_inflow_amt"],
    )
    daily["bankcard_direct_inflow_share"] = np.where(
        daily["direct_inflow_amt"].eq(0),
        0.0,
        daily["bankcard_purchase_inflow_amt"] / daily["direct_inflow_amt"],
    )
    daily["transfer_outflow_share"] = np.where(
        daily["total_outflow_amt"].eq(0),
        0.0,
        daily["transfer_outflow_amt"] / daily["total_outflow_amt"],
    )
    daily = daily.merge(
        fund.rename(columns={"mfd_date": "report_date"}),
        on="report_date",
        how="left",
        validate="one_to_one",
    ).merge(
        shibor.rename(columns={"mfd_date": "report_date"}),
        on="report_date",
        how="left",
        validate="one_to_one",
    )
    daily = daily.sort_values("report_date").reset_index(drop=True)
    return daily


def rate_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rate in RATE_FEATURES:
        rate_values = daily[rate].astype(float)
        for flow in FLOW_COLUMNS:
            target = daily[flow].astype(float)
            log_target = np.log1p(target)
            rows.append(
                {
                    "rate_feature": rate,
                    "flow_target": flow,
                    "spearman_level": spearman(rate_values, target),
                    "pearson_log_level": rate_values.corr(log_target),
                    "same_day_diff1_corr": rate_values.diff(1).corr(log_target.diff(1)),
                    "same_day_diff7_corr": rate_values.diff(7).corr(log_target.diff(7)),
                    "lag1_level_spearman": spearman(rate_values.shift(1), target),
                    "lagged_diff1_corr": rate_values.diff(1).shift(1).corr(
                        log_target.diff(1)
                    ),
                    "days": len(daily),
                    "same_day_use": "descriptive_only",
                    "prediction_rule": "use lagged/published values only",
                }
            )
    result = pd.DataFrame(rows)
    result["abs_same_day_diff1"] = result["same_day_diff1_corr"].abs()
    result["abs_lagged_diff1"] = result["lagged_diff1_corr"].abs()
    return result.sort_values("abs_lagged_diff1", ascending=False).reset_index(drop=True)


def validate_daily(daily: pd.DataFrame) -> None:
    expected_dates = pd.date_range(daily["report_date"].min(), daily["report_date"].max())
    if len(daily) != 427 or not pd.DatetimeIndex(daily["report_date"]).equals(expected_dates):
        raise ValueError("daily table is not the expected 427-day continuous calendar")
    checks = [
        daily["direct_inflow_amt"]
        == daily["balance_purchase_inflow_amt"] + daily["bankcard_purchase_inflow_amt"],
        daily["total_inflow_amt"]
        == daily["direct_inflow_amt"] + daily["profit_share_inflow_amt"],
        daily["total_outflow_amt"]
        == daily["transfer_outflow_amt"] + daily["consume_outflow_amt"],
        daily["transfer_outflow_amt"]
        == daily["transfer_to_balance_amt"] + daily["transfer_to_card_amt"],
    ]
    if not all(check.all() for check in checks):
        raise ValueError("daily flow identities do not reconcile")
    if daily[RATE_FEATURES].isna().any().any():
        raise ValueError("daily rate columns contain missing values")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    date_index = table.schema.get_field_index("report_date")
    table = table.set_column(
        date_index, "report_date", table["report_date"].cast(pa.date32())
    )
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    layer = root / "data" / "derived" / LAYER_VERSION
    output = root / "data" / "derived" / VERSION
    if output.exists():
        print(f"Refusing to overwrite existing table version: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)

    balance_columns = [
        "user_id",
        "report_date",
        "tBalance",
        "yBalance",
        "total_purchase_amt",
        "direct_purchase_amt",
        "purchase_bal_amt",
        "purchase_bank_amt",
        "share_amt",
        "total_redeem_amt",
        "consume_amt",
        "transfer_amt",
        "tftobal_amt",
        "tftocard_amt",
    ]
    balance = pd.read_parquet(layer / "user_balance_daily.parquet", columns=balance_columns)
    balance["report_date"] = pd.to_datetime(balance["report_date"])
    fund = pd.read_parquet(layer / "fund_yield_daily.parquet")
    shibor = pd.read_parquet(layer / "shibor_daily.parquet")
    fund["mfd_date"] = pd.to_datetime(fund["mfd_date"])
    shibor["mfd_date"] = pd.to_datetime(shibor["mfd_date"])

    daily = build_daily(balance, fund, shibor)
    validate_daily(daily)
    correlations = rate_correlations(daily)

    csv_path = output / "daily_flow_rate_features.csv"
    parquet_path = output / "daily_flow_rate_features.parquet"
    correlation_path = output / "rate_flow_correlations.csv"
    daily.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_parquet(daily, parquet_path)
    correlations.to_csv(correlation_path, index=False, encoding="utf-8-sig")

    readme = """# Daily flow and rate feature table

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
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    file_rows = []
    for path in [csv_path, parquet_path, correlation_path, output / "README.md"]:
        file_rows.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest = {
        "dataset_version": VERSION,
        "generator": "scripts/build_daily_flow_feature_table.py",
        "command": ".venv\\Scripts\\python.exe scripts\\build_daily_flow_feature_table.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "source": f"data/derived/{LAYER_VERSION}/manifest.json",
        "rows": len(daily),
        "date_range": [
            daily["report_date"].min().date().isoformat(),
            daily["report_date"].max().date().isoformat(),
        ],
        "core_fields": {
            "direct_inflow_amt": "sum(direct_purchase_amt)",
            "profit_share_inflow_amt": "sum(share_amt)",
            "total_outflow_amt": "sum(total_redeem_amt)",
            "transfer_outflow_amt": "sum(transfer_amt)",
        },
        "files": file_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in [csv_path, parquet_path, correlation_path]:
        path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
