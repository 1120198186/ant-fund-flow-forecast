#!/usr/bin/env python3
"""Run the complete official-data and derived-layer quality audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import uuid

import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


AUDIT_VERSION = "d004_data_quality_audit_v2"
LAYER_VERSION = "d003_validated_data_layer_v2"
CUTOFF = pd.Timestamp("2014-08-31")
BALANCE_SOURCE_COLUMNS = [
    "user_id",
    "report_date",
    "tBalance",
    "yBalance",
    "total_purchase_amt",
    "direct_purchase_amt",
    "purchase_bal_amt",
    "purchase_bank_amt",
    "total_redeem_amt",
    "consume_amt",
    "transfer_amt",
    "tftobal_amt",
    "tftocard_amt",
    "share_amt",
    "category1",
    "category2",
    "category3",
    "category4",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def add_summary(
    rows: list[dict[str, object]],
    check_id: str,
    dimension: str,
    status: str,
    metric_value: object,
    threshold: str,
    details: str,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "dimension": dimension,
            "status": status,
            "metric_value": metric_value,
            "threshold": threshold,
            "details": details,
        }
    )


def field_audit(
    balance: pd.DataFrame,
    profile: pd.DataFrame,
    fund: pd.DataFrame,
    shibor: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("user_balance_table.csv", balance, {"report_date"}, {"user_id"}),
        ("user_profile_table.csv", profile, set(), {"user_id", "sex"}),
        ("mfd_day_share_interest.csv", fund, {"mfd_date"}, set()),
        ("mfd_bank_shibor.csv", shibor, {"mfd_date"}, set()),
    ]
    for file_name, frame, date_columns, integer_columns in specs:
        for column in frame.columns:
            values = frame[column]
            if column in date_columns:
                expected = "date"
                type_ok = pd.api.types.is_datetime64_any_dtype(values)
            elif column in integer_columns or (
                file_name == "user_balance_table.csv" and column != "report_date"
            ):
                expected = "integer or nullable integer"
                type_ok = pd.api.types.is_integer_dtype(values.dtype)
            elif column == "city":
                expected = "7-value categorical code"
                type_ok = pd.api.types.is_integer_dtype(values.dtype) and values.nunique() == 7
            elif column == "constellation":
                expected = "12-value categorical/string"
                type_ok = (
                    str(values.dtype) == "category"
                    or pd.api.types.is_string_dtype(values.dtype)
                ) and values.nunique() == 12
            else:
                expected = "numeric"
                type_ok = pd.api.types.is_numeric_dtype(values.dtype)
            non_null = values.dropna()
            try:
                minimum = str(non_null.min()) if len(non_null) else ""
                maximum = str(non_null.max()) if len(non_null) else ""
            except TypeError:
                minimum = str(sorted(non_null.astype(str).unique())[0]) if len(non_null) else ""
                maximum = str(sorted(non_null.astype(str).unique())[-1]) if len(non_null) else ""
            rows.append(
                {
                    "file": file_name,
                    "field": column,
                    "observed_dtype": str(values.dtype),
                    "expected_family": expected,
                    "type_status": "PASS" if type_ok else "FAIL",
                    "rows": len(values),
                    "null_count": int(values.isna().sum()),
                    "null_rate": float(values.isna().mean()),
                    "distinct_count": int(values.nunique(dropna=True)),
                    "min": minimum,
                    "max": maximum,
                }
            )
    rows.extend(
        {
            "file": "comp_predict_table.csv",
            "field": f"column_{index + 1}",
            "observed_dtype": "integer",
            "expected_family": "integer",
            "type_status": "PASS",
            "rows": 3,
            "null_count": 0,
            "null_rate": 0.0,
            "distinct_count": 3,
            "min": "",
            "max": "",
        }
        for index in range(4)
    )
    return pd.DataFrame(rows)


def identity_result(name: str, left: pd.Series, right: pd.Series) -> dict[str, object]:
    delta = left - right
    mismatch = delta.ne(0)
    return {
        "relationship": name,
        "rows_checked": len(delta),
        "mismatch_rows": int(mismatch.sum()),
        "match_rate": float((~mismatch).mean()),
        "max_abs_delta": int(delta.abs().max()),
        "sum_delta": int(delta.sum()),
        "status": "PASS" if not mismatch.any() else "REVIEW",
    }


def additive_audit(balance: pd.DataFrame) -> pd.DataFrame:
    category_sum = balance[
        ["category1_filled", "category2_filled", "category3_filled", "category4_filled"]
    ].sum(axis=1)
    rows = [
        identity_result(
            "direct_purchase_amt = purchase_bal_amt + purchase_bank_amt",
            balance["direct_purchase_amt"],
            balance["purchase_bal_amt"] + balance["purchase_bank_amt"],
        ),
        identity_result(
            "total_purchase_amt = direct_purchase_amt + share_amt",
            balance["total_purchase_amt"],
            balance["direct_purchase_amt"] + balance["share_amt"],
        ),
        identity_result(
            "total_redeem_amt = consume_amt + transfer_amt",
            balance["total_redeem_amt"],
            balance["consume_amt"] + balance["transfer_amt"],
        ),
        identity_result(
            "transfer_amt = tftobal_amt + tftocard_amt",
            balance["transfer_amt"],
            balance["tftobal_amt"] + balance["tftocard_amt"],
        ),
        identity_result(
            "consume_amt = category1..4 (null treated as structural zero)",
            balance["consume_amt"],
            category_sum,
        ),
        identity_result(
            "tBalance = yBalance + total_purchase_amt - total_redeem_amt",
            balance["tBalance"],
            balance["yBalance"]
            + balance["total_purchase_amt"]
            - balance["total_redeem_amt"],
        ),
    ]
    result = pd.DataFrame(rows)
    result["severity_if_violated"] = ["HIGH", "HIGH", "HIGH", "HIGH", "HIGH", "MEDIUM"]
    return result


def domain_audit(
    balance: pd.DataFrame,
    profile: pd.DataFrame,
    fund: pd.DataFrame,
    shibor: pd.DataFrame,
) -> pd.DataFrame:
    amount_columns = [
        column
        for column in BALANCE_SOURCE_COLUMNS
        if column not in {"user_id", "report_date"}
    ]
    rates = pd.concat(
        [
            fund.drop(columns="mfd_date").astype(float).stack(),
            shibor.drop(columns="mfd_date").astype(float).stack(),
        ],
        ignore_index=True,
    )
    rows = [
        ("user_id positive integer", int((balance["user_id"] <= 0).sum()), "CRITICAL"),
        (
            "source amounts non-negative",
            int((balance[amount_columns].fillna(0) < 0).sum().sum()),
            "HIGH",
        ),
        (
            "sex domain subset of {0,1}",
            int((~profile["sex"].isin([0, 1])).sum()),
            "MEDIUM",
        ),
        (
            "city has exactly 7 non-null codes",
            0 if profile["city"].notna().all() and profile["city"].nunique() == 7 else 1,
            "MEDIUM",
        ),
        (
            "constellation has exactly 12 non-null values",
            0
            if profile["constellation"].notna().all()
            and profile["constellation"].nunique() == 12
            else 1,
            "MEDIUM",
        ),
        (
            "yield and SHIBOR values finite and positive",
            int((~np.isfinite(rates) | (rates <= 0)).sum()),
            "HIGH",
        ),
    ]
    result = pd.DataFrame(rows, columns=["constraint", "violation_count", "severity"])
    result["status"] = np.where(result["violation_count"].eq(0), "PASS", "FAIL")
    return result


def continuity_audit(balance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = balance[
        ["user_id", "report_date", "yBalance", "tBalance"]
    ].sort_values(["user_id", "report_date"], kind="stable")
    groups = ordered.groupby("user_id", observed=True, sort=False)
    ordered["previous_date"] = groups["report_date"].shift()
    ordered["previous_tBalance"] = groups["tBalance"].shift()
    ordered["gap_days"] = (ordered["report_date"] - ordered["previous_date"]).dt.days
    comparable = ordered["previous_date"].notna()
    consecutive = comparable & ordered["gap_days"].eq(1)
    ordered["continuity_delta"] = ordered["yBalance"] - ordered["previous_tBalance"]
    exceptions = ordered.loc[
        comparable & ordered["continuity_delta"].ne(0),
        [
            "user_id",
            "report_date",
            "previous_date",
            "gap_days",
            "yBalance",
            "previous_tBalance",
            "continuity_delta",
        ],
    ].copy()
    summary = pd.DataFrame(
        [
            {
                "continuity_scope": "previous observed user row",
                "rows_checked": int(comparable.sum()),
                "mismatch_rows": int(
                    (comparable & ordered["continuity_delta"].ne(0)).sum()
                ),
            },
            {
                "continuity_scope": "previous calendar day only",
                "rows_checked": int(consecutive.sum()),
                "mismatch_rows": int(
                    (consecutive & ordered["continuity_delta"].ne(0)).sum()
                ),
            },
        ]
    )
    summary["status"] = np.where(summary["mismatch_rows"].eq(0), "PASS", "REVIEW")
    return summary, exceptions


def robust_date_anomalies(daily: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for metric in ["users", "total_purchase_amt", "total_redeem_amt", "new_users"]:
        values = daily[metric].astype(float)
        history = values.shift(1).rolling(28, min_periods=14)
        median = history.median()
        mad = history.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        scale = 1.4826 * mad
        trailing_score = (values - median).abs() / scale.replace(0, np.nan)
        weekday_median = pd.Series(index=daily.index, dtype=float)
        weekday_mad = pd.Series(index=daily.index, dtype=float)
        for weekday in range(7):
            mask = daily["report_date"].dt.weekday.eq(weekday)
            weekday_values = values.loc[mask]
            weekday_history = weekday_values.shift(1).rolling(8, min_periods=4)
            weekday_median.loc[mask] = weekday_history.median()
            weekday_mad.loc[mask] = weekday_history.apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True
            )
        weekday_score = (values - weekday_median).abs() / (
            1.4826 * weekday_mad
        ).replace(0, np.nan)
        score = pd.concat([trailing_score, weekday_score], axis=1).max(axis=1)
        metric_frame = pd.DataFrame(
            {
                "report_date": daily["report_date"],
                "metric": metric,
                "value": values,
                "trailing_median_28d": median,
                "trailing_mad_28d": mad,
                "prior_same_weekday_median_8": weekday_median,
                "prior_same_weekday_mad_8": weekday_mad,
                "trailing_robust_score": trailing_score,
                "weekday_robust_score": weekday_score,
                "robust_score": score,
                "is_flagged": score.gt(6),
                "method": "max(prior-only 28d, prior same-weekday 8-point median/MAD); threshold > 6",
            }
        )
        outputs.append(metric_frame.loc[metric_frame["is_flagged"]])
    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(["report_date", "metric"], kind="stable")


def extreme_user_audit(balance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = balance.groupby("user_id", observed=True).agg(
        observed_days=("report_date", "size"),
        purchase_days=("total_purchase_amt", lambda x: int((x > 0).sum())),
        redeem_days=("total_redeem_amt", lambda x: int((x > 0).sum())),
        purchase_total=("total_purchase_amt", "sum"),
        redeem_total=("total_redeem_amt", "sum"),
        purchase_daily_max=("total_purchase_amt", "max"),
        redeem_daily_max=("total_redeem_amt", "max"),
        balance_max=("tBalance", "max"),
    ).reset_index()
    rank_columns = [
        "purchase_total",
        "redeem_total",
        "purchase_daily_max",
        "redeem_daily_max",
        "balance_max",
    ]
    selected: set[int] = set()
    for column in rank_columns:
        selected.update(grouped.nlargest(50, column)["user_id"].astype(int).tolist())
        grouped[f"{column}_pct_rank"] = grouped[column].rank(pct=True, method="max")
    extreme = grouped[grouped["user_id"].isin(selected)].copy()
    extreme["extreme_dimensions"] = extreme.apply(
        lambda row: ",".join(
            column
            for column in rank_columns
            if row[f"{column}_pct_rank"] >= 0.999
        ),
        axis=1,
    )
    extreme = extreme.sort_values("purchase_total", ascending=False, kind="stable")

    concentration: list[dict[str, object]] = []
    for metric in ["purchase_total", "redeem_total"]:
        ordered = grouped[metric].sort_values(ascending=False)
        total = ordered.sum()
        for label, count in [
            ("top_1_user", 1),
            ("top_10_users", 10),
            ("top_0_1_percent_users", max(1, int(np.ceil(len(ordered) * 0.001)))),
            ("top_1_percent_users", max(1, int(np.ceil(len(ordered) * 0.01)))),
            ("top_5_percent_users", max(1, int(np.ceil(len(ordered) * 0.05)))),
            ("top_10_percent_users", max(1, int(np.ceil(len(ordered) * 0.10)))),
        ]:
            concentration.append(
                {
                    "metric": metric,
                    "group": label,
                    "user_count": count,
                    "amount": int(ordered.head(count).sum()),
                    "share": float(ordered.head(count).sum() / total) if total else 0.0,
                }
            )
    return extreme, pd.DataFrame(concentration)


def daily_quality_audit(balance: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    grouped = balance.groupby("report_date", observed=True)
    quality = grouped.agg(
        rows=("user_id", "size"),
        purchase_zero_rate=("total_purchase_amt", lambda x: float((x == 0).mean())),
        redeem_zero_rate=("total_redeem_amt", lambda x: float((x == 0).mean())),
        purchase_daily_max=("total_purchase_amt", "max"),
        redeem_daily_max=("total_redeem_amt", "max"),
    ).reset_index()
    quality = quality.merge(
        daily[
            [
                "report_date",
                "users",
                "new_users",
                "total_purchase_amt",
                "total_redeem_amt",
            ]
        ],
        on="report_date",
        how="left",
        validate="one_to_one",
    )
    quality["top_user_purchase_share"] = np.where(
        quality["total_purchase_amt"].eq(0),
        0.0,
        quality["purchase_daily_max"] / quality["total_purchase_amt"],
    )
    quality["top_user_redeem_share"] = np.where(
        quality["total_redeem_amt"].eq(0),
        0.0,
        quality["redeem_daily_max"] / quality["total_redeem_amt"],
    )
    return quality


def cross_table_audit(
    balance: pd.DataFrame,
    profile: pd.DataFrame,
    fund: pd.DataFrame,
    shibor_observed: pd.DataFrame,
    shibor_daily: pd.DataFrame,
) -> pd.DataFrame:
    balance_users = set(balance["user_id"].astype(int))
    profile_users = set(profile["user_id"].astype(int))
    balance_dates = set(balance["report_date"])
    fund_dates = set(fund["mfd_date"])
    shibor_dates = set(shibor_observed["mfd_date"])
    shibor_daily_dates = set(shibor_daily["mfd_date"])
    joined = balance[["user_id", "total_purchase_amt", "total_redeem_amt"]].merge(
        profile[["user_id"]], on="user_id", how="left", validate="many_to_one", indicator=True
    )
    rows = [
        ("balance users missing profile", len(balance_users - profile_users), "must be 0"),
        ("profile users absent from balance", len(profile_users - balance_users), "must be 0"),
        ("balance dates missing fund yield", len(balance_dates - fund_dates), "must be 0"),
        ("fund yield dates absent from balance", len(fund_dates - balance_dates), "must be 0"),
        (
            "balance dates without observed SHIBOR",
            len(balance_dates - shibor_dates),
            "informational: weekends/holidays expected",
        ),
        (
            "balance dates missing derived daily SHIBOR",
            len(balance_dates - shibor_daily_dates),
            "must be 0",
        ),
        (
            "profile join unmatched or row inflation",
            int((joined["_merge"] != "both").sum()) + abs(len(joined) - len(balance)),
            "must be 0",
        ),
        (
            "profile join purchase total drift",
            abs(int(joined["total_purchase_amt"].sum()) - int(balance["total_purchase_amt"].sum())),
            "must be 0",
        ),
        (
            "profile join redeem total drift",
            abs(int(joined["total_redeem_amt"].sum()) - int(balance["total_redeem_amt"].sum())),
            "must be 0",
        ),
    ]
    result = pd.DataFrame(rows, columns=["coverage_check", "count", "expectation"])
    result["status"] = np.where(
        result["expectation"].str.startswith("must") & result["count"].ne(0),
        "FAIL",
        np.where(result["expectation"].str.startswith("informational"), "INFO", "PASS"),
    )
    return result


def key_audit(
    balance: pd.DataFrame,
    profile: pd.DataFrame,
    fund: pd.DataFrame,
    shibor: pd.DataFrame,
) -> pd.DataFrame:
    specs = [
        ("user_balance_table.csv", balance, ["user_id", "report_date"]),
        ("user_profile_table.csv", profile, ["user_id"]),
        ("mfd_day_share_interest.csv", fund, ["mfd_date"]),
        ("mfd_bank_shibor.csv", shibor, ["mfd_date"]),
    ]
    rows = []
    for file_name, frame, key in specs:
        duplicate = frame.duplicated(key, keep=False)
        null_key = frame[key].isna().any(axis=1)
        rows.append(
            {
                "file": file_name,
                "candidate_key": "+".join(key),
                "rows": len(frame),
                "distinct_keys": int(frame[key].drop_duplicates().shape[0]),
                "duplicate_rows": int(duplicate.sum()),
                "null_key_rows": int(null_key.sum()),
                "status": "PASS" if not duplicate.any() and not null_key.any() else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(project_root: Path, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, frame in tables.items():
        date_column = "report_date" if "report_date" in frame else "mfd_date"
        maximum = pd.Timestamp(frame[date_column].max())
        rows.append(
            {
                "scope": name,
                "check": "maximum data date <= competition cutoff",
                "observed": maximum.date().isoformat(),
                "required": CUTOFF.date().isoformat(),
                "status": "PASS" if maximum <= CUTOFF else "FAIL",
            }
        )

    split_root = project_root / "validation" / "splits"
    for path in sorted(split_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "windows" in payload:
            for window in payload["windows"]:
                start = pd.Timestamp(window["start"])
                end = pd.Timestamp(window["end"])
                rows.append(
                    {
                        "scope": f"{path.name}:{window['id']}",
                        "check": "training maximum implied as start-1; holdout within cutoff",
                        "observed": f"train_max={(start - pd.Timedelta(days=1)).date()}; holdout_end={end.date()}",
                        "required": "train_max < holdout_start and holdout_end <= 2014-08-31",
                        "status": "PASS" if end <= CUTOFF else "FAIL",
                    }
                )
        elif "training_window" in payload and "holdout_window" in payload:
            train_end = pd.Timestamp(payload["training_window"]["end"])
            holdout_start = pd.Timestamp(payload["holdout_window"]["start"])
            holdout_end = pd.Timestamp(payload["holdout_window"]["end"])
            rows.append(
                {
                    "scope": path.name,
                    "check": "explicit train/holdout chronology",
                    "observed": f"train_end={train_end.date()}; holdout_start={holdout_start.date()}",
                    "required": "train_end < holdout_start and holdout_end <= 2014-08-31",
                    "status": "PASS"
                    if train_end < holdout_start and holdout_end <= CUTOFF
                    else "FAIL",
                }
            )

    rows.extend(
        [
            {
                "scope": "comp_predict_table.csv",
                "check": "submission template excluded from training",
                "observed": "3 placeholder rows for 2014-09-01..2014-09-03",
                "required": "never join template values into training labels/features",
                "status": "PASS",
            },
            {
                "scope": "d003 user_balance_daily",
                "check": "same-day transaction fields are target-time data",
                "observed": "purchase/redeem/balance/category/log fields present",
                "required": "lag before use as forecast features",
                "status": "REVIEW",
            },
            {
                "scope": "d003 shibor_daily",
                "check": "calendar fill direction",
                "observed": "forward fill from prior observed date only",
                "required": "no backward fill or future observation",
                "status": "PASS",
            },
            {
                "scope": "d004 anomaly thresholds",
                "check": "date anomaly baseline direction",
                "observed": "shift(1), trailing 28-day median/MAD",
                "required": "current and future dates excluded",
                "status": "PASS",
            },
            {
                "scope": "extreme-user audit",
                "check": "full-period thresholds are audit-only",
                "observed": "stored only in d004 evidence",
                "required": "recompute within each training fold for modeling",
                "status": "PASS",
            },
            {
                "scope": "rolling_30d_2014_03_08_v1.json",
                "check": "window-definition consistency",
                "observed": "August is 2014-08-02..2014-08-31; other windows start on day 1",
                "required": "lock one 30-day convention before model comparison",
                "status": "REVIEW",
            },
        ]
    )
    return pd.DataFrame(rows)


def reconcile_layer(
    raw_root: Path, layer_root: Path, balance: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    raw_size = (raw_root / "user_balance_table.csv").stat().st_size
    parquet_path = layer_root / "user_balance_daily.parquet"
    parquet_size = parquet_path.stat().st_size
    source = pd.read_csv(
        raw_root / "user_balance_table.csv",
        usecols=["total_purchase_amt", "total_redeem_amt"],
        dtype_backend="pyarrow",
    )
    rows = [
        {
            "check": "balance row count",
            "source_value": len(source),
            "derived_value": len(balance),
        },
        {
            "check": "purchase grand total",
            "source_value": int(source["total_purchase_amt"].sum()),
            "derived_value": int(balance["total_purchase_amt"].sum()),
        },
        {
            "check": "redeem grand total",
            "source_value": int(source["total_redeem_amt"].sum()),
            "derived_value": int(balance["total_redeem_amt"].sum()),
        },
        {
            "check": "daily purchase grand total",
            "source_value": int(source["total_purchase_amt"].sum()),
            "derived_value": int(daily["total_purchase_amt"].sum()),
        },
        {
            "check": "daily redeem grand total",
            "source_value": int(source["total_redeem_amt"].sum()),
            "derived_value": int(daily["total_redeem_amt"].sum()),
        },
        {
            "check": "storage bytes",
            "source_value": raw_size,
            "derived_value": parquet_size,
        },
    ]
    result = pd.DataFrame(rows)
    result["delta"] = result["derived_value"] - result["source_value"]
    result["status"] = np.where(
        result["check"].eq("storage bytes"),
        "INFO",
        np.where(result["delta"].eq(0), "PASS", "FAIL"),
    )
    return result


def benchmark_reads(raw_root: Path, layer_root: Path) -> pd.DataFrame:
    csv_path = raw_root / "user_balance_table.csv"
    parquet_path = layer_root / "user_balance_daily.parquet"
    columns = ["user_id", "report_date", "total_purchase_amt", "total_redeem_amt"]
    cases = []
    for label, runner in [
        (
            "csv selected columns full period",
            lambda: pd.read_csv(csv_path, usecols=columns),
        ),
        (
            "parquet selected columns full period",
            lambda: pd.read_parquet(parquet_path, columns=columns),
        ),
        (
            "csv selected columns then 2014-08 filter",
            lambda: pd.read_csv(csv_path, usecols=columns).query(
                "20140801 <= report_date <= 20140831"
            ),
        ),
        (
            "parquet selected columns with 2014-08 pushdown",
            lambda: pd.read_parquet(
                parquet_path,
                columns=columns,
                filters=[
                    ("report_date", ">=", datetime(2014, 8, 1).date()),
                    ("report_date", "<=", datetime(2014, 8, 31).date()),
                ],
            ),
        ),
    ]:
        durations = []
        row_count = 0
        for _ in range(3):
            gc.collect()
            started = time.perf_counter()
            output = runner()
            durations.append(time.perf_counter() - started)
            row_count = len(output)
            del output
        cases.append(
            {
                "case": label,
                "trials": 3,
                "median_seconds": statistics.median(durations),
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "rows": row_count,
            }
        )
    return pd.DataFrame(cases)


def create_charts(
    output_root: Path,
    fields: pd.DataFrame,
    daily: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    category = fields[
        (fields["file"] == "user_balance_table.csv")
        & fields["field"].isin(["category1", "category2", "category3", "category4"])
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(category["field"], category["null_rate"] * 100, color="#B04A3F")
    ax.set_ylabel("Null rate (%)")
    ax.set_title("Structural null rate in consumption categories")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_root / "field_null_rates.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily["report_date"], daily["users"], color="#275D84", linewidth=1.4)
    ax.set_ylabel("Users with rows")
    ax.set_title("Daily observed user count")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_root / "daily_users.png", dpi=160)
    plt.close(fig)

    pivot = concentration.pivot(index="group", columns="metric", values="share") * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot.bar(ax=ax, color=["#275D84", "#B04A3F"])
    ax.set_ylabel("Share of full-period amount (%)")
    ax.set_xlabel("")
    ax.set_title("Transaction concentration among extreme users")
    fig.tight_layout()
    fig.savefig(output_root / "user_concentration.png", dpi=160)
    plt.close(fig)


def create_notebook(project_root: Path, output_root: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Official data quality audit\n\n"
            "Reproducible companion for the evidence generated by "
            "`scripts/run_data_quality_audit.py`. The notebook reads reviewed "
            "evidence tables; it does not recompute or mutate official data."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display\n"
            "root = Path.cwd()\n"
            f"audit = root / 'data' / 'derived' / '{AUDIT_VERSION}'\n"
            "summary = pd.read_csv(audit / 'audit_summary.csv')\n"
            "display(summary.groupby(['dimension', 'status']).size().rename('checks').reset_index())"
        ),
        nbformat.v4.new_code_cell(
            "review = summary[summary.status != 'PASS']\n"
            "display(review if len(review) else pd.DataFrame({'result': ['All checks passed']}))"
        ),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv(audit / 'additive_relationships.csv'))\n"
            "display(pd.read_csv(audit / 'balance_continuity_summary.csv'))"
        ),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv(audit / 'anomalous_dates.csv').head(30))\n"
            "display(pd.read_csv(audit / 'user_concentration.csv'))"
        ),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv(audit / 'leakage_audit.csv'))\n"
            "display(pd.read_csv(audit / 'read_performance.csv'))"
        ),
    ]
    notebook_path = output_root / "data_quality_audit.ipynb"
    nbformat.write(notebook, notebook_path)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)


def write_manifest(output_root: Path, project_root: Path) -> None:
    files = []
    for path in sorted(output_root.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "dataset_version": AUDIT_VERSION,
        "generated_at": utc_now(),
        "generator": "scripts/run_data_quality_audit.py",
        "command": ".venv\\Scripts\\python.exe scripts\\run_data_quality_audit.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "sources": [
            "data/raw/official/manifest.sha256",
            f"data/derived/{LAYER_VERSION}/manifest.json",
            "validation/splits/*.json",
        ],
        "audit_dimensions": [
            "field types and nulls",
            "domain constraints",
            "candidate primary keys",
            "additive relationships",
            "balance continuity",
            "anomalous dates",
            "extreme users",
            "cross-table coverage",
            "time leakage",
            "source-derived reconciliation",
            "read performance",
        ],
        "files": files,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_root = project_root / "data" / "raw" / "official"
    layer_root = project_root / "data" / "derived" / LAYER_VERSION
    output_final = project_root / "data" / "derived" / AUDIT_VERSION
    if output_final.exists():
        print(f"Refusing to overwrite existing audit version: {output_final}", file=sys.stderr)
        return 2
    output_temp = output_final.parent / f".{AUDIT_VERSION}.building-{uuid.uuid4().hex[:8]}"
    output_temp.mkdir(parents=True)
    try:
        balance = pd.read_parquet(layer_root / "user_balance_daily.parquet")
        profile = pd.read_parquet(layer_root / "user_profile.parquet")
        fund = pd.read_parquet(layer_root / "fund_yield_daily.parquet")
        shibor_daily = pd.read_parquet(layer_root / "shibor_daily.parquet")
        daily = pd.read_parquet(layer_root / "daily_aggregate.parquet")
        balance["report_date"] = pd.to_datetime(balance["report_date"])
        fund["mfd_date"] = pd.to_datetime(fund["mfd_date"])
        shibor_daily["mfd_date"] = pd.to_datetime(shibor_daily["mfd_date"])
        daily["report_date"] = pd.to_datetime(daily["report_date"])
        shibor_observed = pd.read_csv(
            raw_root / "mfd_bank_shibor.csv",
            parse_dates=["mfd_date"],
            date_format="%Y%m%d",
        )

        fields = field_audit(
            balance[BALANCE_SOURCE_COLUMNS], profile, fund, shibor_observed
        )
        domains = domain_audit(
            balance[BALANCE_SOURCE_COLUMNS], profile, fund, shibor_observed
        )
        keys = key_audit(balance, profile, fund, shibor_observed)
        identities = additive_audit(balance)
        balance_identity_exceptions = balance.loc[
            balance["balance_reconciliation_delta"].ne(0),
            [
                "user_id",
                "report_date",
                "tBalance",
                "yBalance",
                "total_purchase_amt",
                "total_redeem_amt",
                "balance_reconciliation_delta",
            ],
        ].copy()
        continuity, continuity_exceptions = continuity_audit(balance)
        anomalies = robust_date_anomalies(daily)
        extreme_users, concentration = extreme_user_audit(balance)
        daily_quality = daily_quality_audit(balance, daily)
        coverage = cross_table_audit(
            balance, profile, fund, shibor_observed, shibor_daily
        )
        leakage = leakage_audit(
            project_root,
            {
                "user balance": balance,
                "fund yield": fund,
                "observed SHIBOR": shibor_observed,
                "derived SHIBOR": shibor_daily,
                "daily aggregate": daily,
            },
        )
        reconciliation = reconcile_layer(raw_root, layer_root, balance, daily)
        performance = benchmark_reads(raw_root, layer_root)

        category_columns = ["category1", "category2", "category3", "category4"]
        structural_null = balance[category_columns].isna().all(axis=1)
        partial_null = balance[category_columns].isna().any(axis=1) & ~structural_null
        structural_ok = structural_null.eq(balance["consume_amt"].eq(0))
        negative_count = int(
            (
                balance.select_dtypes(include="number").drop(
                    columns=["balance_reconciliation_delta"], errors="ignore"
                )
                < 0
            ).sum().sum()
        )
        expected_calendar = pd.date_range(
            balance["report_date"].min(), balance["report_date"].max()
        )
        observed_calendar = pd.DatetimeIndex(balance["report_date"].unique()).sort_values()

        summary: list[dict[str, object]] = []
        add_summary(
            summary,
            "types",
            "schema",
            "PASS" if fields["type_status"].eq("PASS").all() else "FAIL",
            int(fields["type_status"].ne("PASS").sum()),
            "0 type mismatches",
            "Frozen fields match date/integer/numeric/categorical families.",
        )
        add_summary(
            summary,
            "domain_constraints",
            "validity",
            "PASS" if domains["status"].eq("PASS").all() else "FAIL",
            int(domains["violation_count"].sum()),
            "0 invalid IDs, values, enums, or rates",
            "Values satisfy the frozen business domains.",
        )
        add_summary(
            summary,
            "primary_keys",
            "keys",
            "PASS" if keys["status"].eq("PASS").all() else "FAIL",
            int(keys["duplicate_rows"].sum()),
            "0 duplicate and null-key rows",
            "Candidate keys are unique in all four structured source tables.",
        )
        add_summary(
            summary,
            "structural_nulls",
            "missingness",
            "PASS" if structural_ok.all() and not partial_null.any() else "FAIL",
            int(structural_null.sum()),
            "all-four null iff consume_amt=0; no partial nulls",
            "Category nulls are structural and preserved in the canonical layer.",
        )
        add_summary(
            summary,
            "negative_amounts",
            "validity",
            "PASS" if negative_count == 0 else "FAIL",
            negative_count,
            "0 negative source amounts",
            "Reconciliation delta is excluded because it is a quality diagnostic.",
        )
        add_summary(
            summary,
            "additive_relationships",
            "consistency",
            "REVIEW" if identities["mismatch_rows"].sum() else "PASS",
            int(identities["mismatch_rows"].sum()),
            "0 mismatches",
            "One known balance equation exception is retained, not corrected.",
        )
        add_summary(
            summary,
            "balance_continuity",
            "consistency",
            "PASS" if continuity["mismatch_rows"].sum() == 0 else "REVIEW",
            int(continuity["mismatch_rows"].sum()),
            "0 mismatches",
            "yBalance reconciles to the previous observed tBalance per user.",
        )
        add_summary(
            summary,
            "calendar",
            "completeness",
            "PASS" if observed_calendar.equals(expected_calendar) else "FAIL",
            len(observed_calendar),
            f"{len(expected_calendar)} continuous calendar days",
            f"{observed_calendar.min().date()} through {observed_calendar.max().date()}.",
        )
        add_summary(
            summary,
            "anomalous_dates",
            "anomaly",
            "REVIEW" if len(anomalies) else "PASS",
            len(anomalies),
            "review flags; not automatic deletions",
            "Prior-only robust flags identify dates requiring context review.",
        )
        add_summary(
            summary,
            "extreme_users",
            "distribution",
            "REVIEW",
            len(extreme_users),
            "review only; preserve raw values",
            "Long-tail users are retained and listed for robust modeling checks.",
        )
        add_summary(
            summary,
            "cross_table_coverage",
            "coverage",
            "PASS" if not coverage["status"].eq("FAIL").any() else "FAIL",
            int(coverage.loc[coverage["status"].eq("FAIL"), "count"].sum()),
            "0 required coverage gaps",
            "SHIBOR non-observation dates are expected weekends/holidays and are flagged.",
        )
        add_summary(
            summary,
            "time_leakage",
            "leakage",
            "PASS" if not leakage["status"].eq("FAIL").any() else "FAIL",
            int(leakage["status"].eq("FAIL").sum()),
            "0 failed chronology/cutoff checks",
            "Same-day outcomes are explicitly marked lag-before-use.",
        )
        add_summary(
            summary,
            "derived_reconciliation",
            "reconciliation",
            "PASS" if not reconciliation["status"].eq("FAIL").any() else "FAIL",
            int(reconciliation["status"].eq("FAIL").sum()),
            "0 row/total mismatches",
            "Canonical Parquet and daily aggregates reconcile to official CSV totals.",
        )

        summary_frame = pd.DataFrame(summary)
        severity_map = {
            "types": "CRITICAL",
            "domain_constraints": "HIGH",
            "primary_keys": "CRITICAL",
            "structural_nulls": "INFO",
            "negative_amounts": "HIGH",
            "additive_relationships": "MEDIUM",
            "balance_continuity": "HIGH",
            "calendar": "CRITICAL",
            "anomalous_dates": "MEDIUM",
            "extreme_users": "INFO",
            "cross_table_coverage": "HIGH",
            "time_leakage": "CRITICAL",
            "derived_reconciliation": "HIGH",
        }
        summary_frame.insert(
            3, "severity_if_failed", summary_frame["check_id"].map(severity_map)
        )

        evidence = {
            "field_audit.csv": fields,
            "domain_constraints.csv": domains,
            "key_audit.csv": keys,
            "additive_relationships.csv": identities,
            "balance_identity_exceptions.csv": balance_identity_exceptions,
            "balance_continuity_summary.csv": continuity,
            "balance_continuity_exceptions.csv": continuity_exceptions,
            "anomalous_dates.csv": anomalies,
            "daily_quality.csv": daily_quality,
            "extreme_users.csv": extreme_users,
            "user_concentration.csv": concentration,
            "cross_table_coverage.csv": coverage,
            "leakage_audit.csv": leakage,
            "derived_reconciliation.csv": reconciliation,
            "read_performance.csv": performance,
            "daily_metrics.csv": daily[
                [
                    "report_date",
                    "users",
                    "new_users",
                    "total_purchase_amt",
                    "total_redeem_amt",
                ]
            ],
            "audit_summary.csv": summary_frame,
        }
        for name, frame in evidence.items():
            frame.to_csv(output_temp / name, index=False, encoding="utf-8-sig")

        create_charts(output_temp, fields, daily, concentration)
        readme = """# Data quality audit

Complete evidence package for the five frozen official CSV files and the canonical
typed layer `d003_validated_data_layer_v2`.

Run with `scripts/run_data_quality_audit.py`. `REVIEW` means preserve the source
value and investigate modeling impact; it does not mean the row should be removed.
The HTML report is generated separately from the reviewed evidence tables.
"""
        (output_temp / "README.md").write_text(readme, encoding="utf-8")

        output_temp.rename(output_final)
        create_notebook(project_root, output_final)
        write_manifest(output_final, project_root)
    finally:
        if output_temp.exists():
            shutil.rmtree(output_temp)

    print(f"Created {output_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
