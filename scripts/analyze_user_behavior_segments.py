#!/usr/bin/env python3
"""Research leakage-safe manual and automatic user behavior segmentation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


AMOUNT_COLUMNS = [
    "direct_purchase_amt",
    "purchase_bank_amt",
    "purchase_bal_amt",
    "total_redeem_amt",
    "transfer_amt",
    "consume_amt",
    "share_amt",
]
EVENT_COLUMNS = [
    "direct_purchase_amt",
    "total_redeem_amt",
    "transfer_amt",
    "consume_amt",
]
FUTURE_COLUMNS = [
    "direct_purchase_amt",
    "total_redeem_amt",
    "transfer_amt",
    "consume_amt",
]
CLUSTER_FEATURES = [
    "log_end_balance",
    "log_direct_90",
    "log_redeem_90",
    "log_consume_90",
    "direct_days_per_30_90",
    "redeem_days_per_30_90",
    "consume_days_per_30_90",
    "recency_active_scaled",
    "roundtrip_ratio_90",
    "consume_share_90",
    "net_flow_share_90",
    "bank_purchase_share_90",
]
RULE_COLUMNS = [
    "rule_no_active_history",
    "rule_history_insufficient",
    "rule_observed_never_used",
    "rule_passive_balance_only",
    "rule_funded_never_withdrawn",
    "rule_dormant_zero_balance",
    "rule_low_turnover_holder",
    "rule_frequent_large_roundtrip",
    "rule_frequent_small_consumer",
    "rule_transfer_oriented",
    "rule_consumption_oriented",
]

WINDOW_LABELS = {
    "rolling_2014_03": "2014年3月",
    "rolling_2014_04": "2014年4月",
    "rolling_2014_05": "2014年5月",
    "rolling_2014_06": "2014年6月",
    "rolling_2014_07": "2014年7月",
    "holdout_2014_08": "2014年8月",
}
RULE_LABELS = {
    "no_active_history": "历史无主动行为",
    "history_insufficient": "历史不足28天",
    "observed_never_used": "可观测期从未使用",
    "passive_balance_only": "仅有被动余额",
    "funded_never_withdrawn": "充值后从未取出",
    "dormant_zero_balance": "沉寂且零余额",
    "low_turnover_holder": "低周转持有",
    "frequent_large_roundtrip": "频繁大额进出",
    "frequent_small_consumer": "频繁小额消费",
    "transfer_oriented": "转账导向",
    "consumption_oriented": "消费导向",
}
SEGMENT_LABELS = {
    **RULE_LABELS,
    "balanced_or_other_active": "均衡或其他活跃",
    "inflow_dominant": "净转入导向",
    "outflow_dominant": "净转出导向",
}
METHOD_LABELS = {"manual": "手工规则", "automatic": "自动聚类", "hybrid": "混合分层"}
TARGET_LABELS = {
    "direct_purchase_amt": "直接转入",
    "total_redeem_amt": "总转出",
    "transfer_amt": "主动转账",
    "consume_amt": "消费",
}


@dataclass(frozen=True)
class Window:
    window_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "data/derived/d003_validated_data_layer_v2/user_balance_daily.parquet",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=root / "validation/splits/rolling_30d_2014_03_08_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/derived/d007_user_behavior_segmentation_research_v6",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_windows(path: Path) -> list[Window]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Window(row["id"], pd.Timestamp(row["start"]), pd.Timestamp(row["end"]))
        for row in payload["windows"]
    ]


def load_source(path: Path) -> pd.DataFrame:
    columns = [
        "user_id",
        "report_date",
        "tBalance",
        "yBalance",
        *AMOUNT_COLUMNS,
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    for column in ["user_id", "tBalance", "yBalance", *AMOUNT_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values(["report_date", "user_id"], ignore_index=True)
    return frame


def last_positive_date(frame: pd.DataFrame, column: str) -> pd.Series:
    active = frame.loc[frame[column] > 0, ["user_id", "report_date"]]
    if active.empty:
        return pd.Series(dtype="datetime64[ns]")
    return active.groupby("user_id", observed=True)["report_date"].max()


def positive_median(frame: pd.DataFrame, column: str) -> pd.Series:
    active = frame.loc[frame[column] > 0, ["user_id", column]]
    if active.empty:
        return pd.Series(dtype="float64")
    return active.groupby("user_id", observed=True)[column].median()


def aggregate_period(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    work = frame.copy()
    for column in EVENT_COLUMNS:
        work[f"{column}_event"] = work[column] > 0
    work["active_flow_event"] = (
        (work["direct_purchase_amt"] > 0) | (work["total_redeem_amt"] > 0)
    )
    grouped = work.groupby("user_id", observed=True)
    named: dict[str, tuple[str, str]] = {
        f"rows_{suffix}": ("report_date", "size"),
        f"first_seen_{suffix}": ("report_date", "min"),
        f"last_seen_{suffix}": ("report_date", "max"),
        f"start_balance_{suffix}": ("yBalance", "first"),
        f"end_balance_{suffix}": ("tBalance", "last"),
        f"max_balance_{suffix}": ("tBalance", "max"),
        f"mean_balance_{suffix}": ("tBalance", "mean"),
        f"std_balance_{suffix}": ("tBalance", "std"),
        f"positive_balance_days_{suffix}": ("tBalance", lambda x: int((x > 0).sum())),
        f"active_flow_days_{suffix}": ("active_flow_event", "sum"),
    }
    for column in AMOUNT_COLUMNS:
        named[f"{column}_sum_{suffix}"] = (column, "sum")
        named[f"{column}_max_{suffix}"] = (column, "max")
    for column in EVENT_COLUMNS:
        named[f"{column}_days_{suffix}"] = (f"{column}_event", "sum")
    result = grouped.agg(**named)
    for column in EVENT_COLUMNS:
        result[f"{column}_median_nonzero_{suffix}"] = positive_median(work, column)
    return result


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    left = numerator.astype("float64")
    right = denominator.astype("float64")
    return pd.Series(
        np.divide(left, right, out=np.zeros(len(left), dtype=float), where=right != 0),
        index=numerator.index,
    )


def build_snapshot(
    source: pd.DataFrame,
    window: Window,
) -> tuple[pd.DataFrame, dict[str, float | int | str], dict[str, float | int]]:
    history = source.loc[source["report_date"] < window.start].copy()
    recent_90 = history.loc[history["report_date"] >= window.start - pd.Timedelta(days=90)]
    recent_30 = history.loc[history["report_date"] >= window.start - pd.Timedelta(days=30)]
    future = source.loc[source["report_date"].between(window.start, window.end)].copy()

    lifetime = aggregate_period(history, "life")
    last_90 = aggregate_period(recent_90, "90")
    last_30 = aggregate_period(recent_30, "30")
    snapshot = lifetime.join(last_90, how="left").join(last_30, how="left")

    date_columns = [column for column in snapshot if "seen" in column]
    numeric_columns = [column for column in snapshot if column not in date_columns]
    snapshot[numeric_columns] = snapshot[numeric_columns].fillna(0)
    snapshot["cutoff_date"] = window.start
    snapshot["window_id"] = window.window_id
    snapshot["tenure_days"] = (
        window.start - snapshot["first_seen_life"]
    ).dt.days.clip(lower=1)

    for event, short in [
        ("direct_purchase_amt", "purchase"),
        ("total_redeem_amt", "redeem"),
        ("transfer_amt", "transfer"),
        ("consume_amt", "consume"),
    ]:
        last_date = last_positive_date(history, event)
        snapshot[f"last_{short}_date"] = last_date
        snapshot[f"recency_{short}_days"] = (
            window.start - snapshot[f"last_{short}_date"]
        ).dt.days.fillna(999).clip(lower=1, upper=999)

    snapshot["last_active_date"] = snapshot[
        ["last_purchase_date", "last_redeem_date"]
    ].max(axis=1)
    snapshot["recency_active_days"] = (
        window.start - snapshot["last_active_date"]
    ).dt.days.fillna(999).clip(lower=1, upper=999)

    direct_90 = snapshot["direct_purchase_amt_sum_90"]
    redeem_90 = snapshot["total_redeem_amt_sum_90"]
    transfer_90 = snapshot["transfer_amt_sum_90"]
    consume_90 = snapshot["consume_amt_sum_90"]
    transfer_volume_90 = direct_90 + transfer_90
    snapshot["active_transfer_volume_90"] = transfer_volume_90
    snapshot["roundtrip_ratio_90"] = safe_divide(
        pd.concat([direct_90, transfer_90], axis=1).min(axis=1),
        pd.concat([direct_90, transfer_90], axis=1).max(axis=1),
    )
    volume_90 = direct_90 + redeem_90
    snapshot["net_flow_share_90"] = safe_divide(direct_90 - redeem_90, volume_90)
    snapshot["consume_share_90"] = safe_divide(consume_90, redeem_90)
    snapshot["transfer_share_90"] = safe_divide(transfer_90, redeem_90)
    snapshot["bank_purchase_share_90"] = safe_divide(
        snapshot["purchase_bank_amt_sum_90"], direct_90
    )
    snapshot["positive_balance_ratio_90"] = safe_divide(
        snapshot["positive_balance_days_90"], snapshot["rows_90"]
    )
    snapshot["balance_cv_90"] = safe_divide(
        snapshot["std_balance_90"].fillna(0), snapshot["mean_balance_90"].abs() + 1
    )
    for event, short in [
        ("direct_purchase_amt", "direct"),
        ("total_redeem_amt", "redeem"),
        ("consume_amt", "consume"),
    ]:
        snapshot[f"{short}_days_per_30_90"] = (
            snapshot[f"{event}_days_90"].astype(float) / 3.0
        )

    active_90 = snapshot["active_flow_days_90"] > 0
    two_way_90 = (snapshot["direct_purchase_amt_days_90"] > 0) & (
        snapshot["transfer_amt_days_90"] > 0
    )
    consumer_90 = snapshot["consume_amt_days_90"] > 0
    frequency_threshold = float(
        snapshot.loc[active_90, "active_flow_days_90"].quantile(0.75)
    )
    volume_threshold = float(
        snapshot.loc[two_way_90, "active_transfer_volume_90"].quantile(0.90)
    )
    consumer_frequency_threshold = float(
        snapshot.loc[consumer_90, "consume_amt_days_90"].quantile(0.75)
    )
    positive_consumption = recent_90.loc[recent_90["consume_amt"] > 0, "consume_amt"]
    small_consumption_threshold = float(positive_consumption.quantile(0.50))
    thresholds: dict[str, float | int | str] = {
        "window_id": window.window_id,
        "cutoff_date": window.start.date().isoformat(),
        "active_flow_days_90_p75": frequency_threshold,
        "active_transfer_volume_90_p90": volume_threshold,
        "consume_days_90_p75": consumer_frequency_threshold,
        "nonzero_consume_event_p50": small_consumption_threshold,
        "roundtrip_ratio_min": 0.50,
        "small_consume_share_min": 0.50,
        "orientation_share_min": 0.80,
        "dormancy_days": 90,
    }

    direct_life = snapshot["direct_purchase_amt_sum_life"]
    redeem_life = snapshot["total_redeem_amt_sum_life"]
    end_balance = snapshot["end_balance_life"]
    snapshot["rule_no_active_history"] = (direct_life == 0) & (redeem_life == 0)
    snapshot["rule_history_insufficient"] = snapshot["tenure_days"] < 28
    snapshot["rule_observed_never_used"] = (
        snapshot["rule_no_active_history"]
        & ~snapshot["rule_history_insufficient"]
        & (snapshot["max_balance_life"] == 0)
    )
    snapshot["rule_passive_balance_only"] = (
        snapshot["rule_no_active_history"]
        & ~snapshot["rule_history_insufficient"]
        & (snapshot["max_balance_life"] > 0)
    )
    snapshot["rule_funded_never_withdrawn"] = (
        (direct_life > 0) & (redeem_life == 0) & (end_balance > 0)
    )
    snapshot["rule_dormant_zero_balance"] = (
        ~snapshot["rule_no_active_history"]
        & (snapshot["active_flow_days_90"] == 0)
        & (end_balance == 0)
    )
    snapshot["rule_low_turnover_holder"] = (
        (end_balance > 0)
        & (snapshot["active_flow_days_90"] <= 1)
        & (redeem_90 <= 0.05 * (end_balance + direct_90 + 1))
    )
    snapshot["rule_frequent_large_roundtrip"] = (
        (snapshot["active_flow_days_90"] >= max(4.0, frequency_threshold))
        & (snapshot["direct_purchase_amt_days_90"] >= 3)
        & (snapshot["transfer_amt_days_90"] >= 3)
        & (snapshot["active_transfer_volume_90"] >= volume_threshold)
        & (direct_90 > 0)
        & (transfer_90 > 0)
        & (snapshot["roundtrip_ratio_90"] >= 0.50)
    )
    snapshot["rule_frequent_small_consumer"] = (
        (snapshot["consume_amt_days_90"] >= max(3.0, consumer_frequency_threshold))
        & (snapshot["consume_amt_median_nonzero_90"] <= small_consumption_threshold)
        & (snapshot["consume_share_90"] >= 0.50)
    )
    snapshot["rule_transfer_oriented"] = (
        (redeem_90 > 0) & (snapshot["transfer_share_90"] >= 0.80)
    )
    snapshot["rule_consumption_oriented"] = (
        (redeem_90 > 0) & (snapshot["consume_share_90"] >= 0.80)
    )

    primary = np.full(len(snapshot), "balanced_or_other_active", dtype=object)
    rule_order = [
        ("rule_history_insufficient", "history_insufficient"),
        ("rule_observed_never_used", "observed_never_used"),
        ("rule_passive_balance_only", "passive_balance_only"),
        ("rule_funded_never_withdrawn", "funded_never_withdrawn"),
        ("rule_frequent_large_roundtrip", "frequent_large_roundtrip"),
        ("rule_frequent_small_consumer", "frequent_small_consumer"),
        ("rule_dormant_zero_balance", "dormant_zero_balance"),
        ("rule_low_turnover_holder", "low_turnover_holder"),
    ]
    assigned = np.zeros(len(snapshot), dtype=bool)
    for column, label in rule_order:
        mask = snapshot[column].to_numpy(dtype=bool) & ~assigned
        primary[mask] = label
        assigned |= mask
    remaining = ~assigned
    transfer_mask = remaining & snapshot["rule_transfer_oriented"].to_numpy(dtype=bool)
    primary[transfer_mask] = "transfer_oriented"
    assigned |= transfer_mask
    remaining = ~assigned
    consume_mask = remaining & snapshot["rule_consumption_oriented"].to_numpy(dtype=bool)
    primary[consume_mask] = "consumption_oriented"
    assigned |= consume_mask
    remaining = ~assigned
    inflow_mask = remaining & (snapshot["net_flow_share_90"].to_numpy() >= 0.50)
    primary[inflow_mask] = "inflow_dominant"
    assigned |= inflow_mask
    remaining = ~assigned
    outflow_mask = remaining & (snapshot["net_flow_share_90"].to_numpy() <= -0.50)
    primary[outflow_mask] = "outflow_dominant"
    snapshot["manual_primary_segment"] = primary

    future_group = future.groupby("user_id", observed=True)
    future_agg = future_group[FUTURE_COLUMNS].sum().add_prefix("future_")
    for column in FUTURE_COLUMNS:
        future_agg[f"future_{column}_days"] = future_group[column].apply(
            lambda values: int((values > 0).sum())
        )
    snapshot = snapshot.join(future_agg, how="left")
    future_cols = [column for column in snapshot if column.startswith("future_")]
    snapshot[future_cols] = snapshot[future_cols].fillna(0)
    snapshot["future_any_direct"] = snapshot["future_direct_purchase_amt"] > 0
    snapshot["future_any_redeem"] = snapshot["future_total_redeem_amt"] > 0

    history_ids = set(snapshot.index.to_numpy())
    future_new = future.loc[~future["user_id"].isin(history_ids)]
    future_new_stats: dict[str, float | int] = {
        "window_id": window.window_id,
        "future_users": int(future["user_id"].nunique()),
        "known_future_users": int(future.loc[future["user_id"].isin(history_ids), "user_id"].nunique()),
        "new_future_users": int(future_new["user_id"].nunique()),
        "new_user_share": float(
            future_new["user_id"].nunique() / max(1, future["user_id"].nunique())
        ),
    }
    for column in FUTURE_COLUMNS:
        total = float(future[column].sum())
        new_total = float(future_new[column].sum())
        future_new_stats[f"new_{column}"] = new_total
        future_new_stats[f"new_{column}_share"] = new_total / total if total else 0.0

    snapshot = snapshot.reset_index()
    return snapshot, thresholds, future_new_stats


def robust_scale(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    values = frame[columns].to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scales = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scales = np.where(scales > 1e-12, scales, np.where(fallback > 1e-12, fallback, 1.0))
    scaled = np.clip((values - medians) / scales, -6.0, 6.0)
    metadata = {
        column: {"median": float(median), "scale": float(scale)}
        for column, median, scale in zip(columns, medians, scales)
    }
    return scaled, metadata


def kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, x.shape[1]), dtype=float)
    centers[0] = x[rng.integers(len(x))]
    closest = ((x - centers[0]) ** 2).sum(axis=1)
    for index in range(1, k):
        total = closest.sum()
        if total <= 0:
            centers[index] = x[rng.integers(len(x))]
        else:
            centers[index] = x[rng.choice(len(x), p=closest / total)]
        distance = ((x - centers[index]) ** 2).sum(axis=1)
        closest = np.minimum(closest, distance)
    return centers


def run_kmeans_once(
    x: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    centers = kmeans_plus_plus(x, k, rng)
    labels = np.zeros(len(x), dtype=np.int16)
    for _ in range(max_iter):
        distance = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distance.argmin(axis=1).astype(np.int16)
        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels
        new_centers = centers.copy()
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                new_centers[cluster] = x[rng.integers(len(x))]
        if np.allclose(centers, new_centers, atol=1e-7, rtol=0):
            centers = new_centers
            break
        centers = new_centers
    inertia = float(((x - centers[labels]) ** 2).sum())
    return labels, centers, inertia


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    table = pd.crosstab(pd.Series(a, name="a"), pd.Series(b, name="b")).to_numpy()
    n = int(table.sum())
    if n < 2:
        return 1.0

    def comb2(values: np.ndarray) -> float:
        values = values.astype(float)
        return float((values * (values - 1) / 2).sum())

    sum_cells = comb2(table)
    sum_rows = comb2(table.sum(axis=1))
    sum_cols = comb2(table.sum(axis=0))
    total = n * (n - 1) / 2
    expected = sum_rows * sum_cols / total if total else 0.0
    maximum = 0.5 * (sum_rows + sum_cols)
    denominator = maximum - expected
    return (sum_cells - expected) / denominator if denominator else 1.0


def cluster_metrics(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> tuple[float, float]:
    k = len(centers)
    n = len(x)
    within = float(((x - centers[labels]) ** 2).sum())
    overall = x.mean(axis=0)
    counts = np.bincount(labels, minlength=k)
    between = float((((centers - overall) ** 2).sum(axis=1) * counts).sum())
    calinski = (between / max(1, k - 1)) / (within / max(1, n - k)) if within else math.inf
    scatter = np.array(
        [
            np.linalg.norm(x[labels == cluster] - centers[cluster], axis=1).mean()
            if np.any(labels == cluster)
            else 0.0
            for cluster in range(k)
        ]
    )
    center_distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    ratios = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i != j and center_distance[i, j] > 0:
                ratios[i, j] = (scatter[i] + scatter[j]) / center_distance[i, j]
    davies_bouldin = float(ratios.max(axis=1).mean())
    return calinski, davies_bouldin


def prepare_cluster_features(snapshot: pd.DataFrame) -> None:
    snapshot["log_end_balance"] = np.log1p(snapshot["end_balance_life"].clip(lower=0))
    snapshot["log_direct_90"] = np.log1p(snapshot["direct_purchase_amt_sum_90"].clip(lower=0))
    snapshot["log_redeem_90"] = np.log1p(snapshot["total_redeem_amt_sum_90"].clip(lower=0))
    snapshot["log_consume_90"] = np.log1p(snapshot["consume_amt_sum_90"].clip(lower=0))
    snapshot["recency_active_scaled"] = snapshot["recency_active_days"].clip(upper=90) / 90.0


def select_cluster_model(
    snapshot: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str]], dict[str, dict[str, float]]]:
    prepare_cluster_features(snapshot)
    eligible = (
        (snapshot["active_flow_days_90"] > 0)
        | (snapshot["end_balance_life"] > 0)
    ) & ~snapshot["rule_no_active_history"] & ~snapshot["rule_history_insufficient"]
    eligible_frame = snapshot.loc[eligible].copy()
    x, scaling = robust_scale(eligible_frame, CLUSTER_FEATURES)
    diagnostic_rows: list[dict[str, float | int | str]] = []
    candidates: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for k in range(3, 9):
        runs = [run_kmeans_once(x, k, seed + 1009 * k + offset) for offset in range(5)]
        best = min(runs, key=lambda item: item[2])
        pairwise_ari = [
            adjusted_rand_index(runs[i][0], runs[j][0])
            for i in range(len(runs))
            for j in range(i + 1, len(runs))
        ]
        calinski, davies = cluster_metrics(x, best[0], best[1])
        min_share = float(np.bincount(best[0], minlength=k).min() / len(best[0]))
        diagnostic_rows.append(
            {
                "window_id": str(snapshot["window_id"].iloc[0]),
                "k": k,
                "inertia_per_user": best[2] / len(x),
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
                "algorithmic_stability_ari": float(np.median(pairwise_ari)),
                "smallest_cluster_share": min_share,
                "eligible_users": len(x),
            }
        )
        candidates[k] = best
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics["rank_ch"] = diagnostics["calinski_harabasz"].rank(ascending=False)
    diagnostics["rank_db"] = diagnostics["davies_bouldin"].rank(ascending=True)
    diagnostics["rank_stability"] = diagnostics["algorithmic_stability_ari"].rank(ascending=False)
    diagnostics["rank_min_share"] = diagnostics["smallest_cluster_share"].rank(ascending=False)
    diagnostics["selection_rank_sum"] = diagnostics[
        ["rank_ch", "rank_db", "rank_stability", "rank_min_share"]
    ].sum(axis=1)
    selected_k = int(
        diagnostics.sort_values(["selection_rank_sum", "k"]).iloc[0]["k"]
    )
    diagnostics["selected"] = diagnostics["k"] == selected_k
    labels, _, _ = candidates[selected_k]
    snapshot["auto_cluster"] = "not_cluster_eligible"
    snapshot.loc[eligible, "auto_cluster"] = [f"k{selected_k}_c{label}" for label in labels]
    structural = snapshot["manual_primary_segment"].isin(
        [
            "history_insufficient",
            "observed_never_used",
            "passive_balance_only",
            "funded_never_withdrawn",
            "dormant_zero_balance",
        ]
    )
    snapshot["hybrid_segment"] = snapshot["auto_cluster"]
    snapshot.loc[structural, "hybrid_segment"] = snapshot.loc[
        structural, "manual_primary_segment"
    ]
    return snapshot, diagnostics.to_dict("records"), scaling


def eta_squared(categories: pd.Series, values: pd.Series) -> float:
    y = np.log1p(values.to_numpy(dtype=float))
    if len(y) == 0 or np.var(y) <= 0:
        return 0.0
    overall = y.mean()
    frame = pd.DataFrame({"category": categories.astype(str), "y": y})
    grouped = frame.groupby("category", observed=True)["y"].agg(["count", "mean"])
    between = float((grouped["count"] * (grouped["mean"] - overall) ** 2).sum())
    total = float(((y - overall) ** 2).sum())
    return between / total if total else 0.0


def summarize_segmentation(snapshot: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    future_totals = {target: float(snapshot[f"future_{target}"].sum()) for target in FUTURE_COLUMNS}
    overall_direct_rate = float(snapshot["future_any_direct"].mean())
    overall_redeem_rate = float(snapshot["future_any_redeem"].mean())
    rows: list[dict[str, float | int | str]] = []
    for segment, group in snapshot.groupby(column, observed=True):
        row: dict[str, float | int | str] = {
            "window_id": str(snapshot["window_id"].iloc[0]),
            "method": method,
            "segment": str(segment),
            "users": len(group),
            "user_share": len(group) / len(snapshot),
            "future_direct_active_rate": float(group["future_any_direct"].mean()),
            "future_redeem_active_rate": float(group["future_any_redeem"].mean()),
            "future_direct_active_rate_lift": float(group["future_any_direct"].mean()) / overall_direct_rate if overall_direct_rate else 0.0,
            "future_redeem_active_rate_lift": float(group["future_any_redeem"].mean()) / overall_redeem_rate if overall_redeem_rate else 0.0,
            "current_balance_share": float(group["end_balance_life"].sum()) / max(1.0, float(snapshot["end_balance_life"].sum())),
        }
        for target in FUTURE_COLUMNS:
            field = f"future_{target}"
            row[f"{field}_share"] = float(group[field].sum()) / future_totals[target] if future_totals[target] else 0.0
            row[f"{field}_mean_per_user"] = float(group[field].mean())
            row[f"{field}_median_per_user"] = float(group[field].median())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_rules(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    overall_direct_rate = float(snapshot["future_any_direct"].mean())
    overall_redeem_rate = float(snapshot["future_any_redeem"].mean())
    for rule in RULE_COLUMNS:
        group = snapshot.loc[snapshot[rule]]
        rows.append(
            {
                "window_id": str(snapshot["window_id"].iloc[0]),
                "rule": rule.removeprefix("rule_"),
                "users": len(group),
                "coverage": len(group) / len(snapshot),
                "future_direct_active_rate": float(group["future_any_direct"].mean()) if len(group) else 0.0,
                "future_redeem_active_rate": float(group["future_any_redeem"].mean()) if len(group) else 0.0,
                "future_direct_active_rate_lift": float(group["future_any_direct"].mean()) / overall_direct_rate if len(group) and overall_direct_rate else 0.0,
                "future_redeem_active_rate_lift": float(group["future_any_redeem"].mean()) / overall_redeem_rate if len(group) and overall_redeem_rate else 0.0,
                "future_direct_purchase_mean": float(group["future_direct_purchase_amt"].mean()) if len(group) else 0.0,
                "future_total_redeem_mean": float(group["future_total_redeem_amt"].mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_overlap(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for index, left in enumerate(RULE_COLUMNS):
        for right in RULE_COLUMNS[index + 1 :]:
            left_mask = snapshot[left].to_numpy(dtype=bool)
            right_mask = snapshot[right].to_numpy(dtype=bool)
            union = (left_mask | right_mask).sum()
            intersection = (left_mask & right_mask).sum()
            rows.append(
                {
                    "window_id": str(snapshot["window_id"].iloc[0]),
                    "rule_a": left.removeprefix("rule_"),
                    "rule_b": right.removeprefix("rule_"),
                    "intersection_users": int(intersection),
                    "jaccard": float(intersection / union) if union else 0.0,
                }
            )
    return pd.DataFrame(rows)


def predictive_strength(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    methods = {
        "manual": "manual_primary_segment",
        "automatic": "auto_cluster",
        "hybrid": "hybrid_segment",
    }
    for method, column in methods.items():
        for target in FUTURE_COLUMNS:
            rows.append(
                {
                    "window_id": str(snapshot["window_id"].iloc[0]),
                    "method": method,
                    "future_target": target,
                    "eta_squared_log1p": eta_squared(snapshot[column], snapshot[f"future_{target}"]),
                    "segments": int(snapshot[column].nunique()),
                }
            )
    return pd.DataFrame(rows)


def cluster_profile(snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "end_balance_life",
        "direct_purchase_amt_sum_90",
        "total_redeem_amt_sum_90",
        "transfer_amt_sum_90",
        "consume_amt_sum_90",
        "active_flow_days_90",
        "recency_active_days",
        "roundtrip_ratio_90",
        "consume_share_90",
        "bank_purchase_share_90",
        "net_flow_share_90",
    ]
    eligible = snapshot["auto_cluster"] != "not_cluster_eligible"
    grouped = snapshot.loc[eligible].groupby("auto_cluster", observed=True)
    profile = grouped[columns].median().reset_index()
    profile.insert(0, "window_id", str(snapshot["window_id"].iloc[0]))
    profile["users"] = grouped.size().to_numpy()
    return profile


def temporal_segment_stability(snapshots: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    windows = list(dict.fromkeys(snapshots["window_id"].astype(str)))
    rows: list[dict[str, float | int | str]] = []
    for previous, current in zip(windows, windows[1:]):
        left = snapshots.loc[snapshots["window_id"] == previous, ["user_id", column]]
        right = snapshots.loc[snapshots["window_id"] == current, ["user_id", column]]
        merged = left.merge(right, on="user_id", suffixes=("_previous", "_current"))
        rows.append(
            {
                "method": method,
                "previous_window": previous,
                "current_window": current,
                "common_users": len(merged),
                "adjusted_rand_index": adjusted_rand_index(
                    pd.factorize(merged[f"{column}_previous"])[0],
                    pd.factorize(merged[f"{column}_current"])[0],
                ),
                "unchanged_label_share": float(
                    (merged[f"{column}_previous"] == merged[f"{column}_current"]).mean()
                ) if len(merged) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_rule_coverage(rule_summary: pd.DataFrame, path: Path) -> None:
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=r"C:\Windows\Fonts\msyh.ttc"
    ).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    window_order = list(dict.fromkeys(rule_summary["window_id"].astype(str)))
    ordered = rule_summary.copy()
    ordered["window_id"] = pd.Categorical(
        ordered["window_id"], categories=window_order, ordered=True
    )
    pivot = ordered.pivot(index="window_id", columns="rule", values="coverage").sort_index()
    focus = [
        "no_active_history",
        "history_insufficient",
        "observed_never_used",
        "funded_never_withdrawn",
        "frequent_large_roundtrip",
        "frequent_small_consumer",
        "dormant_zero_balance",
    ]
    plot_data = pivot[focus].copy()
    plot_data.index = [WINDOW_LABELS.get(str(value), str(value)) for value in plot_data.index]
    plot_data.columns = [RULE_LABELS.get(str(value), str(value)) for value in plot_data.columns]
    ax = plot_data.plot(kind="line", marker="o", figsize=(11, 6))
    ax.set_title("六个滚动截止点的行为标签覆盖率")
    ax.set_xlabel("")
    ax.set_ylabel("占已知用户比例")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="行为标签", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_predictive_strength(strength: pd.DataFrame, path: Path) -> None:
    summary = (
        strength.groupby(["method", "future_target"], observed=True)["eta_squared_log1p"]
        .median()
        .unstack("method")
    )
    summary = summary.reindex(columns=["manual", "automatic", "hybrid"])
    summary.index = [TARGET_LABELS.get(str(value), str(value)) for value in summary.index]
    summary.columns = [METHOD_LABELS.get(str(value), str(value)) for value in summary.columns]
    ax = summary.plot(kind="bar", figsize=(10, 6))
    ax.set_title("六个滚动窗口中未来30天结果的中位区分度")
    ax.set_xlabel("")
    ax.set_ylabel("用户结果 log1p 的 Eta 平方")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    decimals: int = 3,
    column_labels: dict[str, str] | None = None,
    value_labels: dict[str, dict[str, str]] | None = None,
) -> str:
    display = frame[columns].copy()
    for column, labels in (value_labels or {}).items():
        if column in display:
            display[column] = display[column].astype(str).map(lambda value: labels.get(value, value))
    for column in display.select_dtypes(include="number"):
        if pd.api.types.is_integer_dtype(display[column].dtype):
            display[column] = display[column].map(lambda value: f"{int(value)}")
        else:
            display[column] = display[column].map(lambda value: f"{value:.{decimals}f}")
    display = display.rename(columns=column_labels or {})
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *rows])


def build_report(
    output: Path,
    rule_summary: pd.DataFrame,
    primary_summary: pd.DataFrame,
    cluster_diagnostics: pd.DataFrame,
    strength: pd.DataFrame,
    cold_start: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    rule_rollup = rule_summary.groupby("rule", observed=True).agg(
        median_coverage=("coverage", "median"),
        min_coverage=("coverage", "min"),
        max_coverage=("coverage", "max"),
        median_future_direct_lift=("future_direct_active_rate_lift", "median"),
        median_future_redeem_lift=("future_redeem_active_rate_lift", "median"),
    ).reset_index().sort_values("median_coverage", ascending=False)
    strength_rollup = strength.groupby(["method", "future_target"], observed=True).agg(
        median_eta2=("eta_squared_log1p", "median"),
        min_eta2=("eta_squared_log1p", "min"),
        max_eta2=("eta_squared_log1p", "max"),
    ).reset_index()
    selected_k = {
        int(key): int(value)
        for key, value in cluster_diagnostics.loc[cluster_diagnostics["selected"]]
        .groupby("k")
        .size()
        .items()
    }
    selected_models = cluster_diagnostics.loc[cluster_diagnostics["selected"]]
    selected_stability_min = selected_models["algorithmic_stability_ari"].min()
    selected_stability_max = selected_models["algorithmic_stability_ari"].max()
    final_selected_stability = selected_models.loc[
        selected_models["window_id"] == selected_models["window_id"].iloc[-1],
        "algorithmic_stability_ari",
    ].iloc[0]
    final_window = str(primary_summary["window_id"].drop_duplicates().iloc[-1])
    final_segments = primary_summary.loc[primary_summary["window_id"] == final_window].sort_values(
        "user_share", ascending=False
    )
    manual_stability = stability.loc[stability["method"] == "manual", "adjusted_rand_index"].median()
    auto_stability = stability.loc[stability["method"] == "automatic", "adjusted_rand_index"].median()
    hybrid_stability = stability.loc[stability["method"] == "hybrid", "adjusted_rand_index"].median()
    selected_k_text = "、".join(
        f"{key}类出现在{value}个窗口" for key, value in selected_k.items()
    )
    rule_report = rule_rollup.assign(
        median_coverage_pct=rule_rollup["median_coverage"] * 100,
        min_coverage_pct=rule_rollup["min_coverage"] * 100,
        max_coverage_pct=rule_rollup["max_coverage"] * 100,
    )
    final_report = final_segments.assign(
        users=final_segments["users"].astype(int),
        user_share_pct=final_segments["user_share"] * 100,
        future_direct_active_rate_pct=final_segments["future_direct_active_rate"] * 100,
        future_redeem_active_rate_pct=final_segments["future_redeem_active_rate"] * 100,
        future_direct_purchase_amt_share_pct=final_segments["future_direct_purchase_amt_share"] * 100,
        future_total_redeem_amt_share_pct=final_segments["future_total_redeem_amt_share"] * 100,
    )
    report = f"""# 用户行为分层研究报告

## 核心结论

用户行为分层这条路线成立，但建议采用**混合分层**，而不是在纯手工规则和纯自动聚类之间二选一。

- 手工规则先识别历史不足、可观测期从未使用、充值后从未取出、沉寂且零余额等生命周期状态。这些状态有明确含义，否则会主导距离型聚类。
- 自动聚类只用于其余活跃用户，探索余额、金额、频率、资金方向和消费结构等难以手工穷举的组合。
- 六个无泄漏窗口选出的簇数为：{selected_k_text}。所选模型不同初始化之间的 ARI 为 {selected_stability_min:.3f}–{selected_stability_max:.3f}，其中8月只有 {final_selected_stability:.3f}，因此自动簇只能视为滚动行为风格，不能视为永久用户身份。
- 相邻月份 ARI 中位数为：手工规则 {manual_stability:.3f}、自动聚类 {auto_stability:.3f}、混合分层 {hybrid_stability:.3f}。混合分层对直接转入、总转出和消费的区分度最高；主动转账则是手工规则略强，没有一种方法在所有目标上都胜出。
- 未来窗口内活跃用户中的新用户占比中位数为 {cold_start['new_user_share'].median():.1%}。真正的新用户在首次交易前没有历史行为，仍需单独处理冷启动。

## 四类行为在本数据中的定义

官方数据没有单独的开户事件。因此，“开通后没用过”只能保守定义为：至少已观测28天、历史主动资金流为零、历史最高余额也为零。历史不足28天的用户单列为“历史不足”，避免把冷启动误判成长期未使用。“充值后一直没取”要求历史上发生过直接转入、没有任何转出，且当前余额为正。

“频繁大额进出”和“频繁小额消费”使用各截止点之前的数据动态计算阈值。大额和高频依据历史90天人群分位数；小额消费依据截止点前非零消费日金额中位数。所有标签和阈值均不使用未来30天结果。

{markdown_table(rule_report, ['rule', 'median_coverage_pct', 'min_coverage_pct', 'max_coverage_pct', 'median_future_direct_lift', 'median_future_redeem_lift'], decimals=2, column_labels={'rule': '行为标签', 'median_coverage_pct': '覆盖率中位数(%)', 'min_coverage_pct': '最低覆盖率(%)', 'max_coverage_pct': '最高覆盖率(%)', 'median_future_direct_lift': '未来转入活跃率倍数', 'median_future_redeem_lift': '未来转出活跃率倍数'}, value_labels={'rule': RULE_LABELS})}

![六个滚动截止点的行为标签覆盖率](rule_coverage.png)

## 8月截止点的手工分层

下表描述8月测试窗开始前已经出现的用户。未来资金量占比只用于评价标签是否有区分能力，不参与标签分配。

{markdown_table(final_report, ['segment', 'users', 'user_share_pct', 'future_direct_active_rate_pct', 'future_redeem_active_rate_pct', 'future_direct_purchase_amt_share_pct', 'future_total_redeem_amt_share_pct'], decimals=2, column_labels={'segment': '用户层', 'users': '用户数', 'user_share_pct': '用户占比(%)', 'future_direct_active_rate_pct': '未来转入活跃率(%)', 'future_redeem_active_rate_pct': '未来转出活跃率(%)', 'future_direct_purchase_amt_share_pct': '未来直接转入量占比(%)', 'future_total_redeem_amt_share_pct': '未来总转出量占比(%)'}, value_labels={'segment': SEGMENT_LABELS})}

## 手工、自动与混合分层的区分度

Eta平方衡量用户层级的分组能够区分多少未来30天对数结果差异。它只是关联诊断，不代表因果关系，也不等同于真实预测模型的分数提升。

{markdown_table(strength_rollup, ['method', 'future_target', 'median_eta2', 'min_eta2', 'max_eta2'], column_labels={'method': '分层方法', 'future_target': '未来目标', 'median_eta2': 'Eta平方中位数', 'min_eta2': '最低值', 'max_eta2': '最高值'}, value_labels={'method': METHOD_LABELS, 'future_target': TARGET_LABELS})}

![三种分层方法对未来30天结果的区分度](predictive_strength.png)

## 建议的建模方式

1. 生命周期和状态规则保留为多个独立标签，不要为了展示方便而强制压成一个互斥类别。
2. 只对历史充足的活跃用户聚类；金额先做对数变换，再稳健标准化。每个滚动折都重新拟合缩放参数和聚类中心。
3. 将规则标签和聚类结果聚合成日级特征：各层用户数、活跃用户数、余额、滞后转入转出、近期分层迁移量。
4. 按“基础时序模型→增加手工标签→增加活跃用户聚类→增加状态迁移”依次做消融。只有滚动分数中位数提升且最差窗口没有明显退化时才保留。
5. 新用户使用到达人数和早期生命周期行为单独建模，因为预测时无法获得完整历史分层。

## 建议补充的行为维度

- 生命周期：新出现、刚完成首次充值、成熟、沉寂、重新激活。
- 资金方向：净转入、净转出、双向均衡周转。
- 资金用途：主动转账导向、消费导向。
- 价值与集中度：高余额、高资金周转、一次性大额、少数日期贡献大部分金额。
- 时间偏好：工作日或周末、月初或月末、固定日期周期、行为最近发生时间。
- 渠道偏好：银行卡转入或余额转入、转银行卡或转余额。

## 结论边界

本研究覆盖2014年3月至8月六个滚动窗口，只使用各截止点之前的行为构造标签，并用随后30天评价区分度。当前结果不说明聚类与交易存在因果关系，也尚未证明能够提高比赛分数。下一道验收关是把这些分层放进真实预测流程，执行无泄漏的六折消融测试。
"""
    (output / "用户行为分层研究报告.md").write_text(report, encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def make_read_only(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_file():
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source_path = resolve(args.source, root)
    split_path = resolve(args.splits, root)
    output = resolve(args.output, root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    output.mkdir(parents=True)

    source = load_source(source_path)
    windows = load_windows(split_path)
    snapshots: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, float | int | str]] = []
    cold_start_rows: list[dict[str, float | int]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    scaling_payload: dict[str, dict[str, dict[str, float]]] = {}
    rule_summaries: list[pd.DataFrame] = []
    segment_summaries: list[pd.DataFrame] = []
    overlap_summaries: list[pd.DataFrame] = []
    strength_summaries: list[pd.DataFrame] = []
    profile_summaries: list[pd.DataFrame] = []

    for offset, window in enumerate(windows):
        snapshot, thresholds, cold_start = build_snapshot(source, window)
        snapshot, diagnostics, scaling = select_cluster_model(
            snapshot, args.seed + offset * 100_000
        )
        snapshots.append(snapshot)
        threshold_rows.append(thresholds)
        cold_start_rows.append(cold_start)
        diagnostic_rows.extend(diagnostics)
        scaling_payload[window.window_id] = scaling
        rule_summaries.append(summarize_rules(snapshot))
        segment_summaries.extend(
            [
                summarize_segmentation(snapshot, "manual_primary_segment", "manual"),
                summarize_segmentation(snapshot, "auto_cluster", "automatic"),
                summarize_segmentation(snapshot, "hybrid_segment", "hybrid"),
            ]
        )
        overlap_summaries.append(summarize_overlap(snapshot))
        strength_summaries.append(predictive_strength(snapshot))
        profile_summaries.append(cluster_profile(snapshot))
        print(f"built {window.window_id}: users={len(snapshot):,}")

    all_snapshots = pd.concat(snapshots, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    cold_start = pd.DataFrame(cold_start_rows)
    cluster_diagnostics = pd.DataFrame(diagnostic_rows)
    rule_summary = pd.concat(rule_summaries, ignore_index=True)
    segment_summary = pd.concat(segment_summaries, ignore_index=True)
    overlap_summary = pd.concat(overlap_summaries, ignore_index=True)
    strength = pd.concat(strength_summaries, ignore_index=True)
    profiles = pd.concat(profile_summaries, ignore_index=True)
    stability = pd.concat(
        [
            temporal_segment_stability(all_snapshots, "manual_primary_segment", "manual"),
            temporal_segment_stability(all_snapshots, "auto_cluster", "automatic"),
            temporal_segment_stability(all_snapshots, "hybrid_segment", "hybrid"),
        ],
        ignore_index=True,
    )

    snapshot_path = output / "user_behavior_snapshots.parquet"
    all_snapshots.to_parquet(snapshot_path, index=False, compression="zstd")
    write_csv(thresholds, output / "rule_thresholds.csv")
    write_csv(cold_start, output / "future_new_user_summary.csv")
    write_csv(cluster_diagnostics, output / "cluster_k_diagnostics.csv")
    write_csv(rule_summary, output / "manual_rule_summary.csv")
    write_csv(segment_summary, output / "segment_future_outcome_summary.csv")
    write_csv(overlap_summary, output / "manual_rule_overlap.csv")
    write_csv(strength, output / "segment_predictive_strength.csv")
    write_csv(profiles, output / "auto_cluster_profiles.csv")
    write_csv(stability, output / "segment_temporal_stability.csv")
    (output / "cluster_scaling.json").write_text(
        json.dumps(scaling_payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    plot_rule_coverage(rule_summary, output / "rule_coverage.png")
    plot_predictive_strength(strength, output / "predictive_strength.png")
    primary_summary = segment_summary.loc[segment_summary["method"] == "manual"]
    build_report(
        output,
        rule_summary,
        primary_summary,
        cluster_diagnostics,
        strength,
        cold_start,
        stability,
    )

    field_definitions = {
        "grain": "one row per user observed before each rolling cutoff",
        "assignment_boundary": "all behavior features and thresholds use report_date < cutoff_date",
        "evaluation_boundary": "future_* fields use the 30-day rolling window only",
        "manual_rules": {
            "no_active_history": "no historical direct purchase and no historical redemption at any tenure",
            "history_insufficient": "fewer than 28 days between first observation and cutoff",
            "observed_never_used": "at least 28 days observed, no active flow, and zero historical maximum balance",
            "passive_balance_only": "at least 28 days observed, no active flow, but a positive historical balance",
            "funded_never_withdrawn": "historical direct purchase, zero historical redemption, positive current balance",
            "dormant_zero_balance": "previously active, no active flow in 90 days, zero current balance",
            "low_turnover_holder": "positive balance, at most one active-flow day in 90 days, low recent redemption",
            "frequent_large_roundtrip": "90-day frequency >= fold p75, at least three direct-purchase and transfer days, transfer volume >= fold p90, roundtrip ratio >= 0.5",
            "frequent_small_consumer": "consumption frequency >= fold p75, median event <= fold event p50, consumption >= 50% of redemption",
            "transfer_oriented": "transfer is at least 80% of 90-day redemption",
            "consumption_oriented": "consumption is at least 80% of 90-day redemption",
        },
        "automatic_clustering": {
            "population": "users with at least 28 days of history, active history, and recent active flow or positive current balance",
            "preprocessing": "log transforms for amounts, robust median/IQR scaling, clip to [-6, 6]",
            "candidate_k": "3 through 8",
            "selection": "lowest combined rank over Calinski-Harabasz, Davies-Bouldin, repeated-start ARI, and minimum cluster share",
            "future_labels_used_for_clustering": False,
        },
    }
    (output / "FIELD_DEFINITIONS.json").write_text(
        json.dumps(field_definitions, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    generated_files = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "dataset_version": output.name,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": str(Path(__file__).relative_to(root)).replace("/", "\\"),
        "generator_sha256": sha256(Path(__file__)),
        "sources": [
            {"path": str(source_path.relative_to(root)).replace("/", "\\"), "sha256": sha256(source_path)},
            {"path": str(split_path.relative_to(root)).replace("/", "\\"), "sha256": sha256(split_path)},
        ],
        "competition_cutoff": "2014-08-31",
        "windows": [
            {"id": w.window_id, "start": w.start.date().isoformat(), "end": w.end.date().isoformat()}
            for w in windows
        ],
        "snapshot_rows": len(all_snapshots),
        "snapshot_users_by_window": all_snapshots.groupby("window_id")["user_id"].nunique().to_dict(),
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated_files
        ],
        "runtime": {"python": os.sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    make_read_only(output.iterdir())
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
