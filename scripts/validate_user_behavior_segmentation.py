#!/usr/bin/env python3
"""Validate the canonical user behavior segmentation research artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data/derived/d007_user_behavior_segmentation_research_v6"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    snapshots = pd.read_parquet(output / "user_behavior_snapshots.parquet")
    thresholds = pd.read_csv(output / "rule_thresholds.csv")
    diagnostics = pd.read_csv(output / "cluster_k_diagnostics.csv")
    rule_summary = pd.read_csv(output / "manual_rule_summary.csv")
    segment_summary = pd.read_csv(output / "segment_future_outcome_summary.csv")
    strength = pd.read_csv(output / "segment_predictive_strength.csv")
    cold_start = pd.read_csv(output / "future_new_user_summary.csv")

    windows = [row["id"] for row in manifest["windows"]]
    require(list(snapshots["window_id"].drop_duplicates()) == windows, "window order mismatch")
    require(len(snapshots) == manifest["snapshot_rows"], "snapshot row count mismatch")
    require(
        not snapshots.duplicated(["window_id", "user_id"]).any(),
        "duplicate user-window rows",
    )
    require(
        (pd.to_datetime(snapshots["first_seen_life"]) < pd.to_datetime(snapshots["cutoff_date"])).all(),
        "history includes first-seen date at or after cutoff",
    )
    require(
        (pd.to_datetime(snapshots["last_seen_life"]) < pd.to_datetime(snapshots["cutoff_date"])).all(),
        "history includes rows at or after cutoff",
    )
    require(
        snapshots[[column for column in snapshots if column.startswith("future_") and column.endswith("_amt")]].ge(0).all().all(),
        "negative future outcomes",
    )

    no_active = (
        (snapshots["direct_purchase_amt_sum_life"] == 0)
        & (snapshots["total_redeem_amt_sum_life"] == 0)
    )
    observed_never_used = (
        no_active
        & (snapshots["tenure_days"] >= 28)
        & (snapshots["max_balance_life"] == 0)
    )
    passive_balance_only = (
        no_active
        & (snapshots["tenure_days"] >= 28)
        & (snapshots["max_balance_life"] > 0)
    )
    require(
        np.array_equal(snapshots["rule_no_active_history"].to_numpy(), no_active.to_numpy()),
        "no-active rule mismatch",
    )
    require(
        np.array_equal(
            snapshots["rule_history_insufficient"].to_numpy(),
            (snapshots["tenure_days"] < 28).to_numpy(),
        ),
        "history-insufficient rule mismatch",
    )
    require(
        np.array_equal(snapshots["rule_observed_never_used"].to_numpy(), observed_never_used.to_numpy()),
        "observed-never-used rule mismatch",
    )
    require(
        np.array_equal(snapshots["rule_passive_balance_only"].to_numpy(), passive_balance_only.to_numpy()),
        "passive-balance-only rule mismatch",
    )
    require(
        not (snapshots["rule_observed_never_used"] & snapshots["rule_history_insufficient"]).any(),
        "long-observed and history-insufficient states overlap",
    )

    for window_id, group in diagnostics.groupby("window_id", sort=False):
        require(group["k"].tolist() == list(range(3, 9)), f"invalid K grid for {window_id}")
        require(int(group["selected"].sum()) == 1, f"selected K count != 1 for {window_id}")
        require(group["algorithmic_stability_ari"].between(-1, 1).all(), f"invalid ARI for {window_id}")
        require(group["smallest_cluster_share"].between(0, 1).all(), f"invalid cluster share for {window_id}")

    for method in ["manual", "automatic", "hybrid"]:
        subset = segment_summary.loc[segment_summary["method"] == method]
        actual = subset.groupby("window_id", sort=False)["users"].sum().astype(int)
        expected = snapshots.groupby("window_id", sort=False).size().astype(int)
        require(actual.to_dict() == expected.to_dict(), f"{method} segment counts do not sum")

    rule_columns = [column for column in snapshots if column.startswith("rule_")]
    for rule in rule_columns:
        expected = snapshots.groupby("window_id", sort=False)[rule].sum().astype(int)
        actual = (
            rule_summary.loc[rule_summary["rule"] == rule.removeprefix("rule_")]
            .set_index("window_id")["users"]
            .astype(int)
        )
        require(actual.to_dict() == expected.to_dict(), f"rule count mismatch: {rule}")

    require(
        set(strength["method"]) == {"manual", "automatic", "hybrid"},
        "missing method in predictive strength",
    )
    require(strength["eta_squared_log1p"].between(0, 1).all(), "invalid eta-squared")
    require(thresholds["active_transfer_volume_90_p90"].gt(0).all(), "invalid volume threshold")
    require(thresholds["nonzero_consume_event_p50"].gt(0).all(), "invalid consumption threshold")
    require(cold_start["new_user_share"].between(0, 1).all(), "invalid cold-start share")

    for entry in manifest["files"]:
        path = output / entry["path"]
        require(path.exists(), f"missing manifest file: {path.name}")
        require(path.stat().st_size == entry["bytes"], f"size mismatch: {path.name}")
        require(sha256(path) == entry["sha256"], f"hash mismatch: {path.name}")
    require(all(not path.stat().st_mode & 0o200 for path in output.iterdir()), "output is not read-only")

    print(
        "USER BEHAVIOR SEGMENTATION VALIDATION PASSED: "
        f"windows={len(windows)}, rows={len(snapshots):,}, files={len(manifest['files']) + 1}"
    )


if __name__ == "__main__":
    main()
