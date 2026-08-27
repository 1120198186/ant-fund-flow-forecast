#!/usr/bin/env python3
"""Validate the direct-feature analysis evidence and delivery artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import nbformat
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "derived" / "d005_direct_feature_analysis_v1"
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    demographic = pd.read_csv(output / "demographic_associations.csv")
    groups = pd.read_csv(output / "demographic_group_summary.csv")
    rates = pd.read_csv(output / "rate_associations.csv")
    transactions = pd.read_csv(output / "transaction_component_associations.csv")
    catalog = pd.read_csv(output / "feature_combination_catalog.csv")

    require(len(demographic) == 3 * 12 * 3, "demographic association grid is incomplete")
    require(len(rates) == 10 * 12, "rate association grid is incomplete")
    correlation_columns = [
        "pearson_raw",
        "pearson_log_target",
        "spearman",
        "residual_corr_log_target",
        "lag1_spearman",
        "lag7_spearman",
        "diff1_corr_log_target",
        "diff7_corr_log_target",
        "lagged_diff1_corr_log_target",
    ]
    for column in correlation_columns:
        require(rates[column].dropna().between(-1, 1).all(), f"invalid correlation: {column}")
    require(
        transactions[
            ["daily_spearman", "daily_log_pearson", "user_spearman"]
        ].stack().between(-1, 1).all(),
        "transaction correlation outside [-1,1]",
    )

    for feature, expected_groups in [("sex", 2), ("city", 7), ("constellation", 12)]:
        sample = groups[
            groups["feature"].eq(feature)
            & groups["target"].eq("total_purchase_amt")
        ]
        require(len(sample) == expected_groups, f"unexpected groups for {feature}")
        require(int(sample["users"].sum()) == 28_041, f"user count does not reconcile for {feature}")

    formulas = "\n".join(catalog["formula"].astype(str))
    require(
        "mfd_7daily_yield - mfd_daily_yield" not in formulas,
        "catalog subtracts incompatible fund-yield units",
    )
    predictive_composition = catalog[catalog["family"].eq("behavior_composition")]
    require(
        predictive_composition["formula"].str.startswith("lag(").all(),
        "same-day behavior composition appears as predictive feature",
    )
    require(
        catalog.loc[catalog["priority"].isin(["P0", "P1"]), "leakage_rule"]
        .str.len()
        .gt(20)
        .all(),
        "predictive feature missing leakage rule",
    )

    notebook = nbformat.read(output / "direct_feature_analysis.ipynb", as_version=4)
    errors = [
        result
        for cell in notebook.cells
        if cell.cell_type == "code"
        for result in cell.get("outputs", [])
        if result.get("output_type") == "error"
    ]
    require(not errors, "executed notebook contains errors")

    artifact = json.loads((output / "report_artifact.json").read_text(encoding="utf-8"))
    require(len(artifact["manifest"]["blocks"]) == 12, "report block count is not 12")
    require(
        all(source.get("query", {}).get("sql") for source in artifact["sources"]),
        "report source lacks reproducible SQL",
    )
    html = output / "direct_feature_report.html"
    require(html.is_file() and html.stat().st_size > 100_000, "portable report missing")
    if html.is_file():
        text = html.read_text(encoding="utf-8")
        require("直接特征与目标关联分析" in text, "report title missing")
        require(text.count("data-artifact-block-id=") == 12, "HTML block count mismatch")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = output / row["path"]
        require(path.is_file(), f"manifest file missing: {row['path']}")
        if path.is_file():
            require(sha256(path) == row["sha256"], f"manifest hash mismatch: {row['path']}")

    if failures:
        print("DIRECT FEATURE ANALYSIS VALIDATION FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("DIRECT FEATURE ANALYSIS VALIDATION PASSED")
    print("- demographic grid: 3 features x 12 targets x 3 behavior metrics")
    print("- rate grid: 10 rate fields x 12 targets at date grain")
    print("- leakage rules, notebook, 12-block HTML, and manifest hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
