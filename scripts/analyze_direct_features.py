#!/usr/bin/env python3
"""Analyze direct feature/target associations and propose derived features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_VERSION = "d005_direct_feature_analysis_v1"
LAYER_VERSION = "d003_validated_data_layer_v2"
DEMOGRAPHICS = ["sex", "city", "constellation"]
PRIMARY_TARGETS = [
    "total_purchase_amt",
    "total_redeem_amt",
    "direct_purchase_amt",
    "purchase_bank_amt",
    "transfer_amt",
]
FLOW_TARGETS = [
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
STOCK_TARGETS = ["tBalance", "yBalance"]
TARGETS = FLOW_TARGETS + STOCK_TARGETS
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": values.astype(float), "group": groups}).dropna()
    if frame.empty:
        return float("nan")
    grand_mean = frame["value"].mean()
    denominator = ((frame["value"] - grand_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    grouped = frame.groupby("group", observed=True)["value"].agg(["count", "mean"])
    numerator = (grouped["count"] * (grouped["mean"] - grand_mean) ** 2).sum()
    return float(numerator / denominator)


def effect_label(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value < 0.001:
        return "negligible"
    if value < 0.01:
        return "weak"
    if value < 0.06:
        return "small"
    if value < 0.14:
        return "moderate"
    return "large"


def spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": left.astype(float), "right": right.astype(float)}).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame["left"].rank(method="average").corr(frame["right"].rank(method="average")))


def residualize(values: pd.Series, dates: pd.Series) -> pd.Series:
    numeric = values.astype(float).to_numpy()
    trend = np.arange(len(numeric), dtype=float)
    weekdays = pd.get_dummies(pd.to_datetime(dates).dt.weekday, drop_first=True).to_numpy(
        dtype=float
    )
    design = np.column_stack([np.ones(len(numeric)), trend, weekdays])
    valid = np.isfinite(numeric) & np.isfinite(design).all(axis=1)
    result = np.full(len(numeric), np.nan)
    if valid.sum() > design.shape[1]:
        coefficients, *_ = np.linalg.lstsq(design[valid], numeric[valid], rcond=None)
        result[valid] = numeric[valid] - design[valid] @ coefficients
    return pd.Series(result, index=values.index)


def build_user_metrics(balance: pd.DataFrame) -> pd.DataFrame:
    grouped = balance.groupby("user_id", observed=True)
    users = grouped.agg(
        first_date=("report_date", "min"),
        last_date=("report_date", "max"),
        observed_rows=("report_date", "size"),
    )
    global_end = balance["report_date"].max()
    users["exposure_days"] = (global_end - users["first_date"]).dt.days + 1
    for target in TARGETS:
        users[f"{target}__total"] = grouped[target].sum()
        users[f"{target}__active_days"] = grouped[target].agg(
            lambda values: int((values > 0).sum())
        )
        users[f"{target}__per_exposure_day"] = (
            users[f"{target}__total"] / users["exposure_days"]
        )
        users[f"{target}__active_day_rate"] = (
            users[f"{target}__active_days"] / users["exposure_days"]
        )
        users[f"{target}__positive_day_amount"] = np.where(
            users[f"{target}__active_days"].gt(0),
            users[f"{target}__total"] / users[f"{target}__active_days"],
            0.0,
        )
    return users.reset_index()


def demographic_associations(
    user_metrics: pd.DataFrame, profile: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = user_metrics.merge(profile, on="user_id", how="left", validate="one_to_one")
    association_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    combination_rows: list[dict[str, object]] = []

    for feature in DEMOGRAPHICS:
        for target in TARGETS:
            amount = np.log1p(joined[f"{target}__per_exposure_day"])
            activity = joined[f"{target}__active_day_rate"]
            positive_amount = np.log1p(joined[f"{target}__positive_day_amount"])
            amount_eta = eta_squared(amount, joined[feature])
            activity_eta = eta_squared(activity, joined[feature])
            positive_amount_eta = eta_squared(positive_amount, joined[feature])
            association_rows.extend(
                [
                    {
                        "feature": feature,
                        "target": target,
                        "metric": "log1p_amount_per_exposure_day",
                        "eta_squared": amount_eta,
                        "effect_label": effect_label(amount_eta),
                        "users": len(joined),
                        "groups": joined[feature].nunique(),
                        "analysis_grain": "user",
                    },
                    {
                        "feature": feature,
                        "target": target,
                        "metric": "active_day_rate",
                        "eta_squared": activity_eta,
                        "effect_label": effect_label(activity_eta),
                        "users": len(joined),
                        "groups": joined[feature].nunique(),
                        "analysis_grain": "user",
                    },
                    {
                        "feature": feature,
                        "target": target,
                        "metric": "log1p_positive_day_amount",
                        "eta_squared": positive_amount_eta,
                        "effect_label": effect_label(positive_amount_eta),
                        "users": len(joined),
                        "groups": joined[feature].nunique(),
                        "analysis_grain": "user",
                    },
                ]
            )

            grouped = joined.groupby(feature, observed=True).agg(
                users=("user_id", "size"),
                amount_mean=(f"{target}__per_exposure_day", "mean"),
                amount_median=(f"{target}__per_exposure_day", "median"),
                amount_p90=(f"{target}__per_exposure_day", lambda x: x.quantile(0.9)),
                active_day_rate_mean=(f"{target}__active_day_rate", "mean"),
                positive_day_amount_mean=(f"{target}__positive_day_amount", "mean"),
                positive_day_amount_median=(f"{target}__positive_day_amount", "median"),
            )
            grouped = grouped.reset_index().rename(columns={feature: "feature_value"})
            grouped.insert(0, "feature", feature)
            grouped.insert(2, "target", target)
            group_rows.extend(grouped.to_dict(orient="records"))

    for feature_name, columns in [
        ("sex_x_city", ["sex", "city"]),
        ("sex_x_constellation", ["sex", "constellation"]),
        ("city_x_constellation", ["city", "constellation"]),
        ("sex_x_city_x_constellation", ["sex", "city", "constellation"]),
    ]:
        combination = joined[columns].astype(str).agg("|".join, axis=1)
        group_sizes = combination.value_counts()
        eligible = combination.where(combination.map(group_sizes).ge(50), "__SMALL_GROUP__")
        for target in TARGETS:
            amount_eta = eta_squared(
                np.log1p(joined[f"{target}__per_exposure_day"]), eligible
            )
            combination_rows.append(
                {
                    "feature": feature_name,
                    "components": "+".join(columns),
                    "target": target,
                    "eta_squared": amount_eta,
                    "effect_label": effect_label(amount_eta),
                    "raw_groups": int(combination.nunique()),
                    "groups_after_min_50_pooling": int(eligible.nunique()),
                    "minimum_group_users": 50,
                    "analysis_grain": "user",
                }
            )
    return (
        pd.DataFrame(association_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(combination_rows),
    )


def build_daily_table(
    balance: pd.DataFrame, fund: pd.DataFrame, shibor: pd.DataFrame
) -> pd.DataFrame:
    daily = (
        balance.groupby("report_date", observed=True)[TARGETS]
        .sum()
        .reset_index()
        .merge(
            fund.rename(columns={"mfd_date": "report_date"}),
            on="report_date",
            how="left",
            validate="one_to_one",
        )
        .merge(
            shibor.rename(columns={"mfd_date": "report_date"}),
            on="report_date",
            how="left",
            validate="one_to_one",
        )
        .sort_values("report_date")
        .reset_index(drop=True)
    )
    return daily


def rate_associations(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(daily["report_date"])
    for feature in RATE_FEATURES:
        feature_values = daily[feature].astype(float)
        feature_residual = residualize(feature_values, dates)
        for target in TARGETS:
            raw_target = daily[target].astype(float)
            log_target = np.log1p(raw_target)
            target_residual = residualize(log_target, dates)
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "pearson_raw": feature_values.corr(raw_target, method="pearson"),
                    "pearson_log_target": feature_values.corr(log_target, method="pearson"),
                    "spearman": spearman(feature_values, raw_target),
                    "residual_corr_log_target": feature_residual.corr(
                        target_residual, method="pearson"
                    ),
                    "lag1_spearman": spearman(feature_values.shift(1), raw_target),
                    "lag7_spearman": spearman(feature_values.shift(7), raw_target),
                    "diff1_corr_log_target": feature_values.diff(1).corr(
                        log_target.diff(1), method="pearson"
                    ),
                    "diff7_corr_log_target": feature_values.diff(7).corr(
                        log_target.diff(7), method="pearson"
                    ),
                    "lagged_diff1_corr_log_target": feature_values.diff(1).shift(1).corr(
                        log_target.diff(1), method="pearson"
                    ),
                    "days": len(daily),
                    "analysis_grain": "date",
                }
            )
    result = pd.DataFrame(rows)
    result["direct_abs_spearman"] = result["spearman"].abs()
    result["predictor_safe_abs_lag1"] = result["lag1_spearman"].abs()
    result["short_run_abs_diff1"] = result["diff1_corr_log_target"].abs()
    result["predictor_safe_abs_lagged_diff1"] = result[
        "lagged_diff1_corr_log_target"
    ].abs()
    return result


def pair_type(left: str, right: str) -> str:
    pair = frozenset([left, right])
    mechanical = {
        frozenset(["total_purchase_amt", "direct_purchase_amt"]),
        frozenset(["direct_purchase_amt", "purchase_bal_amt"]),
        frozenset(["direct_purchase_amt", "purchase_bank_amt"]),
        frozenset(["total_purchase_amt", "share_amt"]),
        frozenset(["total_redeem_amt", "consume_amt"]),
        frozenset(["total_redeem_amt", "transfer_amt"]),
        frozenset(["transfer_amt", "tftobal_amt"]),
        frozenset(["transfer_amt", "tftocard_amt"]),
    }
    if pair in mechanical:
        return "mechanical_component_relationship"
    if left in STOCK_TARGETS or right in STOCK_TARGETS:
        return "balance_behavior_relationship"
    return "behavioral_flow_relationship"


def transaction_associations(
    daily: pd.DataFrame, user_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    user_frame = user_metrics[
        [f"{target}__per_exposure_day" for target in TARGETS]
    ].rename(columns=lambda name: name.replace("__per_exposure_day", ""))
    for index, left in enumerate(TARGETS):
        for right in TARGETS[index + 1 :]:
            relationship = pair_type(left, right)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "relationship_type": relationship,
                    "daily_spearman": spearman(daily[left], daily[right]),
                    "daily_log_pearson": np.log1p(daily[left]).corr(
                        np.log1p(daily[right]), method="pearson"
                    ),
                    "user_spearman": spearman(user_frame[left], user_frame[right]),
                    "prediction_use": "descriptive_only_same_day; use lagged form for forecasting",
                }
            )
    result = pd.DataFrame(rows)
    result["max_abs_correlation"] = result[
        ["daily_spearman", "daily_log_pearson", "user_spearman"]
    ].abs().max(axis=1)
    return result.sort_values("max_abs_correlation", ascending=False)


def rate_collinearity(daily: pd.DataFrame) -> pd.DataFrame:
    matrix = daily[RATE_FEATURES].rank(method="average").corr(method="pearson")
    rows = []
    for index, left in enumerate(RATE_FEATURES):
        for right in RATE_FEATURES[index + 1 :]:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": matrix.loc[left, right],
                    "abs_spearman": abs(matrix.loc[left, right]),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)


def feature_catalog() -> pd.DataFrame:
    rows: list[tuple[str, str, str, str, str, str, str]] = []

    def add(
        priority: str,
        family: str,
        name: str,
        formula: str,
        grain: str,
        rationale: str,
        leakage_rule: str,
    ) -> None:
        rows.append((priority, family, name, formula, grain, rationale, leakage_rule))

    for feature in DEMOGRAPHICS:
        add(
            "P0",
            "raw_demographic",
            feature,
            feature,
            "user",
            "Stable user segmentation baseline.",
            "Safe as static input; fit categorical encoding inside each fold.",
        )
    for name, formula in [
        ("sex_x_city", "sex + city"),
        ("sex_x_constellation", "sex + constellation"),
        ("city_x_constellation", "city + constellation"),
        ("sex_x_city_x_constellation", "sex + city + constellation"),
    ]:
        add(
            "P1" if name.startswith("sex_x_city_x") else "P0",
            "demographic_cross",
            name,
            formula,
            "user",
            "Captures heterogeneous demographic segments missed by main effects.",
            "Pool cells below 50 training users; create mapping within each fold.",
        )
    for rate in RATE_FEATURES:
        for lag in [1, 7]:
            add(
                "P0",
                "rate_lag",
                f"{rate}_lag{lag}",
                f"lag({rate}, {lag})",
                "date",
                "Tests delayed response to published rates.",
                "Only values published by forecast cutoff may be used.",
            )
        for delta in [1, 7]:
            add(
                "P0",
                "rate_change",
                f"{rate}_delta{delta}_lag1",
                f"lag({rate} - lag({rate},{delta}), 1)",
                "date",
                "Captures rate direction and short-term repricing.",
                "Difference first, then lag the result before forecast use.",
            )
        for window in [7, 14, 28]:
            add(
                "P1",
                "rate_rolling",
                f"{rate}_mean{window}",
                f"rolling_mean(lag({rate},1), {window})",
                "date",
                "Captures rate regime rather than one-day noise.",
                "Shift before rolling; never center the window.",
            )
        for window in [7, 28]:
            add(
                "P1",
                "rate_volatility",
                f"{rate}_std{window}",
                f"rolling_std(lag({rate},1), {window})",
                "date",
                "Captures stable versus volatile rate regimes.",
                "Shift before rolling and fit any normalization inside the fold.",
            )
    for name, formula in [
        ("alipay_vs_shibor_on", "mfd_7daily_yield - Interest_O_N"),
        ("alipay_vs_shibor_1w", "mfd_7daily_yield - Interest_1_W"),
        ("alipay_vs_shibor_1m", "mfd_7daily_yield - Interest_1_M"),
        ("alipay_vs_shibor_3m", "mfd_7daily_yield - Interest_3_M"),
        ("shibor_slope_1y_on", "Interest_1_Y - Interest_O_N"),
        ("shibor_slope_3m_on", "Interest_3_M - Interest_O_N"),
        ("shibor_slope_1m_1w", "Interest_1_M - Interest_1_W"),
        (
            "shibor_curve_curvature",
            "2 * Interest_3_M - Interest_O_N - Interest_1_Y",
        ),
    ]:
        add(
            "P0",
            "rate_spread_curve",
            name,
            f"lag({formula}, 1)",
            "date",
            "Measures relative attractiveness and term-structure regime.",
            "Use lagged published rates; same-day version is descriptive only.",
        )
    for target in PRIMARY_TARGETS:
        for lag in [1, 7, 14, 28]:
            add(
                "P0",
                "target_history",
                f"{target}_lag{lag}",
                f"lag(daily_sum({target}), {lag})",
                "date",
                "Core autoregressive signal for daily forecasting.",
                "Compute independently inside each rolling split.",
            )
        for window in [7, 14, 28]:
            add(
                "P0",
                "target_history",
                f"{target}_rolling_mean{window}",
                f"rolling_mean(lag(daily_sum({target}),1), {window})",
                "date",
                "Captures recent level and seasonality.",
                "Shift one day before rolling.",
            )
    for name, formula in [
        ("bank_purchase_share", "purchase_bank_amt / direct_purchase_amt"),
        ("balance_purchase_share", "purchase_bal_amt / direct_purchase_amt"),
        ("share_purchase_share", "share_amt / total_purchase_amt"),
        ("transfer_redeem_share", "transfer_amt / total_redeem_amt"),
        ("consume_redeem_share", "consume_amt / total_redeem_amt"),
        ("transfer_to_card_share", "tftocard_amt / transfer_amt"),
        ("transfer_to_balance_share", "tftobal_amt / transfer_amt"),
        ("net_flow", "total_purchase_amt - total_redeem_amt"),
    ]:
        add(
            "P0",
            "behavior_composition",
            f"{name}_lag1",
            f"lag(daily({formula}), 1)",
            "date",
            "Summarizes payment-channel and redemption composition.",
            "Same-day formula is target leakage; only lagged/rolling forms are predictive.",
        )
    for demographic in DEMOGRAPHICS:
        for target in PRIMARY_TARGETS:
            add(
                "P1",
                "segment_history",
                f"{demographic}_{target}_rolling7",
                f"rolling_mean(lag(daily_sum({target}) by {demographic},1),7)",
                "date x segment",
                "Tests whether segment mix contributes incremental forecast signal.",
                "Aggregate and shift before joining forecast date.",
            )
    for name, formula in [
        (
            "rate_spread_x_bank_share",
            "alipay_vs_shibor_1w_lag1 * bank_purchase_share_lag1",
        ),
        (
            "rate_change_x_transfer_share",
            "delta7(mfd_7daily_yield)_lag1 * transfer_redeem_share_lag1",
        ),
        ("weekday_x_rate_spread", "weekday * alipay_vs_shibor_1w_lag1"),
        ("new_user_share_x_rate", "new_user_share_lag1 * mfd_7daily_yield_lag1"),
    ]:
        add(
            "P1",
            "cross_domain_interaction",
            name,
            formula,
            "date",
            "Tests context-dependent response to rates and user mix.",
            "All continuous inputs must be lagged before interaction.",
        )
    for name, formula in [
        ("all_pairwise_polynomials", "every numeric pair and square"),
        ("high_cardinality_user_id_cross", "user_id x demographic x rate"),
        ("full_period_target_encoding", "category mean target on all dates"),
        ("same_day_component_ratios", "ratios built from forecast-day outcomes"),
    ]:
        add(
            "P2_NOT_NOW",
            "defer_or_reject",
            name,
            formula,
            "mixed",
            "High overfit, dimensionality, or leakage risk at current stage.",
            "Do not use until a fold-safe pipeline and incremental-value evidence exist.",
        )
    return pd.DataFrame(
        rows,
        columns=[
            "priority",
            "family",
            "feature_name",
            "formula",
            "grain",
            "rationale",
            "leakage_rule",
        ],
    )


def create_charts(
    output: Path,
    demographic: pd.DataFrame,
    rates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"sex": "#275D84", "city": "#C08A2E", "constellation": "#8A5A7B"}

    demo = demographic[
        demographic["metric"].eq("log1p_amount_per_exposure_day")
        & demographic["target"].isin(PRIMARY_TARGETS)
    ].copy()
    demo["label"] = demo["feature"] + " / " + demo["target"]
    demo = demo.sort_values("eta_squared").tail(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(demo["label"], demo["eta_squared"], color=demo["feature"].map(colors))
    ax.set_xlabel("Eta-squared on log1p amount per exposure day")
    ax.set_title("Demographic association with primary transaction outcomes")
    fig.tight_layout()
    fig.savefig(output / "demographic_effect_sizes.png", dpi=170)
    plt.close(fig)

    rate_top = rates[rates["target"].isin(PRIMARY_TARGETS)].copy()
    rate_top["abs_short_run"] = rate_top["diff1_corr_log_target"].abs()
    rate_top = rate_top.nlargest(20, "abs_short_run").sort_values("diff1_corr_log_target")
    rate_top["label"] = rate_top["feature"] + " / " + rate_top["target"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        rate_top["label"],
        rate_top["diff1_corr_log_target"],
        color=np.where(rate_top["diff1_corr_log_target"] >= 0, "#275D84", "#C08A2E"),
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Correlation between one-day rate change and log target change")
    ax.set_title("Short-run rate association with primary daily outcomes")
    fig.tight_layout()
    fig.savefig(output / "rate_residual_correlations.png", dpi=170)
    plt.close(fig)

    component = transactions[
        transactions["left"].isin(PRIMARY_TARGETS)
        | transactions["right"].isin(PRIMARY_TARGETS)
    ].nlargest(20, "max_abs_correlation")
    component = component.sort_values("daily_spearman")
    component["label"] = component["left"] + " / " + component["right"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(component["label"], component["daily_spearman"], color="#275D84")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Daily Spearman correlation")
    ax.set_title("Direct relationship among transaction components")
    fig.tight_layout()
    fig.savefig(output / "transaction_component_correlations.png", dpi=170)
    plt.close(fig)


def write_feature_markdown(catalog: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Feature combination catalog",
        "",
        "Predictive features must use information available strictly before the forecast date.",
        "Same-day component ratios are useful for descriptive analysis only; use lagged forms in models.",
        "",
    ]
    for priority in ["P0", "P1", "P2_NOT_NOW"]:
        lines.extend([f"## {priority}", ""])
        subset = catalog[catalog["priority"].eq(priority)]
        for family, family_rows in subset.groupby("family", sort=False):
            lines.append(f"### {family}")
            lines.append("")
            lines.append("| Feature | Formula | Grain | Leakage rule |")
            lines.append("|---|---|---|---|")
            for row in family_rows.itertuples(index=False):
                lines.append(
                    f"| `{row.feature_name}` | `{row.formula}` | {row.grain} | {row.leakage_rule} |"
                )
            lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output: Path, project_root: Path) -> None:
    files = []
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        files.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": utc_now(),
        "generator": "scripts/analyze_direct_features.py",
        "command": ".venv\\Scripts\\python.exe scripts\\analyze_direct_features.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "source": f"data/derived/{LAYER_VERSION}/manifest.json",
        "units_of_analysis": {
            "demographics": "user, normalized by days since first appearance",
            "rates": "calendar date, 427 daily observations",
            "transaction_components": "daily aggregate and user normalized behavior",
        },
        "primary_targets": PRIMARY_TARGETS,
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    layer = project_root / "data" / "derived" / LAYER_VERSION
    output_final = project_root / "data" / "derived" / ANALYSIS_VERSION
    if output_final.exists():
        print(f"Refusing to overwrite existing analysis: {output_final}", file=sys.stderr)
        return 2
    output_temp = output_final.parent / f".{ANALYSIS_VERSION}.building-{uuid.uuid4().hex[:8]}"
    output_temp.mkdir(parents=True)
    try:
        balance = pd.read_parquet(
            layer / "user_balance_daily.parquet",
            columns=["user_id", "report_date", *TARGETS],
        )
        balance["report_date"] = pd.to_datetime(balance["report_date"])
        profile = pd.read_parquet(layer / "user_profile.parquet")
        fund = pd.read_parquet(layer / "fund_yield_daily.parquet")
        shibor = pd.read_parquet(layer / "shibor_daily.parquet")
        fund["mfd_date"] = pd.to_datetime(fund["mfd_date"])
        shibor["mfd_date"] = pd.to_datetime(shibor["mfd_date"])

        user_metrics = build_user_metrics(balance)
        demographic, group_summary, demographic_combinations = demographic_associations(
            user_metrics, profile
        )
        daily = build_daily_table(balance, fund, shibor)
        rates = rate_associations(daily)
        transactions = transaction_associations(daily, user_metrics)
        collinearity = rate_collinearity(daily)
        catalog = feature_catalog()

        outputs = {
            "demographic_associations.csv": demographic,
            "demographic_group_summary.csv": group_summary,
            "demographic_combination_associations.csv": demographic_combinations,
            "rate_associations.csv": rates,
            "rate_collinearity.csv": collinearity,
            "transaction_component_associations.csv": transactions,
            "daily_analysis_table.csv": daily,
            "feature_combination_catalog.csv": catalog,
        }
        for name, frame in outputs.items():
            frame.to_csv(output_temp / name, index=False, encoding="utf-8-sig")
        create_charts(output_temp, demographic, rates, transactions)
        write_feature_markdown(catalog, output_temp / "FEATURE_COMBINATIONS.md")
        (output_temp / "README.md").write_text(
            "# Direct feature analysis\n\n"
            "Direct association analysis for demographics, fund/bank rates, and transaction "
            "components. This is descriptive evidence, not a final feature-selection result. "
            "All predictive variants in `FEATURE_COMBINATIONS.md` include an explicit leakage rule.\n",
            encoding="utf-8",
        )
        output_temp.rename(output_final)
        write_manifest(output_final, project_root)
    finally:
        if output_temp.exists():
            shutil.rmtree(output_temp)
    print(f"Created {output_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
