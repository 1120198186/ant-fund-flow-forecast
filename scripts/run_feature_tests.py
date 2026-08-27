#!/usr/bin/env python3
"""Build reproducible periodicity, cold-start, and score-proxy diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from local_score import (
    PURCHASE_WEIGHT,
    REDEEM_WEIGHT,
    absolute_relative_error,
    score_from_error,
    weighted_daily_score,
)


TARGETS = ["total_purchase_amt", "total_redeem_amt"]
TARGET_LABELS = {
    "total_purchase_amt": "Purchase",
    "total_redeem_amt": "Redeem",
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "data/raw/official/user_balance_table.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/derived/d001_feature_tests",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def prepare_output(output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    material_files = [path for path in output.iterdir() if path.name != "feature_tests.ipynb"]
    if material_files and not overwrite:
        names = ", ".join(sorted(path.name for path in material_files)[:5])
        raise FileExistsError(
            f"output is not empty ({names}); use --overwrite or a new versioned folder"
        )


def load_balance(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        source,
        usecols=["user_id", "report_date", *TARGETS],
        dtype={
            "user_id": "int64",
            "report_date": "int32",
            "total_purchase_amt": "int64",
            "total_redeem_amt": "int64",
        },
    )
    frame["report_date"] = pd.to_datetime(
        frame["report_date"].astype(str), format="%Y%m%d"
    )
    return frame


def load_validation_definitions(
    project_root: Path,
) -> tuple[dict, dict[str, tuple[pd.Timestamp, pd.Timestamp]], list[Path]]:
    stress_path = project_root / "validation/splits/backtest_2013_09_v1.json"
    rolling_path = project_root / "validation/splits/rolling_30d_2014_03_08_v1.json"
    cold_path = project_root / "validation/splits/cold_start_random_2fold_v1.json"
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    rolling = json.loads(rolling_path.read_text(encoding="utf-8"))
    cold = json.loads(cold_path.read_text(encoding="utf-8"))
    windows = {
        stress["split_id"].removesuffix("_v1"): (
            pd.Timestamp(stress["holdout_window"]["start"]),
            pd.Timestamp(stress["holdout_window"]["end"]),
        )
    }
    windows.update(
        {
            row["id"]: (pd.Timestamp(row["start"]), pd.Timestamp(row["end"]))
            for row in rolling["windows"]
        }
    )
    return cold, windows, [stress_path, rolling_path, cold_path]


def build_daily(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.groupby("report_date", as_index=True)[TARGETS].sum().sort_index()
    expected_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing_dates = expected_index.difference(daily.index)
    if len(missing_dates):
        raise RuntimeError(f"missing calendar dates: {missing_dates.tolist()}")
    return daily.reindex(expected_index).rename_axis("report_date")


def build_quality_summary(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("rows", len(frame), "count"),
        ("columns_loaded", len(frame.columns), "count"),
        ("unique_users", frame["user_id"].nunique(), "count"),
        ("date_min", frame["report_date"].min().date().isoformat(), "date"),
        ("date_max", frame["report_date"].max().date().isoformat(), "date"),
        ("distinct_dates", frame["report_date"].nunique(), "count"),
        ("expected_calendar_dates", len(daily), "count"),
        ("duplicate_user_date_rows", frame.duplicated(["user_id", "report_date"]).sum(), "count"),
        ("null_user_id", frame["user_id"].isna().sum(), "count"),
        ("null_report_date", frame["report_date"].isna().sum(), "count"),
        ("null_purchase", frame[TARGETS[0]].isna().sum(), "count"),
        ("null_redeem", frame[TARGETS[1]].isna().sum(), "count"),
        ("negative_purchase_rows", (frame[TARGETS[0]] < 0).sum(), "count"),
        ("negative_redeem_rows", (frame[TARGETS[1]] < 0).sum(), "count"),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "unit"])


def build_user_diagnostics(
    frame: pd.DataFrame,
    seed: int,
    sample_size: int,
    folds: int,
    cold_split: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if sample_size % folds:
        raise ValueError("sample_size must be divisible by folds")

    first_seen = frame.groupby("user_id")["report_date"].min().sort_index()
    first_seen_month = (
        first_seen.dt.to_period("M")
        .value_counts()
        .sort_index()
        .rename_axis("first_seen_month")
        .reset_index(name="new_users")
    )
    first_seen_month["first_seen_month"] = first_seen_month[
        "first_seen_month"
    ].astype(str)

    train_end = pd.Timestamp(cold_split["eligibility"]["training_window_end"])
    holdout_start = pd.Timestamp(
        cold_split["eligibility"]["holdout_window"]["start"]
    )
    holdout_end = pd.Timestamp(cold_split["eligibility"]["holdout_window"]["end"])
    training_users = set(frame.loc[frame["report_date"] <= train_end, "user_id"].unique())
    holdout_mask = frame["report_date"].between(holdout_start, holdout_end)
    holdout_users = set(frame.loc[holdout_mask, "user_id"].unique())
    eligible = np.array(sorted(training_users), dtype=np.int64)
    if len(eligible) < sample_size:
        raise RuntimeError(
            f"only {len(eligible)} eligible users for requested sample {sample_size}"
        )
    salt = cold_split["sampling"]["salt"]
    ranked = pd.DataFrame({"user_id": eligible})
    ranked["selection_hash"] = ranked["user_id"].map(
        lambda user_id: hashlib.sha256(
            f"{salt}|{user_id}".encode("ascii")
        ).hexdigest()
    )
    ranked = ranked.sort_values(["selection_hash", "user_id"], ignore_index=True)
    sampled = ranked.head(sample_size).copy()
    sampled["selection_rank"] = np.arange(1, sample_size + 1)
    users_per_fold = sample_size // folds
    sampled["fold"] = np.repeat(np.arange(1, folds + 1), users_per_fold)
    sampled["seed"] = seed
    sampled["cutoff_date"] = train_end.date().isoformat()
    sampled["eligibility"] = "observed_on_or_before_cutoff"
    membership = sampled[
        [
            "user_id",
            "fold",
            "selection_rank",
            "selection_hash",
            "seed",
            "cutoff_date",
            "eligibility",
        ]
    ]

    holdout_sample = frame.loc[holdout_mask].merge(
        membership[["user_id", "fold"]], on="user_id", how="inner"
    )
    fold_activity = holdout_sample.groupby("fold").agg(
        active_users=("user_id", "nunique"),
        user_days=("user_id", "size"),
        purchase_amount=(TARGETS[0], "sum"),
        redeem_amount=(TARGETS[1], "sum"),
    )
    fold_summary = pd.DataFrame(
        {"fold": np.arange(1, folds + 1), "sampled_users": users_per_fold}
    ).merge(fold_activity.reset_index(), on="fold", how="left").fillna(0)
    holdout_totals = frame.loc[holdout_mask, TARGETS].sum()
    fold_summary["purchase_share_of_holdout"] = (
        fold_summary["purchase_amount"] / holdout_totals[TARGETS[0]]
    )
    fold_summary["redeem_share_of_holdout"] = (
        fold_summary["redeem_amount"] / holdout_totals[TARGETS[1]]
    )
    fold_hashes = {
        fold: hashlib.sha256(
            "".join(
                f"{user_id}\n"
                for user_id in membership.loc[membership["fold"] == fold, "user_id"]
            ).encode("ascii")
        ).hexdigest()
        for fold in range(1, folds + 1)
    }
    fold_summary["id_list_sha256"] = fold_summary["fold"].map(fold_hashes)

    cohort_masks = {
        "first_seen_2013_09": first_seen.between("2013-09-01", "2013-09-30"),
        "first_seen_2014_08_02_31": first_seen.between("2014-08-02", "2014-08-31"),
    }
    cohort_rows = []
    for cohort, mask in cohort_masks.items():
        cohort_rows.append(
            pd.DataFrame(
                {
                    "user_id": first_seen.index[mask],
                    "first_seen_date": first_seen.loc[mask].dt.strftime("%Y-%m-%d"),
                    "cohort": cohort,
                }
            )
        )
    true_new = pd.concat(cohort_rows, ignore_index=True).sort_values(
        ["cohort", "user_id"], ignore_index=True
    )
    august_new_ids = first_seen.index[cohort_masks["first_seen_2014_08_02_31"]]
    august_new_rows = frame.loc[
        holdout_mask & frame["user_id"].isin(august_new_ids)
    ]
    august_new_active_days = august_new_rows.groupby("user_id")["report_date"].nunique()
    counts: dict[str, float] = {
        "all_users": int(first_seen.size),
        "training_users_to_2014_08_01": len(training_users),
        "holdout_users_2014_08_02_31": len(holdout_users),
        "eligible_identity_holdout_users": len(eligible),
        "sampled_identity_holdout_users": sample_size,
        "users_per_fold": users_per_fold,
        "true_new_2013_09": int(cohort_masks["first_seen_2013_09"].sum()),
        "true_new_2014_08_02_31": int(cohort_masks["first_seen_2014_08_02_31"].sum()),
        "true_new_2014_08_active_at_least_2_days": int((august_new_active_days >= 2).sum()),
        "true_new_2014_08_purchase_share": float(
            august_new_rows[TARGETS[0]].sum() / holdout_totals[TARGETS[0]]
        ),
        "true_new_2014_08_redeem_share": float(
            august_new_rows[TARGETS[1]].sum() / holdout_totals[TARGETS[1]]
        ),
    }
    for row in fold_summary.itertuples(index=False):
        counts[f"fold_{int(row.fold)}_active_users"] = int(row.active_users)
    return first_seen_month, membership, fold_summary, true_new, counts


def detrended_log(series: pd.Series) -> pd.Series:
    logged = np.log1p(series.astype(float))
    local_level = logged.rolling(window=28, center=True, min_periods=14).mean()
    return (logged - local_level).dropna()


def build_periodicity_tables(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    weekday_rows = []
    autocorrelation_rows = []
    spectral_rows = []
    summary: dict[str, float] = {}

    for target in TARGETS:
        values = daily[target].astype(float)
        logged = np.log1p(values)
        weekday = daily.index.dayofweek
        overall_mean = values.mean()
        weekday_stats = values.groupby(weekday).agg(["mean", "median", "count"])
        design = pd.get_dummies(weekday, dtype=float).to_numpy()
        coefficients = np.linalg.lstsq(design, logged.to_numpy(), rcond=None)[0]
        fitted = design @ coefficients
        denominator = np.square(logged.to_numpy() - logged.mean()).sum()
        r_squared = 1.0 - np.square(logged.to_numpy() - fitted).sum() / denominator
        summary[f"{target}_weekday_r2_log"] = float(r_squared)
        for day_number, row in weekday_stats.iterrows():
            weekday_rows.append(
                {
                    "target": target,
                    "weekday_number": int(day_number),
                    "weekday": WEEKDAYS[int(day_number)],
                    "mean_amount": row["mean"],
                    "median_amount": row["median"],
                    "days": int(row["count"]),
                    "mean_index": row["mean"] / overall_mean,
                    "weekday_r2_log": r_squared,
                }
            )

        residual = detrended_log(values)
        for lag in range(1, 61):
            raw_corr = logged.autocorr(lag=lag)
            residual_corr = residual.autocorr(lag=lag)
            autocorrelation_rows.append(
                {
                    "target": target,
                    "lag_days": lag,
                    "raw_log_autocorrelation": raw_corr,
                    "detrended_log_autocorrelation": residual_corr,
                }
            )
        summary[f"{target}_lag7_detrended_acf"] = float(residual.autocorr(lag=7))

        centered = residual.to_numpy() - residual.mean()
        windowed = centered * np.hanning(len(centered))
        frequencies = np.fft.rfftfreq(len(windowed), d=1.0)
        power = np.square(np.abs(np.fft.rfft(windowed)))
        valid = frequencies > 0
        periods = 1.0 / frequencies[valid]
        powers = power[valid]
        spectrum = pd.DataFrame({"period_days": periods, "power": powers})
        spectrum = spectrum[spectrum["period_days"].between(2, 90)].nlargest(10, "power")
        spectrum.insert(0, "target", target)
        spectral_rows.append(spectrum)

    return (
        pd.DataFrame(weekday_rows),
        pd.DataFrame(autocorrelation_rows),
        pd.concat(spectral_rows, ignore_index=True),
        summary,
    )


def forecast_methods(training: pd.DataFrame, forecast_index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    methods: dict[str, pd.DataFrame] = {}
    recent_30 = training.tail(30)
    methods["mean_last_30d"] = pd.DataFrame(
        np.tile(recent_30[TARGETS].mean().to_numpy(), (len(forecast_index), 1)),
        index=forecast_index,
        columns=TARGETS,
    )

    recent_56 = training.tail(56).copy()
    recent_56["weekday"] = recent_56.index.dayofweek
    weekday_mean = recent_56.groupby("weekday")[TARGETS].mean()
    weekday_median = recent_56.groupby("weekday")[TARGETS].median()
    methods["weekday_mean_last_8w"] = pd.DataFrame(
        [weekday_mean.loc[date.dayofweek].to_numpy() for date in forecast_index],
        index=forecast_index,
        columns=TARGETS,
    )
    methods["weekday_median_last_8w"] = pd.DataFrame(
        [weekday_median.loc[date.dayofweek].to_numpy() for date in forecast_index],
        index=forecast_index,
        columns=TARGETS,
    )

    recursive = training[TARGETS].copy()
    forecast_rows = []
    for date in forecast_index:
        prior_week = date - pd.Timedelta(days=7)
        forecast_rows.append(recursive.loc[prior_week].to_numpy())
        recursive.loc[date] = forecast_rows[-1]
    methods["seasonal_naive_7d_recursive"] = pd.DataFrame(
        forecast_rows, index=forecast_index, columns=TARGETS
    )
    return methods


def build_backtests(
    daily: pd.DataFrame,
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    metric_rows = []
    for window_name, (start, end) in windows.items():
        actual = daily.loc[start:end, TARGETS]
        if len(actual) != 30:
            raise RuntimeError(f"{window_name} expected 30 days, found {len(actual)}")
        training = daily.loc[daily.index < start, TARGETS]
        for method, predicted in forecast_methods(training, actual.index).items():
            predicted = predicted.clip(lower=0.0)
            purchase_error = absolute_relative_error(
                actual[TARGETS[0]].to_numpy(), predicted[TARGETS[0]].to_numpy()
            )
            redeem_error = absolute_relative_error(
                actual[TARGETS[1]].to_numpy(), predicted[TARGETS[1]].to_numpy()
            )
            metrics = {
                "window": window_name,
                "method": method,
                "purchase_mean_relative_error": np.mean(purchase_error),
                "redeem_mean_relative_error": np.mean(redeem_error),
                "weighted_mean_relative_error": (
                    PURCHASE_WEIGHT * np.mean(purchase_error)
                    + REDEEM_WEIGHT * np.mean(redeem_error)
                ),
                "purchase_days_error_ge_0_3": int(np.sum(purchase_error >= 0.3)),
                "redeem_days_error_ge_0_3": int(np.sum(redeem_error >= 0.3)),
            }
            for power, label in [(1, "linear"), (2, "quadratic"), (3, "cubic")]:
                metrics[f"proxy_score_{label}"] = np.mean(
                    weighted_daily_score(
                        actual[TARGETS[0]].to_numpy(),
                        predicted[TARGETS[0]].to_numpy(),
                        actual[TARGETS[1]].to_numpy(),
                        predicted[TARGETS[1]].to_numpy(),
                        power=power,
                    )
                )
            metric_rows.append(metrics)

            for date in actual.index:
                prediction_rows.append(
                    {
                        "window": window_name,
                        "method": method,
                        "report_date": date.strftime("%Y-%m-%d"),
                        "actual_purchase": int(actual.loc[date, TARGETS[0]]),
                        "predicted_purchase": predicted.loc[date, TARGETS[0]],
                        "actual_redeem": int(actual.loc[date, TARGETS[1]]),
                        "predicted_redeem": predicted.loc[date, TARGETS[1]],
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def build_score_curves() -> pd.DataFrame:
    errors = np.round(np.arange(0.0, 0.351, 0.005), 3)
    return pd.DataFrame(
        {
            "relative_error": errors,
            "linear_score": score_from_error(errors, power=1),
            "quadratic_score": score_from_error(errors, power=2),
            "cubic_score": score_from_error(errors, power=3),
        }
    )


def configure_plots() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#293241",
            "axes.labelcolor": "#293241",
            "text.color": "#293241",
            "grid.color": "#d9dee7",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def save_plots(
    output: Path,
    daily: pd.DataFrame,
    weekday: pd.DataFrame,
    autocorrelation: pd.DataFrame,
    first_seen_month: pd.DataFrame,
    score_curves: pd.DataFrame,
) -> None:
    configure_plots()
    import matplotlib.pyplot as plt

    colors = {TARGETS[0]: "#2f6f9f", TARGETS[1]: "#d99032"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis, target in zip(axes, TARGETS):
        axis.plot(daily.index, daily[target] / 1e8, color=colors[target], linewidth=1.25)
        axis.axvspan(pd.Timestamp("2013-09-01"), pd.Timestamp("2013-09-30"), color="#d9dee7", alpha=0.55)
        axis.axvspan(pd.Timestamp("2014-08-02"), pd.Timestamp("2014-08-31"), color="#d9dee7", alpha=0.55)
        axis.set_title(f"Daily {TARGET_LABELS[target]} Total")
        axis.set_ylabel("Amount (100m cents)")
        axis.grid(axis="y", linewidth=0.7)
    axes[-1].set_xlabel("Date | shaded: 2013-09 and 2014-08 holdouts")
    fig.tight_layout()
    fig.savefig(output / "daily_totals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    for axis, target in zip(axes, TARGETS):
        subset = autocorrelation[autocorrelation["target"] == target]
        axis.bar(
            subset["lag_days"],
            subset["detrended_log_autocorrelation"],
            color=colors[target],
            width=0.8,
        )
        for lag in [7, 14, 28]:
            axis.axvline(lag, color="#293241", linestyle="--", linewidth=0.8)
        axis.axhline(0, color="#293241", linewidth=0.8)
        axis.set_title(f"Detrended Log Autocorrelation: {TARGET_LABELS[target]}")
        axis.set_ylabel("Correlation")
        axis.grid(axis="y", linewidth=0.7)
    axes[-1].set_xlabel("Lag (days) | dashed: 7, 14, 28")
    fig.tight_layout()
    fig.savefig(output / "periodicity_autocorrelation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(7)
    width = 0.36
    for offset, target in zip([-width / 2, width / 2], TARGETS):
        subset = weekday[weekday["target"] == target].sort_values("weekday_number")
        axis.bar(
            x + offset,
            subset["mean_index"],
            width,
            label=TARGET_LABELS[target],
            color=colors[target],
        )
    axis.axhline(1.0, color="#293241", linewidth=1.0)
    axis.set_xticks(x, WEEKDAYS)
    axis.set_ylim(bottom=0)
    axis.set_title("Weekday Mean Index")
    axis.set_ylabel("Mean / overall daily mean")
    axis.set_xlabel("Weekday | full history, 2013-07-01 to 2014-08-31")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output / "weekday_pattern.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        first_seen_month["first_seen_month"],
        first_seen_month["new_users"],
        color="#7b8f52",
    )
    axis.set_title("Users by First Observed Month")
    axis.set_ylabel("Users")
    axis.set_xlabel("First observed month | identity first appearance, not signup time")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output / "first_seen_user_counts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(score_curves["relative_error"], score_curves["linear_score"], label="Linear", color="#2f6f9f")
    axis.plot(score_curves["relative_error"], score_curves["quadratic_score"], label="Quadratic", color="#d99032")
    axis.plot(score_curves["relative_error"], score_curves["cubic_score"], label="Cubic", color="#c05a73")
    axis.axvline(0.30, color="#293241", linestyle="--", linewidth=1.0)
    axis.set_xlim(0, 0.35)
    axis.set_ylim(0, 10.5)
    axis.set_title("Local Error-to-Score Proxies")
    axis.set_ylabel("Daily score (0-10)")
    axis.set_xlabel("Absolute relative error | score is zero at error >= 0.30")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="both", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output / "score_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_report(
    quality: pd.DataFrame,
    periodicity_summary: dict[str, float],
    spectral: pd.DataFrame,
    user_counts: dict[str, int],
    backtest_metrics: pd.DataFrame,
) -> str:
    quality_lookup = quality.set_index("check")["value"].to_dict()
    best_rows = (
        backtest_metrics.sort_values(["window", "proxy_score_linear"], ascending=[True, False])
        .groupby("window", as_index=False)
        .head(1)
    )
    best_lines = []
    for row in best_rows.itertuples(index=False):
        best_lines.append(
            f"- `{row.window}`: `{row.method}`, linear={row.proxy_score_linear:.3f}, "
            f"quadratic={row.proxy_score_quadratic:.3f}, cubic={row.proxy_score_cubic:.3f}."
        )
    peak_lines = []
    for target in TARGETS:
        peaks = spectral[spectral["target"] == target].head(3)
        periods = ", ".join(f"{value:.2f}d" for value in peaks["period_days"])
        peak_lines.append(f"- {TARGET_LABELS[target]} top detrended spectral periods: {periods}.")

    purchase_lag7 = periodicity_summary[f"{TARGETS[0]}_lag7_detrended_acf"]
    redeem_lag7 = periodicity_summary[f"{TARGETS[1]}_lag7_detrended_acf"]
    purchase_weekday_r2 = periodicity_summary[f"{TARGETS[0]}_weekday_r2_log"]
    redeem_weekday_r2 = periodicity_summary[f"{TARGETS[1]}_weekday_r2_log"]
    recent_weekday = backtest_metrics[
        backtest_metrics["window"].str.startswith("rolling_2014_")
        & (backtest_metrics["method"] == "weekday_median_last_8w")
    ]
    august_weekday_error = backtest_metrics.loc[
        (backtest_metrics["window"] == "holdout_2014_08")
        & (backtest_metrics["method"] == "weekday_median_last_8w"),
        "weighted_mean_relative_error",
    ].iloc[0]
    return f"""# Data Feature Test Report

## tl;dr

- The source is structurally suitable for these tests: {int(quality_lookup['rows']):,} rows, {int(quality_lookup['unique_users']):,} users, and a complete {int(quality_lookup['distinct_dates'])}-day calendar from {quality_lookup['date_min']} through {quality_lookup['date_max']}. There are no duplicate user-date rows or negative target rows in the loaded target columns.
- Weekly structure should be treated as a real candidate feature, but not as proof of stable annual seasonality. Detrended lag-7 autocorrelation is {purchase_lag7:.3f} for purchase and {redeem_lag7:.3f} for redeem; weekday indicators explain {purchase_weekday_r2:.1%} and {redeem_weekday_r2:.1%} of log-scale daily variance.
- Use both requested windows, but label 2013-09 as a stress test: it has only 62 prior days, while the 2014-08 holdout reflects a much more mature series. For model selection, add rolling 30-day origins in 2014 rather than relying on only these two windows.
- The requested random 4,000-user split is valid as an identity-holdout stress test, not as a true-new-user test. It samples only from the {int(user_counts['eligible_identity_holdout_users']):,} users observed by 2014-08-01 and creates two disjoint folds of {int(user_counts['users_per_fold']):,} users without using future activity for selection.
- Only {user_counts['true_new_2014_08_02_31']:,} users first appear during 2014-08-02 to 2014-08-31, so a 4,000-user true cold-start cohort cannot be constructed from that holdout. Keep this temporal cohort as a complementary test.
- For the weekday-median diagnostic, the median weighted relative error across 2014-03..07 is {recent_weekday['weighted_mean_relative_error'].median():.3f}; the 2014-08 holdout is {august_weekday_error:.3f}. This is why August should not be the only model-selection window.

## Periodicity Evidence

- Lag-7 detrended log autocorrelation: purchase={purchase_lag7:.3f}, redeem={redeem_lag7:.3f}.
- Weekday log-scale R-squared: purchase={purchase_weekday_r2:.3f}, redeem={redeem_weekday_r2:.3f}.
{chr(10).join(peak_lines)}

The 427-day history contains only a little more than one annual cycle. Month-of-year or September-specific annual seasonality is therefore not identifiable with confidence. The 2013-09 result is useful for robustness, but its shorter training history and different growth stage make it non-exchangeable with 2014-08.

## Backtest Snapshot

Best of the included transparent baselines under the linear proxy:

{chr(10).join(best_lines)}

See `backtest_metrics.csv` for every baseline and all three score curves. These baselines are diagnostics, not proposed final models.

## Cold-start Design

`cold_start_user_folds.csv` is generated by sorting SHA-256 hashes of `coldstart-v1-seed-20260824|user_id`. Eligibility requires only one row on or before 2014-08-01; validation-period activity is never used for sampling. Fold 1 has {int(user_counts['fold_1_active_users']):,} active users in the August holdout and Fold 2 has {int(user_counts['fold_2_active_users']):,}.

For a user-level model, hide the selected fold's full user history, profile, ID encodings, and user-derived statistics, then measure its holdout contribution separately. Keep the macro daily aggregate history because it is available to the competition forecaster. For a pure daily aggregate model that never consumes user identity, these random user folds do not test anything meaningful; use the temporal first-seen cohort and aggregate time backtests instead.

`true_new_user_cohorts.csv` contains users whose first observed record falls inside 2013-09 or 2014-08-02..31. This is the closer cold-start simulation. In the August cohort, {int(user_counts['true_new_2014_08_active_at_least_2_days']):,} users are active on at least two days; the cohort contributes {user_counts['true_new_2014_08_purchase_share']:.2%} of purchase and {user_counts['true_new_2014_08_redeem_share']:.2%} of redeem. First observed record is not necessarily account signup, so conclusions should say "first observed" rather than "newly registered". A 2,000-user scenario can be reported by reweighting this cohort, but it must be labeled a simulation rather than a validation score.

## Local Score Proxies

Let `e = abs(predicted - actual) / abs(actual)` and `x = min(e / 0.30, 1)`.

- Linear: `10 * (1 - x)`.
- Quadratic: `10 * (1 - x)^2`.
- Cubic: `10 * (1 - x)^3`.
- Daily purchase/redeem combination: `0.45 * purchase_score + 0.55 * redeem_score`.
- Holdout score: mean daily combined score.

All three proxies return 10 at zero error and 0 at error greater than or equal to 0.30. Quadratic and cubic curves are intentionally harsher for every non-zero sub-threshold error. Because official intermediate mapping and zero-target handling are unpublished, these are sensitivity measures rather than leaderboard replicas. Locally, actual=0/predicted=0 maps to zero error; actual=0/non-zero prediction maps to infinite error and zero score.

## Files

- Tables: data quality, daily aggregate, weekday effects, autocorrelation, spectral peaks, user cohorts, random folds, backtest predictions/metrics, and score curves.
- Figures: daily totals, periodic autocorrelation, weekday pattern, first-seen users, and score curves.
- Reproduction: `scripts/run_feature_tests.py` and `data/derived/d001_feature_tests/feature_tests.ipynb`.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_hash(project_root: Path, source: Path) -> str:
    manifest = project_root / "data/raw/official/manifest.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if name.lstrip(" *") == source.name:
            return digest
    return sha256(source)


def write_manifest(
    project_root: Path,
    source: Path,
    output: Path,
    seed: int,
    sample_size: int,
    folds: int,
    definition_paths: list[Path],
) -> None:
    import matplotlib

    generated_files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "feature_tests.ipynb"}:
            generated_files.append(
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    manifest = {
        "dataset_version": output.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source.relative_to(project_root)).replace("\\", "/"),
        "source_sha256": read_source_hash(project_root, source),
        "generator": "scripts/run_feature_tests.py",
        "generator_sha256": sha256(project_root / "scripts/run_feature_tests.py"),
        "scoring_module_sha256": sha256(project_root / "scripts/local_score.py"),
        "command": (
            "python scripts/run_feature_tests.py "
            f"--output data/derived/{output.name} --seed {seed} "
            f"--sample-size {sample_size} --folds {folds} --overwrite"
        ),
        "parameters": {"seed": seed, "sample_size": sample_size, "folds": folds},
        "validation_definitions": [
            {
                "path": str(path.relative_to(project_root)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for path in definition_paths
        ],
        "environment": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "generated_files": generated_files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source = resolve_path(args.source, project_root).resolve()
    output = resolve_path(args.output, project_root).resolve()
    if project_root not in output.parents:
        raise ValueError("output must stay inside the project")
    prepare_output(output, args.overwrite)

    cold_split, backtest_windows, definition_paths = load_validation_definitions(
        project_root
    )
    expected_sampling = cold_split["sampling"]
    requested_sampling = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "folds": args.folds,
    }
    for key, value in requested_sampling.items():
        if value != expected_sampling[key]:
            raise ValueError(
                f"CLI {key}={value} does not match cold split definition "
                f"{expected_sampling[key]}"
            )
    frame = load_balance(source)
    daily = build_daily(frame)
    quality = build_quality_summary(frame, daily)
    first_seen_month, membership, fold_summary, true_new, user_counts = build_user_diagnostics(
        frame, args.seed, args.sample_size, args.folds, cold_split
    )
    weekday, autocorrelation, spectral, periodicity_summary = build_periodicity_tables(daily)
    backtest_predictions, backtest_metrics = build_backtests(daily, backtest_windows)
    score_curves = build_score_curves()

    daily.reset_index().to_csv(output / "daily_aggregate.csv", index=False)
    quality.to_csv(output / "data_quality_summary.csv", index=False)
    first_seen_month.to_csv(output / "monthly_first_seen_users.csv", index=False)
    membership.to_csv(output / "cold_start_user_folds.csv", index=False)
    fold_summary.to_csv(output / "cold_start_fold_summary.csv", index=False)
    true_new.to_csv(output / "true_new_user_cohorts.csv", index=False)
    weekday.to_csv(output / "weekday_effects.csv", index=False)
    autocorrelation.to_csv(output / "autocorrelation.csv", index=False)
    spectral.to_csv(output / "spectral_peaks.csv", index=False)
    backtest_predictions.to_csv(output / "backtest_predictions.csv", index=False)
    backtest_metrics.to_csv(output / "backtest_metrics.csv", index=False)
    score_curves.to_csv(output / "score_curves.csv", index=False)

    save_plots(output, daily, weekday, autocorrelation, first_seen_month, score_curves)
    report = build_report(
        quality, periodicity_summary, spectral, user_counts, backtest_metrics
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "README.md").write_text(
        "# d001 feature tests\n\n"
        "Generated, reproducible diagnostics for periodicity, cold-start splits, "
        "and local score proxies. Start with `REPORT.md`; rerun with the command "
        "recorded in `manifest.json`. The script needs Python with pandas, NumPy, "
        "and Matplotlib; the Notebook additionally needs nbformat, nbclient, and "
        "IPython. Exact local versions are recorded in the manifest. Official raw "
        "CSV files are only read.\n",
        encoding="utf-8",
    )
    write_manifest(
        project_root,
        source,
        output,
        args.seed,
        args.sample_size,
        args.folds,
        definition_paths,
    )
    print(output)


if __name__ == "__main__":
    main()
