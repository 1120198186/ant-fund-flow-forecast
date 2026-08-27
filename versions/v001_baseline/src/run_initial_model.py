#!/usr/bin/env python3
"""Run the first leakage-controlled component model and build a submission."""

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
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PURCHASE_WEIGHT = 0.45
REDEEM_WEIGHT = 0.55
ZERO_SCORE_THRESHOLD = 0.30
COMPONENTS = ["direct_inflow_amt", "transfer_outflow_amt", "consume_outflow_amt"]
AUXILIARY_COLUMNS = [
    "observed_users", "opening_balance_amt", "closing_balance_amt",
    "direct_inflow_users", "total_outflow_users", "transfer_outflow_users",
    "mfd_7daily_yield", "Interest_1_W", "Interest_1_M", "Interest_3_M",
]
LAGS = [1, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]


@dataclass(frozen=True)
class Window:
    window_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    role: str


@dataclass
class FittedSeriesModel:
    feature_names: list[str]
    scaler: StandardScaler
    ridge: Ridge
    residual_lower: float
    residual_upper: float
    training_rows: int


def parse_args() -> argparse.Namespace:
    version_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=version_root / "configs/initial_component_ensemble.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_windows(root: Path, config: dict[str, Any]) -> list[Window]:
    rolling = json.loads((root / "validation/splits/rolling_30d_2014_03_08_v1.json").read_text(encoding="utf-8"))
    stress = json.loads((root / "validation/splits/backtest_2013_09_v1.json").read_text(encoding="utf-8"))
    tuning = set(config["tuning_windows"])
    windows = [
        Window(row["id"], pd.Timestamp(row["start"]), pd.Timestamp(row["end"]), "tuning" if row["id"] in tuning else "quasi_holdout")
        for row in rolling["windows"]
    ]
    windows.append(Window(config["stress_window"], pd.Timestamp(stress["holdout_window"]["start"]), pd.Timestamp(stress["holdout_window"]["end"]), "stress"))
    return windows


def load_daily(root: Path) -> pd.DataFrame:
    path = root / "data/derived/d006_daily_flow_feature_table_v1/daily_flow_rate_features.parquet"
    frame = pd.read_parquet(path)
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    return frame.sort_values("report_date").set_index("report_date")


def weighted_recent(values: np.ndarray, count: int, decay: float = 0.82) -> float:
    subset = values[-count:]
    weights = decay ** np.arange(len(subset) - 1, -1, -1)
    return float(np.average(subset, weights=weights))


def seasonal_anchor(values: np.ndarray, dates: pd.DatetimeIndex, origin_idx: int, target_date: pd.Timestamp) -> float:
    history = values[: origin_idx + 1].astype(float)
    history_dates = dates[: origin_idx + 1]
    same_weekday = history[history_dates.weekday == target_date.weekday()]
    recent28 = history[-28:]
    same4 = float(np.mean(same_weekday[-4:]))
    same8 = weighted_recent(same_weekday, min(8, len(same_weekday)))
    level28 = float(np.mean(recent28))
    anchor = 0.55 * same4 + 0.30 * same8 + 0.15 * level28
    return max(anchor, max(1.0, 0.05 * level28))


def normalized_slope(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    y = np.log1p(np.clip(values.astype(float), 0, None))
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])


def feature_row(frame: pd.DataFrame, target_column: str, origin_idx: int, target_date: pd.Timestamp, horizon: int) -> tuple[np.ndarray, list[str], float]:
    values = frame[target_column].to_numpy(dtype=float)
    dates = frame.index
    anchor = seasonal_anchor(values, dates, origin_idx, target_date)
    row: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        row.append(float(value))

    add("horizon", horizon / 30.0)
    add("horizon_squared", (horizon / 30.0) ** 2)
    add("day_of_month", target_date.day / 31.0)
    add("days_to_month_end", (target_date.days_in_month - target_date.day) / 31.0)
    add("is_weekend", float(target_date.weekday() >= 5))
    add("is_month_start", float(target_date.day <= 3))
    add("is_month_end", float(target_date.day >= target_date.days_in_month - 2))
    add("weekday_sin", math.sin(2 * math.pi * target_date.weekday() / 7))
    add("weekday_cos", math.cos(2 * math.pi * target_date.weekday() / 7))
    for weekday in range(7):
        add(f"weekday_{weekday}", float(target_date.weekday() == weekday))
    add("log_anchor", math.log1p(anchor))

    history = values[: origin_idx + 1]
    for lag in LAGS:
        position = origin_idx - lag + 1
        value = values[position] if position >= 0 else anchor
        add(f"lag_{lag}_log", math.log1p(max(0.0, float(value))))
        add(f"lag_{lag}_ratio_anchor", math.log((max(0.0, float(value)) + 1) / (anchor + 1)))
    for window in ROLLING_WINDOWS:
        subset = history[-window:]
        mean = float(np.mean(subset))
        median = float(np.median(subset))
        add(f"rolling_mean_{window}_log", math.log1p(max(0.0, mean)))
        add(f"rolling_median_{window}_log", math.log1p(max(0.0, median)))
        add(f"rolling_cv_{window}", float(np.std(subset)) / (abs(mean) + 1.0))
    add("trend_14", normalized_slope(history[-14:]))
    add("trend_28", normalized_slope(history[-28:]))

    for column in AUXILIARY_COLUMNS:
        aux = frame[column].to_numpy(dtype=float)[: origin_idx + 1]
        current = float(aux[-1])
        mean7 = float(np.mean(aux[-7:]))
        mean28 = float(np.mean(aux[-28:]))
        if column.startswith("Interest_") or column == "mfd_7daily_yield":
            add(f"{column}_last", current)
            add(f"{column}_change7", current - float(aux[max(0, len(aux) - 8)]))
            add(f"{column}_change28", current - float(aux[max(0, len(aux) - 29)]))
        else:
            add(f"{column}_last_log", math.log1p(max(0.0, current)))
            add(f"{column}_mean7_log", math.log1p(max(0.0, mean7)))
            add(f"{column}_mean7_to_28", math.log((max(0.0, mean7) + 1) / (max(0.0, mean28) + 1)))
    return np.asarray(row, dtype=float), names, anchor


def build_training_matrix(frame: pd.DataFrame, target_column: str, horizon: int, minimum_history: int, half_life: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    values = frame[target_column].to_numpy(dtype=float)
    dates = frame.index
    last_date = dates[-1]
    rows: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    names: list[str] | None = None
    for origin_idx in range(minimum_history - 1, len(frame) - 1):
        for step in range(1, min(horizon, len(frame) - 1 - origin_idx) + 1):
            target_idx = origin_idx + step
            features, feature_names, anchor = feature_row(frame, target_column, origin_idx, dates[target_idx], step)
            rows.append(features)
            labels.append(float(np.log1p(values[target_idx]) - np.log1p(anchor)))
            age = max(0, (last_date - dates[target_idx]).days)
            weights.append(0.5 ** (age / half_life))
            names = feature_names
    if not rows or names is None:
        raise RuntimeError(f"insufficient training rows for {target_column}")
    return np.vstack(rows), np.asarray(labels), np.asarray(weights), names


def fit_series_model(frame: pd.DataFrame, target_column: str, config: dict[str, Any]) -> FittedSeriesModel:
    x, y, weights, names = build_training_matrix(frame, target_column, int(config["forecast_horizon"]), int(config["minimum_history_days"]), float(config["training_half_life_days"]))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = Ridge(alpha=float(config["ridge_alpha"]))
    model.fit(x_scaled, y, sample_weight=weights)
    return FittedSeriesModel(names, scaler, model, max(-0.70, float(np.quantile(y, 0.01))), min(0.70, float(np.quantile(y, 0.99))), len(y))


def predict_series(frame: pd.DataFrame, target_column: str, model: FittedSeriesModel, forecast_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    origin_idx = len(frame) - 1
    rows: list[np.ndarray] = []
    anchors: list[float] = []
    for step, target_date in enumerate(forecast_dates, start=1):
        features, names, anchor = feature_row(frame, target_column, origin_idx, target_date, step)
        if names != model.feature_names:
            raise RuntimeError("feature schema changed between fit and prediction")
        rows.append(features)
        anchors.append(anchor)
    x = np.vstack(rows)
    residual = np.clip(model.ridge.predict(model.scaler.transform(x)), model.residual_lower, model.residual_upper)
    anchor_values = np.asarray(anchors)
    predicted = np.expm1(np.log1p(anchor_values) + residual)
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise RuntimeError(f"invalid prediction for {target_column}")
    return anchor_values, predicted


def forecast_profit_share(history: pd.DataFrame, direct: np.ndarray, transfer: np.ndarray, consume: np.ndarray) -> np.ndarray:
    ratios = history["profit_share_per_10000_opening_balance"].replace([np.inf, -np.inf], np.nan).dropna()
    ratio = float(np.clip(ratios.tail(14).median(), ratios.tail(56).quantile(0.10), ratios.tail(56).quantile(0.90)))
    opening = float(history["closing_balance_amt"].iloc[-1])
    result = np.zeros(len(direct), dtype=float)
    for index in range(len(result)):
        result[index] = opening * ratio / 10000.0
        opening = opening + direct[index] + result[index] - transfer[index] - consume[index]
        if opening < 0:
            raise RuntimeError("predicted balance became negative")
    return result


def predict_window(daily: pd.DataFrame, cutoff: pd.Timestamp, forecast_dates: pd.DatetimeIndex, config: dict[str, Any], fit_components: bool = True) -> tuple[pd.DataFrame, dict[str, FittedSeriesModel]]:
    history = daily.loc[daily.index < cutoff].copy()
    output = pd.DataFrame(index=forecast_dates)
    output.index.name = "report_date"
    for total_column, name in [("total_inflow_amt", "purchase"), ("total_outflow_amt", "redeem")]:
        values = history[total_column].to_numpy(dtype=float)
        output[f"seasonal_{name}"] = [seasonal_anchor(values, history.index, len(history) - 1, date) for date in forecast_dates]
    if not fit_components:
        output["component_purchase"] = output["seasonal_purchase"]
        output["component_redeem"] = output["seasonal_redeem"]
        return output, {}

    models: dict[str, FittedSeriesModel] = {}
    predictions: dict[str, np.ndarray] = {}
    for column in COMPONENTS:
        model = fit_series_model(history, column, config)
        _, prediction = predict_series(history, column, model, forecast_dates)
        models[column] = model
        predictions[column] = prediction
    direct = predictions["direct_inflow_amt"]
    transfer = predictions["transfer_outflow_amt"]
    consume = predictions["consume_outflow_amt"]
    share = forecast_profit_share(history, direct, transfer, consume)
    output["ridge_direct"] = direct
    output["rule_share"] = share
    output["ridge_transfer"] = transfer
    output["ridge_consume"] = consume
    output["component_purchase"] = direct + share
    output["component_redeem"] = transfer + consume
    return output, models


def absolute_relative_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)) / np.maximum(np.abs(actual), 1.0)


def score_from_error(error: np.ndarray, power: int) -> np.ndarray:
    return 10.0 * np.power(np.clip(1.0 - error / ZERO_SCORE_THRESHOLD, 0.0, 1.0), power)


def weighted_daily_score(actual_purchase: np.ndarray, predicted_purchase: np.ndarray, actual_redeem: np.ndarray, predicted_redeem: np.ndarray, power: int) -> np.ndarray:
    purchase = score_from_error(absolute_relative_error(actual_purchase, predicted_purchase), power)
    redeem = score_from_error(absolute_relative_error(actual_redeem, predicted_redeem), power)
    return PURCHASE_WEIGHT * purchase + REDEEM_WEIGHT * redeem


def blend(frame: pd.DataFrame, target: str, seasonal_weight: float) -> np.ndarray:
    return seasonal_weight * frame[f"seasonal_{target}"].to_numpy(dtype=float) + (1.0 - seasonal_weight) * frame[f"component_{target}"].to_numpy(dtype=float)


def selection_statistics(fold_predictions: dict[str, pd.DataFrame], tuning_windows: list[str], purchase_weight: float, redeem_weight: float) -> dict[str, float | int]:
    cubic: list[float] = []
    baseline_cubic: list[float] = []
    weighted_error: list[float] = []
    zero_days = 0
    baseline_zero_days = 0
    for window_id in tuning_windows:
        frame = fold_predictions[window_id]
        pp = blend(frame, "purchase", purchase_weight)
        pr = blend(frame, "redeem", redeem_weight)
        ap = frame["actual_purchase"].to_numpy(dtype=float)
        ar = frame["actual_redeem"].to_numpy(dtype=float)
        pe = absolute_relative_error(ap, pp)
        re = absolute_relative_error(ar, pr)
        cubic.append(float(weighted_daily_score(ap, pp, ar, pr, 3).mean()))
        baseline_cubic.append(float(weighted_daily_score(ap, frame["seasonal_purchase"].to_numpy(dtype=float), ar, frame["seasonal_redeem"].to_numpy(dtype=float), 3).mean()))
        weighted_error.append(float(PURCHASE_WEIGHT * pe.mean() + REDEEM_WEIGHT * re.mean()))
        zero_days += int((pe >= ZERO_SCORE_THRESHOLD).sum() + (re >= ZERO_SCORE_THRESHOLD).sum())
        baseline_zero_days += int(
            (absolute_relative_error(ap, frame["seasonal_purchase"].to_numpy(dtype=float)) >= ZERO_SCORE_THRESHOLD).sum()
            + (absolute_relative_error(ar, frame["seasonal_redeem"].to_numpy(dtype=float)) >= ZERO_SCORE_THRESHOLD).sum()
        )
    better_folds = int(sum(candidate > baseline for candidate, baseline in zip(cubic, baseline_cubic)))
    median_improvement = float(np.median(cubic) / max(np.median(baseline_cubic), 1e-12) - 1.0)
    worst_fold_change = float(min(candidate / max(baseline, 1e-12) - 1.0 for candidate, baseline in zip(cubic, baseline_cubic)))
    passes_gate = better_folds >= 4 and median_improvement >= 0.01 and worst_fold_change >= -0.02 and zero_days <= baseline_zero_days
    return {
        "median_tuning_cubic_score": float(np.median(cubic)),
        "worst_tuning_cubic_score": float(np.min(cubic)),
        "mean_tuning_cubic_score": float(np.mean(cubic)),
        "mean_tuning_weighted_mape": float(np.mean(weighted_error)),
        "over_30pct_target_days": zero_days,
        "baseline_over_30pct_target_days": baseline_zero_days,
        "better_folds": better_folds,
        "median_relative_improvement": median_improvement,
        "worst_fold_relative_change": worst_fold_change,
        "passes_entry_gate": passes_gate,
    }


def select_weights(fold_predictions: dict[str, pd.DataFrame], tuning_windows: list[str], step: float) -> tuple[float, float, pd.DataFrame]:
    values = np.arange(0.0, 1.0 + step / 2, step)
    rows = []
    for purchase_weight in values:
        for redeem_weight in values:
            row = {"purchase_seasonal_weight": float(purchase_weight), "redeem_seasonal_weight": float(redeem_weight)}
            row.update(selection_statistics(fold_predictions, tuning_windows, float(purchase_weight), float(redeem_weight)))
            rows.append(row)
    search = pd.DataFrame(rows).sort_values(
        ["passes_entry_gate", "median_tuning_cubic_score", "worst_tuning_cubic_score", "mean_tuning_weighted_mape", "over_30pct_target_days"],
        ascending=[False, False, False, True, True], ignore_index=True,
    )
    eligible = search[search["passes_entry_gate"]]
    if len(eligible):
        winner = eligible.iloc[0]
    else:
        winner = search[(search["purchase_seasonal_weight"] == 1.0) & (search["redeem_seasonal_weight"] == 1.0)].iloc[0]
    return float(winner["purchase_seasonal_weight"]), float(winner["redeem_seasonal_weight"]), search


def add_selected(frame: pd.DataFrame, purchase_weight: float, redeem_weight: float) -> pd.DataFrame:
    output = frame.copy()
    output["selected_purchase"] = blend(output, "purchase", purchase_weight)
    output["selected_redeem"] = blend(output, "redeem", redeem_weight)
    return output


def metric_row(window: Window, model_name: str, actual_purchase: np.ndarray, predicted_purchase: np.ndarray, actual_redeem: np.ndarray, predicted_redeem: np.ndarray) -> dict[str, float | int | str]:
    purchase_error = absolute_relative_error(actual_purchase, predicted_purchase)
    redeem_error = absolute_relative_error(actual_redeem, predicted_redeem)
    return {
        "window_id": window.window_id, "role": window.role, "model": model_name, "days": len(actual_purchase),
        "purchase_mape": float(purchase_error.mean()), "redeem_mape": float(redeem_error.mean()),
        "weighted_mape": float(PURCHASE_WEIGHT * purchase_error.mean() + REDEEM_WEIGHT * redeem_error.mean()),
        "purchase_over_30pct_days": int((purchase_error >= ZERO_SCORE_THRESHOLD).sum()),
        "redeem_over_30pct_days": int((redeem_error >= ZERO_SCORE_THRESHOLD).sum()),
        "linear_score": float(weighted_daily_score(actual_purchase, predicted_purchase, actual_redeem, predicted_redeem, 1).mean()),
        "quadratic_score": float(weighted_daily_score(actual_purchase, predicted_purchase, actual_redeem, predicted_redeem, 2).mean()),
        "cubic_score": float(weighted_daily_score(actual_purchase, predicted_purchase, actual_redeem, predicted_redeem, 3).mean()),
    }


def contamination_test(daily: pd.DataFrame, cutoff: pd.Timestamp, forecast_dates: pd.DatetimeIndex, config: dict[str, Any]) -> dict[str, Any]:
    original, _ = predict_window(daily, cutoff, forecast_dates, config)
    contaminated = daily.copy()
    mask = contaminated.index >= cutoff
    numeric = contaminated.select_dtypes(include=[np.number]).columns
    rng = np.random.default_rng(int(config["seed"]))
    for column in numeric:
        values = contaminated[column].to_numpy(dtype=float, copy=True)
        values[mask] = rng.uniform(1e12, 1e15, size=int(mask.sum()))
        contaminated[column] = values
    changed, _ = predict_window(contaminated, cutoff, forecast_dates, config)
    columns = ["seasonal_purchase", "seasonal_redeem", "component_purchase", "component_redeem"]
    max_diff = float(np.max(np.abs(original[columns].to_numpy() - changed[columns].to_numpy())))
    return {"cutoff": cutoff.date().isoformat(), "columns_checked": columns, "max_absolute_difference": max_diff, "passed": bool(max_diff == 0.0)}


def chinese_report(metrics: pd.DataFrame, selected: dict[str, Any], final_prediction: pd.DataFrame, contamination: dict[str, Any]) -> str:
    tuning = metrics[(metrics["role"] == "tuning") & (metrics["model"] == "selected_ensemble")]
    base = metrics[(metrics["role"] == "tuning") & (metrics["model"] == "seasonal_baseline")]
    locked = metrics[metrics["role"] == "quasi_holdout"]
    stress = metrics[metrics["role"] == "stress"]
    better = int(sum(tuning.set_index("window_id")["cubic_score"] > base.set_index("window_id")["cubic_score"]))
    delta = float(tuning["cubic_score"].median() - base["cubic_score"].median())
    lines = [
        "# 首榜初级结果报告", "", "## 结论", "",
        "首版采用总量季节锚点作为保险项，主动申购、转账赎回和消费分别使用非递归30步Ridge预测，收益转入按历史万份收益和预测余额递推；最终只在总申购、总赎回层各选择一个融合权重。", "",
        f"- 总申购季节权重：{selected['purchase_seasonal_weight']:.2f}；总赎回季节权重：{selected['redeem_seasonal_weight']:.2f}。",
        f"- 3–7月有{better}/5折优于季节基线，三次评分中位数变化{delta:+.4f}。",
        f"- 9月预测日均总申购{final_prediction['selected_purchase'].mean():,.0f}分，日均总赎回{final_prediction['selected_redeem'].mean():,.0f}分。",
        f"- 截止日后哨兵污染测试：{'通过' if contamination['passed'] else '失败'}，最大预测差异{contamination['max_absolute_difference']:.6f}。", "",
        "## 8月准留出审计", "",
        "8月此前已用于行为分层研究，因此这里不称完全独立留出集。本版本在3–7月选定配置后只读取一次8月结果，不再据此修改权重。", "",
        "| 模型 | 申购平均相对误差 | 赎回平均相对误差 | 线性评分 | 二次评分 | 三次评分 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in locked.itertuples(index=False):
        label = "季节基线" if row.model == "seasonal_baseline" else "选定融合"
        lines.append(f"| {label} | {row.purchase_mape:.4f} | {row.redeem_mape:.4f} | {row.linear_score:.4f} | {row.quadratic_score:.4f} | {row.cubic_score:.4f} |")
    lines += ["", "## 3–7月滚动评估", "", "| 窗口 | 模型 | 加权平均相对误差 | 线性评分 | 二次评分 | 三次评分 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in metrics[metrics["role"] == "tuning"].itertuples(index=False):
        label = "季节基线" if row.model == "seasonal_baseline" else "选定融合"
        lines.append(f"| {row.window_id} | {label} | {row.weighted_mape:.4f} | {row.linear_score:.4f} | {row.quadratic_score:.4f} | {row.cubic_score:.4f} |")
    lines += ["", "## 2013年9月压力测试", "", "该窗口只有62天历史，只运行季节基线并观察短历史是否失控，不参与模型选择。", "", "| 加权平均相对误差 | 线性评分 | 二次评分 | 三次评分 |", "| ---: | ---: | ---: | ---: |"]
    for row in stress.itertuples(index=False):
        lines.append(f"| {row.weighted_mape:.4f} | {row.linear_score:.4f} | {row.quadratic_score:.4f} | {row.cubic_score:.4f} |")
    lines += ["", "## 边界", "", "当前评分为本地代理：日相对误差达到或超过0.30记零分，申购和赎回按45%/55%加权，再分别使用线性、二次、三次衰减。官方未公布中间映射，本地分数不能等同于线上成绩。用户分层本版只用于解释和后续挑战模型，未直接进入训练，避免把截止日名单回填历史造成泄漏。"]
    return "\n".join(lines) + "\n"


def prepare_output_dirs(version_root: Path, run_id: str) -> dict[str, Path]:
    folders = {"models": version_root / "artifacts/models" / run_id, "metrics": version_root / "artifacts/metrics" / run_id, "predictions": version_root / "artifacts/predictions" / run_id, "submissions": version_root / "artifacts/submissions" / run_id}
    existing = [path for path in folders.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing run: {existing[0]}")
    for path in folders.values():
        path.mkdir(parents=True)
    return folders


def make_read_only(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_file():
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def main() -> None:
    args = parse_args()
    version_root = Path(__file__).resolve().parents[1]
    root = version_root.parents[1]
    config = load_config(args.config)
    outputs = prepare_output_dirs(version_root, str(config["run_id"]))
    windows = load_windows(root, config)
    daily = load_daily(root)

    fold_predictions: dict[str, pd.DataFrame] = {}
    for window in windows:
        dates = pd.date_range(window.start, window.end, freq="D")
        predictions, _ = predict_window(daily, window.start, dates, config, fit_components=window.role != "stress")
        actual = daily.loc[dates]
        predictions["actual_purchase"] = actual["total_inflow_amt"].to_numpy(dtype=float)
        predictions["actual_redeem"] = actual["total_outflow_amt"].to_numpy(dtype=float)
        fold_predictions[window.window_id] = predictions
        predictions.reset_index().to_csv(outputs["predictions"] / f"{window.window_id}_候选预测.csv", index=False, encoding="utf-8-sig")
        print(f"完成 {window.window_id}")

    purchase_weight, redeem_weight, search = select_weights(fold_predictions, list(config["tuning_windows"]), float(config["ensemble_weight_step"]))
    search.to_csv(outputs["metrics"] / "融合权重搜索.csv", index=False, encoding="utf-8-sig")

    metric_rows: list[dict[str, float | int | str]] = []
    for window in windows:
        frame = fold_predictions[window.window_id]
        if window.role != "stress":
            frame = add_selected(frame, purchase_weight, redeem_weight)
        else:
            frame["selected_purchase"] = frame["seasonal_purchase"]
            frame["selected_redeem"] = frame["seasonal_redeem"]
        actual_purchase = frame["actual_purchase"].to_numpy(dtype=float)
        actual_redeem = frame["actual_redeem"].to_numpy(dtype=float)
        metric_rows.append(metric_row(window, "seasonal_baseline", actual_purchase, frame["seasonal_purchase"].to_numpy(), actual_redeem, frame["seasonal_redeem"].to_numpy()))
        if window.role != "stress":
            metric_rows.append(metric_row(window, "selected_ensemble", actual_purchase, frame["selected_purchase"].to_numpy(), actual_redeem, frame["selected_redeem"].to_numpy()))
        frame.reset_index().to_csv(outputs["predictions"] / f"{window.window_id}_选定预测.csv", index=False, encoding="utf-8-sig")

    final_cutoff = pd.Timestamp(config["final_cutoff"])
    final_dates = pd.date_range(final_cutoff, pd.Timestamp(config["final_end"]), freq="D")
    final_candidates, final_models = predict_window(daily, final_cutoff, final_dates, config)
    final_prediction = add_selected(final_candidates, purchase_weight, redeem_weight)
    final_prediction.reset_index().to_csv(outputs["predictions"] / "2014年9月分量预测.csv", index=False, encoding="utf-8-sig")
    for name, model in final_models.items():
        joblib.dump(model, outputs["models"] / f"{name}.joblib", compress=3)

    submission = pd.DataFrame({"report_date": final_dates.strftime("%Y%m%d"), "purchase": np.rint(final_prediction["selected_purchase"]).astype(np.int64), "redeem": np.rint(final_prediction["selected_redeem"]).astype(np.int64)})
    submission_path = outputs["submissions"] / "首榜初级提交.csv"
    submission.to_csv(submission_path, index=False, header=False, encoding="ascii")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(outputs["metrics"] / "滚动评估结果.csv", index=False, encoding="utf-8-sig")
    selected = {
        "purchase_seasonal_weight": purchase_weight, "redeem_seasonal_weight": redeem_weight,
        "component_weight_is_one_minus_seasonal": True,
        "selection_rule": "3-7月三次评分中位数、最差折、加权误差、超阈值天数依次排序",
        "tuning_windows": config["tuning_windows"], "quasi_holdout_window": config["locked_window"],
    }
    (outputs["metrics"] / "选定配置.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    contamination = contamination_test(daily, pd.Timestamp("2014-07-01"), pd.date_range("2014-07-01", "2014-07-30"), config)
    (outputs["metrics"] / "时间泄漏哨兵测试.json").write_text(json.dumps(contamination, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = outputs["metrics"] / "首榜初级结果报告.md"
    report_path.write_text(chinese_report(metrics, selected, final_prediction, contamination), encoding="utf-8")

    material_files = sorted(path for folder in outputs.values() for path in folder.rglob("*") if path.is_file())
    manifest = {
        "run_id": config["run_id"], "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generator": str(Path(__file__).relative_to(root)).replace("/", "\\"), "generator_sha256": sha256(Path(__file__)),
        "config": str(args.config.relative_to(root)).replace("/", "\\"), "config_sha256": sha256(args.config),
        "sources": ["data/derived/d006_daily_flow_feature_table_v1/daily_flow_rate_features.parquet", "validation/splits/rolling_30d_2014_03_08_v1.json", "validation/splits/backtest_2013_09_v1.json"],
        "selected": selected, "contamination_test": contamination,
        "submission": {"path": str(submission_path.relative_to(root)).replace("/", "\\"), "rows": len(submission), "sha256": sha256(submission_path)},
        "runtime": {"python": os.sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "files": [{"path": str(path.relative_to(version_root)).replace("/", "\\"), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in material_files],
    }
    manifest_path = outputs["metrics"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    make_read_only(path for folder in outputs.values() for path in folder.rglob("*") if path.is_file())
    print(f"报告: {report_path}")
    print(f"提交: {submission_path}")


if __name__ == "__main__":
    main()
