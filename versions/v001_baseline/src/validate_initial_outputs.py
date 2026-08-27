#!/usr/bin/env python3
"""Validate the complete v001 initial-result delivery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


RUN_ID = "initial_20260825"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    version_root = Path(__file__).resolve().parents[1]
    root = version_root.parents[1]
    run_dirs = {
        name: version_root / "artifacts" / name / RUN_ID
        for name in ["models", "metrics", "predictions", "submissions"]
    }
    failures: list[str] = []
    for name, path in run_dirs.items():
        if not path.is_dir():
            failures.append(f"缺少产物目录：{name}")

    submission_path = run_dirs["submissions"] / "首榜初级提交.csv"
    prediction_path = run_dirs["predictions"] / "2014年9月分量预测.csv"
    metrics_path = run_dirs["metrics"] / "滚动评估结果.csv"
    selected_path = run_dirs["metrics"] / "选定配置.json"
    sentinel_path = run_dirs["metrics"] / "时间泄漏哨兵测试.json"
    manifest_path = run_dirs["metrics"] / "manifest.json"
    report_path = run_dirs["metrics"] / "首榜初级结果报告.md"

    required = [submission_path, prediction_path, metrics_path, selected_path, sentinel_path, manifest_path, report_path]
    for path in required:
        if not path.is_file():
            failures.append(f"缺少文件：{path.name}")
    if failures:
        print("首榜产物校验失败")
        for failure in failures:
            print(f"- {failure}")
        return 1

    official = subprocess.run(
        [sys.executable, str(root / "scripts/validate_submission.py"), str(submission_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if official.returncode != 0:
        failures.append(f"提交契约失败：{official.stderr.strip()}")

    submission = pd.read_csv(submission_path, header=None, names=["report_date", "purchase", "redeem"])
    prediction = pd.read_csv(prediction_path)
    prediction["report_date"] = pd.to_datetime(prediction["report_date"])
    expected_dates = pd.date_range("2014-09-01", "2014-09-30", freq="D")
    if not prediction["report_date"].equals(pd.Series(expected_dates, name="report_date")):
        failures.append("9月分量预测日期不完整或顺序错误")
    for column in prediction.columns[1:]:
        values = pd.to_numeric(prediction[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            failures.append(f"9月预测列存在缺失、无穷或负数：{column}")
    if not np.allclose(prediction["component_purchase"], prediction["ridge_direct"] + prediction["rule_share"]):
        failures.append("申购分量加总关系失败")
    if not np.allclose(prediction["component_redeem"], prediction["ridge_transfer"] + prediction["ridge_consume"]):
        failures.append("赎回分量加总关系失败")
    if not np.array_equal(submission["purchase"].to_numpy(), np.rint(prediction["selected_purchase"]).astype(np.int64)):
        failures.append("提交申购值不是最终浮点预测的一次取整")
    if not np.array_equal(submission["redeem"].to_numpy(), np.rint(prediction["selected_redeem"]).astype(np.int64)):
        failures.append("提交赎回值不是最终浮点预测的一次取整")

    metrics = pd.read_csv(metrics_path)
    expected_counts = {"tuning": 10, "quasi_holdout": 2, "stress": 1}
    if metrics.groupby("role").size().to_dict() != expected_counts:
        failures.append("滚动评估窗口或模型行数不符合预期")
    for column in ["linear_score", "quadratic_score", "cubic_score"]:
        if not metrics[column].between(0, 10).all():
            failures.append(f"评分越界：{column}")
    for column in ["purchase_mape", "redeem_mape", "weighted_mape"]:
        if (metrics[column] < 0).any() or not np.isfinite(metrics[column]).all():
            failures.append(f"误差列异常：{column}")

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    for key in ["purchase_seasonal_weight", "redeem_seasonal_weight"]:
        value = float(selected[key])
        if value not in {0.0, 0.25, 0.5, 0.75, 1.0}:
            failures.append(f"融合权重不在预设网格：{key}")
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    if sentinel.get("passed") is not True or float(sentinel.get("max_absolute_difference", -1)) != 0.0:
        failures.append("时间泄漏哨兵测试未通过")

    report = report_path.read_text(encoding="utf-8")
    for heading in ["# 首榜初级结果报告", "## 结论", "## 8月准留出审计", "## 3–7月滚动评估", "## 边界"]:
        if heading not in report:
            failures.append(f"中文报告缺少章节：{heading}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["submission"]["sha256"] != sha256(submission_path):
        failures.append("提交文件哈希与清单不一致")
    for item in manifest["files"]:
        path = version_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            failures.append(f"产物清单校验失败：{item['path']}")
    for folder in run_dirs.values():
        for path in folder.rglob("*"):
            if path.is_file() and os.access(path, os.W_OK):
                failures.append(f"产物未设为只读：{path.name}")

    if failures:
        print("首榜产物校验失败")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("首榜产物校验通过：提交、分量、评估、泄漏、哈希和只读属性均符合约定")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
