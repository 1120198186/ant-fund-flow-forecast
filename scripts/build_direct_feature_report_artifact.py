#!/usr/bin/env python3
"""Build the portable direct-feature analysis report artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


ANALYSIS_VERSION = "d005_direct_feature_analysis_v1"
PRIMARY_TARGETS = [
    "total_purchase_amt",
    "total_redeem_amt",
    "direct_purchase_amt",
    "purchase_bank_amt",
    "transfer_amt",
]


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def build(project_root: Path) -> dict:
    analysis = project_root / "data" / "derived" / ANALYSIS_VERSION
    demographic = pd.read_csv(analysis / "demographic_associations.csv")
    group_summary = pd.read_csv(analysis / "demographic_group_summary.csv")
    combinations = pd.read_csv(analysis / "demographic_combination_associations.csv")
    rates = pd.read_csv(analysis / "rate_associations.csv")
    transactions = pd.read_csv(analysis / "transaction_component_associations.csv")
    catalog = pd.read_csv(analysis / "feature_combination_catalog.csv")

    primary_demo = demographic[
        demographic["target"].isin(PRIMARY_TARGETS)
        & demographic["metric"].eq("log1p_amount_per_exposure_day")
    ].copy()
    primary_demo["label"] = primary_demo["feature"] + " / " + primary_demo["target"]
    primary_demo = primary_demo.sort_values("eta_squared", ascending=False)

    primary_rates = rates[rates["target"].isin(PRIMARY_TARGETS)].copy()
    primary_rates["label"] = primary_rates["feature"] + " / " + primary_rates["target"]
    rate_short = primary_rates.nlargest(20, "short_run_abs_diff1").copy()
    rate_lagged = primary_rates.nlargest(20, "predictor_safe_abs_lagged_diff1").copy()

    component = transactions[
        transactions["left"].isin(PRIMARY_TARGETS)
        | transactions["right"].isin(PRIMARY_TARGETS)
    ].nlargest(20, "max_abs_correlation").copy()
    component["label"] = component["left"] + " / " + component["right"]

    city_purchase = group_summary[
        group_summary["feature"].eq("city")
        & group_summary["target"].eq("total_purchase_amt")
    ].sort_values("amount_mean", ascending=False)
    combo_top = combinations[
        combinations["target"].isin(PRIMARY_TARGETS)
    ].nlargest(20, "eta_squared")
    catalog_counts = (
        catalog.groupby(["priority", "family"], observed=True)
        .size()
        .rename("feature_count")
        .reset_index()
    )

    metrics = pd.DataFrame(
        [
            {
                "users": 28_041,
                "dates": 427,
                "city_max_eta2": primary_demo.loc[
                    primary_demo["feature"].eq("city"), "eta_squared"
                ].max(),
                "sex_max_eta2": demographic.loc[
                    demographic["feature"].eq("sex")
                    & demographic["target"].isin(PRIMARY_TARGETS),
                    "eta_squared",
                ].max(),
                "constellation_max_eta2": demographic.loc[
                    demographic["feature"].eq("constellation")
                    & demographic["target"].isin(PRIMARY_TARGETS),
                    "eta_squared",
                ].max(),
                "max_lagged_rate_change_corr": primary_rates[
                    "predictor_safe_abs_lagged_diff1"
                ].max(),
            }
        ]
    )

    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_rows = [
        (
            "demographic_sql",
            "Demographic association evidence",
            f"data/derived/{ANALYSIS_VERSION}/demographic_associations.csv",
        ),
        (
            "group_sql",
            "Demographic group summary",
            f"data/derived/{ANALYSIS_VERSION}/demographic_group_summary.csv",
        ),
        (
            "combination_sql",
            "Demographic combination evidence",
            f"data/derived/{ANALYSIS_VERSION}/demographic_combination_associations.csv",
        ),
        (
            "rate_sql",
            "Rate association evidence",
            f"data/derived/{ANALYSIS_VERSION}/rate_associations.csv",
        ),
        (
            "transaction_sql",
            "Transaction-component evidence",
            f"data/derived/{ANALYSIS_VERSION}/transaction_component_associations.csv",
        ),
        (
            "catalog_sql",
            "Feature combination catalog",
            f"data/derived/{ANALYSIS_VERSION}/feature_combination_catalog.csv",
        ),
    ]
    manifest_sources = [
        {"id": source_id, "label": label, "path": path}
        for source_id, label, path in source_rows
    ]
    query_sources = [
        {
            "id": source_id,
            "query": {
                "engine": "duckdb",
                "sql": f"SELECT * FROM read_csv_auto('{path}', header = true)",
                "description": label,
                "executed_at": generated,
            },
        }
        for source_id, label, path in source_rows
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "直接特征与目标关联分析",
            "description": "人口属性、支付宝收益、SHIBOR 和交易分量与资金流目标的分层关联分析。",
            "generatedAt": generated,
            "cards": [
                {
                    "id": "analysis_metrics",
                    "dataset": "metrics",
                    "sourceId": "demographic_sql",
                    "description": "分析样本与直接效应上限。",
                    "metrics": [
                        {"label": "用户", "field": "users", "format": "number"},
                        {"label": "日期", "field": "dates", "format": "number"},
                        {"label": "城市最大 eta²", "field": "city_max_eta2", "format": "number"},
                        {"label": "性别最大 eta²", "field": "sex_max_eta2", "format": "number"},
                        {
                            "label": "滞后利率变化最大相关",
                            "field": "max_lagged_rate_change_corr",
                            "format": "number",
                        },
                    ],
                }
            ],
            "charts": [
                {
                    "id": "demographic_chart",
                    "title": "人口属性与主要交易目标的效应量",
                    "subtitle": "用户粒度；目标为自首次出现后的日均金额 log1p；eta²。",
                    "headerMarkdown": "城市为小效应，性别和星座几乎没有直接解释力。",
                    "type": "bar",
                    "dataset": "demographic",
                    "sourceId": "demographic_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "label", "type": "nominal", "label": "特征 / 目标"},
                        "y": {"field": "eta_squared", "type": "quantitative", "label": "eta²"},
                        "color": {"field": "feature", "type": "nominal", "label": "人口属性"},
                    },
                },
                {
                    "id": "rate_short_chart",
                    "title": "利率短期变化与目标变化",
                    "subtitle": "427 日；一日利率变化与目标 log1p 一日变化的 Pearson 相关。",
                    "headerMarkdown": "原始水平相关受共同趋势影响，差分结果更接近短期关系。",
                    "type": "bar",
                    "dataset": "rate_short",
                    "sourceId": "rate_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "label", "type": "nominal", "label": "利率 / 目标"},
                        "y": {
                            "field": "diff1_corr_log_target",
                            "type": "quantitative",
                            "label": "一阶变化相关",
                        },
                        "color": {"field": "target", "type": "nominal", "label": "目标"},
                    },
                },
                {
                    "id": "rate_lagged_chart",
                    "title": "可预测方向的滞后利率变化相关",
                    "subtitle": "前一日利率变化与今日目标变化；绝对值最高约 0.081。",
                    "headerMarkdown": "利率当前应作为候选补充特征，而不是主导信号。",
                    "type": "bar",
                    "dataset": "rate_lagged",
                    "sourceId": "rate_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "label", "type": "nominal", "label": "利率 / 目标"},
                        "y": {
                            "field": "lagged_diff1_corr_log_target",
                            "type": "quantitative",
                            "label": "滞后变化相关",
                        },
                        "color": {"field": "target", "type": "nominal", "label": "目标"},
                    },
                },
                {
                    "id": "component_chart",
                    "title": "交易分量之间的每日相关",
                    "subtitle": "同日每日汇总 Spearman；高相关中包含严格加总关系。",
                    "headerMarkdown": "同日组成项只能解释结构，预测时必须使用滞后值。",
                    "type": "bar",
                    "dataset": "components",
                    "sourceId": "transaction_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "label", "type": "nominal", "label": "交易分量对"},
                        "y": {"field": "daily_spearman", "type": "quantitative", "label": "Spearman"},
                        "color": {
                            "field": "relationship_type",
                            "type": "nominal",
                            "label": "关系类型",
                        },
                    },
                },
            ],
            "tables": [
                {
                    "id": "city_table",
                    "title": "城市与日均总申购",
                    "subtitle": "按用户自首次出现后的暴露天数标准化。",
                    "dataset": "city_purchase",
                    "sourceId": "group_sql",
                    "defaultSort": {"field": "amount_mean", "direction": "desc"},
                    "columns": [
                        {"field": "feature_value", "label": "城市代码", "type": "text"},
                        {"field": "users", "label": "用户", "format": "number"},
                        {"field": "amount_mean", "label": "日均申购", "format": "number"},
                        {
                            "field": "active_day_rate_mean",
                            "label": "交易日占比",
                            "format": "percent",
                        },
                        {
                            "field": "positive_day_amount_mean",
                            "label": "发生交易后的均额",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "combination_table",
                    "title": "人口属性组合效应",
                    "subtitle": "组合提升有限，三阶组合仅作为后续验证项。",
                    "dataset": "combinations",
                    "sourceId": "combination_sql",
                    "defaultSort": {"field": "eta_squared", "direction": "desc"},
                    "columns": [
                        {"field": "feature", "label": "组合", "type": "text"},
                        {"field": "target", "label": "目标", "type": "text"},
                        {"field": "eta_squared", "label": "eta²", "format": "number"},
                        {"field": "raw_groups", "label": "分组数", "format": "number"},
                        {"field": "effect_label", "label": "效应", "type": "text"},
                    ],
                },
                {
                    "id": "catalog_table",
                    "title": "候选特征数量",
                    "subtitle": "完整公式、粒度和泄漏规则见 FEATURE_COMBINATIONS.md。",
                    "dataset": "catalog_counts",
                    "sourceId": "catalog_sql",
                    "columns": [
                        {"field": "priority", "label": "优先级", "type": "text"},
                        {"field": "family", "label": "特征族", "type": "text"},
                        {"field": "feature_count", "label": "数量", "format": "number"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {
                    "id": "summary",
                    "type": "markdown",
                    "body": "## 结论\n\n直接特征中，**城市**有稳定但较小的用户行为差异；**性别和星座**的直接解释力接近零。利率原始水平与资金量的高相关主要受共同时间趋势影响，转为变化量并滞后后信号明显减弱。交易分量的高相关多数来自业务加总关系，适合做结构分析，但不能作为同日预测输入。",
                },
                {"id": "metrics", "type": "metric-strip", "cardIds": ["analysis_metrics"]},
                {"id": "demo", "type": "chart", "chartId": "demographic_chart"},
                {"id": "city", "type": "table", "tableId": "city_table"},
                {"id": "combo", "type": "table", "tableId": "combination_table"},
                {"id": "rate_short", "type": "chart", "chartId": "rate_short_chart"},
                {"id": "rate_lagged", "type": "chart", "chartId": "rate_lagged_chart"},
                {"id": "component", "type": "chart", "chartId": "component_chart"},
                {
                    "id": "interpretation",
                    "type": "markdown",
                    "body": "## 如何解释\n\n- `direct_purchase_amt = purchase_bal_amt + purchase_bank_amt`，`total_redeem_amt = consume_amt + transfer_amt` 等是严格等式，高相关不是独立预测信号。\n- `mfd_daily_yield` 与 SHIBOR 口径不同，不直接做利差；利差只使用七日年化收益率。\n- 人口属性使用用户粒度效应量；利率使用 427 个日期样本，避免将每日利率复制到用户行虚增样本量。\n- 本轮全时期用户统计仅用于描述，正式模型必须在每个滚动训练折内重算。",
                },
                {"id": "catalog", "type": "table", "tableId": "catalog_table"},
                {
                    "id": "priority",
                    "type": "markdown",
                    "body": "## 特征优先级\n\n**P0**：原始人口类别、两两人口组合、利率滞后与变化、七日收益率与 SHIBOR 利差、目标历史滞后/滚动、滞后的交易构成比例。\n\n**P1**：利率波动与滚动状态、分群历史、人口组与利率环境交互、三阶人口组合。\n\n**暂不使用**：同日交易组成项、全量目标编码、用户 ID 高维交叉、全部数值多项式组合。",
                },
                {
                    "id": "next",
                    "type": "markdown",
                    "body": "## 下一步建议\n\n不要按全时期相关系数直接删特征。下一阶段应先实现 P0 的折内、滞后特征生成器，再以六个滚动窗口逐族加入：人口属性 → 历史行为 → 利率 → 利差/交互。只有能稳定改善中位数和最差窗口误差的特征族才保留。",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "metrics": records(metrics),
                "demographic": records(primary_demo),
                "rate_short": records(rate_short),
                "rate_lagged": records(rate_lagged),
                "components": records(component),
                "city_purchase": records(city_purchase),
                "combinations": records(combo_top),
                "catalog_counts": records(catalog_counts),
            },
            "accessIssues": [],
        },
        "sources": query_sources,
        "package_info": {
            "originUrl": "artifact://direct-feature-analysis",
            "controls": {"edit": False, "refresh": False},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output = root / "data" / "derived" / ANALYSIS_VERSION / "report_artifact.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report artifact: {output}")
    output.write_text(
        json.dumps(build(root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
