#!/usr/bin/env python3
"""Build the canonical portable-report artifact from reviewed audit evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


AUDIT_VERSION = "d004_data_quality_audit_v2"


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def source(source_id: str, label: str, path: str) -> dict:
    return {"id": source_id, "label": label, "path": path}


def build_artifact(project_root: Path) -> dict:
    audit_root = project_root / "data" / "derived" / AUDIT_VERSION
    summary = pd.read_csv(audit_root / "audit_summary.csv")
    daily = pd.read_csv(audit_root / "daily_metrics.csv", parse_dates=["report_date"])
    concentration = pd.read_csv(audit_root / "user_concentration.csv")
    identities = pd.read_csv(audit_root / "additive_relationships.csv")
    anomalies = pd.read_csv(audit_root / "anomalous_dates.csv", parse_dates=["report_date"])
    leakage = pd.read_csv(audit_root / "leakage_audit.csv")
    performance = pd.read_csv(audit_root / "read_performance.csv")
    reconciliation = pd.read_csv(audit_root / "derived_reconciliation.csv")

    metrics = pd.DataFrame(
        [
            {
                "source_rows": 2_840_421,
                "users": 28_041,
                "calendar_days": 427,
                "hard_failures": int((summary["status"] == "FAIL").sum()),
                "review_checks": int((summary["status"] == "REVIEW").sum()),
                "balance_exceptions": int(identities["mismatch_rows"].sum()),
                "anomaly_flags": len(anomalies),
            }
        ]
    )
    daily_chart = daily.assign(
        report_date=daily["report_date"].dt.strftime("%Y-%m-%d")
    )[["report_date", "users", "new_users"]]
    concentration_chart = concentration[
        concentration["group"].isin(
            [
                "top_0_1_percent_users",
                "top_1_percent_users",
                "top_5_percent_users",
                "top_10_percent_users",
            ]
        )
    ].copy()
    concentration_chart["metric"] = concentration_chart["metric"].map(
        {"purchase_total": "Purchase", "redeem_total": "Redeem"}
    )
    concentration_chart["group"] = concentration_chart["group"].map(
        {
            "top_0_1_percent_users": "Top 0.1%",
            "top_1_percent_users": "Top 1%",
            "top_5_percent_users": "Top 5%",
            "top_10_percent_users": "Top 10%",
        }
    )
    concentration_chart["share_pct"] = concentration_chart["share"] * 100
    performance_chart = performance.copy()
    performance_chart["format"] = performance_chart["case"].str.split().str[0].str.upper()
    performance_chart["scope"] = performance_chart["case"].apply(
        lambda value: "August filter" if "2014-08" in value else "Full period"
    )
    performance_chart["case_label"] = (
        performance_chart["format"] + " / " + performance_chart["scope"]
    )
    anomaly_table = anomalies.sort_values("robust_score", ascending=False).head(25).copy()
    anomaly_table["report_date"] = anomaly_table["report_date"].dt.strftime("%Y-%m-%d")

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sources = [
        source(
            "official_balance",
            "Frozen official user balance CSV",
            "data/raw/official/user_balance_table.csv",
        ),
        source(
            "official_profile",
            "Frozen official user profile CSV",
            "data/raw/official/user_profile_table.csv",
        ),
        source(
            "canonical_layer",
            "Typed canonical Parquet layer",
            "data/derived/d003_validated_data_layer_v2/manifest.json",
        ),
        source(
            "audit_evidence",
            "Reviewed data-quality evidence package",
            f"data/derived/{AUDIT_VERSION}/manifest.json",
        ),
        source(
            "validation_splits",
            "Frozen rolling and stress-test definitions",
            "validation/splits/rolling_30d_2014_03_08_v1.json",
        ),
        source("audit_summary_sql", "Audit summary evidence", f"data/derived/{AUDIT_VERSION}/audit_summary.csv"),
        source("daily_metrics_sql", "Daily metric evidence", f"data/derived/{AUDIT_VERSION}/daily_metrics.csv"),
        source("concentration_sql", "User concentration evidence", f"data/derived/{AUDIT_VERSION}/user_concentration.csv"),
        source("identities_sql", "Additive relationship evidence", f"data/derived/{AUDIT_VERSION}/additive_relationships.csv"),
        source("anomalies_sql", "Anomalous date evidence", f"data/derived/{AUDIT_VERSION}/anomalous_dates.csv"),
        source("leakage_sql", "Leakage-check evidence", f"data/derived/{AUDIT_VERSION}/leakage_audit.csv"),
        source("performance_sql", "Read benchmark evidence", f"data/derived/{AUDIT_VERSION}/read_performance.csv"),
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "官方数据质量审计",
            "description": "资金流预测官方数据、派生层和时间边界的完整质量审计。",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "quality_metrics",
                    "description": "审计范围和最终状态。",
                    "dataset": "audit_metrics",
                    "sourceId": "audit_summary_sql",
                    "metrics": [
                        {"label": "资金记录", "field": "source_rows", "format": "number"},
                        {"label": "用户", "field": "users", "format": "number"},
                        {"label": "自然日", "field": "calendar_days", "format": "number"},
                        {"label": "硬失败", "field": "hard_failures", "format": "number"},
                        {"label": "需复核项", "field": "review_checks", "format": "number"},
                    ],
                }
            ],
            "charts": [
                {
                    "id": "daily_users_chart",
                    "title": "每日有记录用户数",
                    "subtitle": "427 个自然日连续，但样本规模随平台增长明显上升。",
                    "headerMarkdown": "该变化是业务规模变化，不作为脏数据删除依据。",
                    "type": "line",
                    "dataset": "daily_users",
                    "sourceId": "daily_metrics_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "report_date", "type": "temporal", "label": "日期"},
                        "y": {"field": "users", "type": "quantitative", "label": "用户数"},
                        "tooltip": [
                            {"field": "new_users", "type": "quantitative", "label": "首现用户"}
                        ],
                    },
                },
                {
                    "id": "concentration_chart",
                    "title": "长尾用户贡献率",
                    "subtitle": "Top 1% 用户贡献约 32.6% 申购和 36.7% 赎回。",
                    "headerMarkdown": "极端用户是真实业务长尾，应使用稳健建模而不是直接删除。",
                    "type": "bar",
                    "dataset": "concentration",
                    "sourceId": "concentration_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "group", "type": "nominal", "label": "用户组"},
                        "y": {"field": "share_pct", "type": "quantitative", "label": "金额贡献率 (%)"},
                        "color": {"field": "metric", "type": "nominal", "label": "目标"},
                    },
                },
                {
                    "id": "performance_chart",
                    "title": "读取性能基准",
                    "subtitle": "同机三次读取的中位耗时，越低越好。",
                    "headerMarkdown": "Parquet 同时降低存储和常用时间切片读取成本。",
                    "type": "bar",
                    "dataset": "performance",
                    "sourceId": "performance_sql",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "case_label", "type": "nominal", "label": "读取方式"},
                        "y": {"field": "median_seconds", "type": "quantitative", "label": "秒"},
                        "color": {"field": "format", "type": "nominal", "label": "格式"},
                    },
                },
            ],
            "tables": [
                {
                    "id": "identity_table",
                    "title": "业务加总关系",
                    "subtitle": "六条等式中仅同行余额勾稽有一条 100 分差异。",
                    "dataset": "identities",
                    "sourceId": "identities_sql",
                    "columns": [
                        {"field": "relationship", "label": "关系", "type": "text"},
                        {"field": "rows_checked", "label": "检查行数", "format": "number"},
                        {"field": "mismatch_rows", "label": "违例", "format": "number"},
                        {"field": "max_abs_delta", "label": "最大绝对差", "format": "number"},
                        {"field": "status", "label": "状态", "type": "text"},
                    ],
                },
                {
                    "id": "anomaly_table",
                    "title": "优先复核的异常日期指标",
                    "subtitle": "按无未来信息的滚动及同星期稳健基线排序。",
                    "dataset": "anomalies",
                    "sourceId": "anomalies_sql",
                    "defaultSort": {"field": "robust_score", "direction": "desc"},
                    "columns": [
                        {"field": "report_date", "label": "日期", "type": "date"},
                        {"field": "metric", "label": "指标", "type": "text"},
                        {"field": "value", "label": "值", "format": "number"},
                        {"field": "robust_score", "label": "稳健分数", "format": "number"},
                    ],
                },
                {
                    "id": "leakage_table",
                    "title": "时间泄漏检查",
                    "subtitle": "数据截止、填充方向、异常阈值和验证窗口逐项检查。",
                    "dataset": "leakage",
                    "sourceId": "leakage_sql",
                    "columns": [
                        {"field": "scope", "label": "范围", "type": "text"},
                        {"field": "check", "label": "检查", "type": "text"},
                        {"field": "observed", "label": "观察", "type": "text"},
                        {"field": "required", "label": "要求", "type": "text"},
                        {"field": "status", "label": "状态", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "summary",
                    "type": "markdown",
                    "body": "## 技术摘要\n\n官方数据整体可用于后续建模：主键、跨表覆盖、日期完整性、金额域、派生层总额和时间截止均通过。唯一确定的源数据异常是用户 **9872** 在 **2013-08-13** 的同行余额勾稽差 **-100 分**；该值已保留并显式标记。",
                },
                {"id": "metrics", "type": "metric-strip", "cardIds": ["quality_metrics"]},
                {
                    "id": "findings",
                    "type": "markdown",
                    "body": "## 关键发现\n\n- 2,840,421 条资金记录、28,041 名用户，用户画像双向覆盖 100%。\n- 427 个自然日连续；基金收益全覆盖，SHIBOR 的非发布日仅做历史向前填充并保留新鲜度。\n- `category1-4` 的 93.88% 空值均为 `consume_amt=0` 时的结构性空值，不插值、不删行。\n- 相邻自然日余额连续性 2,795,780 对全部通过。\n- 异常日期和极端用户是复核清单，不是清洗删除清单。",
                },
                {"id": "daily_chart", "type": "chart", "chartId": "daily_users_chart"},
                {"id": "concentration", "type": "chart", "chartId": "concentration_chart"},
                {"id": "identities", "type": "table", "tableId": "identity_table"},
                {"id": "anomalies", "type": "table", "tableId": "anomaly_table"},
                {
                    "id": "scope_method",
                    "type": "markdown",
                    "body": "## 范围、定义与方法\n\n范围包括五个冻结官方 CSV、规范 Parquet 层及现有滚动验证配置。候选主键分别为 `user_id+report_date`、`user_id` 和日期键。异常日期使用只看历史的 28 日滚动中位数/MAD 与最近 8 个同星期日基线，稳健分数大于 6 才进入复核。极端阈值仅用于本审计；建模时必须在每个训练折内重新拟合。",
                },
                {"id": "leakage", "type": "table", "tableId": "leakage_table"},
                {"id": "performance", "type": "chart", "chartId": "performance_chart"},
                {
                    "id": "limits",
                    "type": "markdown",
                    "body": "## 限制与稳健性\n\n本审计能确认结构、勾稽、覆盖和时间方向，不能单凭统计异常解释业务原因。SHIBOR 的实际可用发布时间没有更细粒度时间戳，因此预测特征仍应至少滞后一天。当前滚动配置的八月窗口为 `08-02..08-31`，与其他从月初开始的窗口不一致，正式模型比较前需要锁定统一口径。性能结果来自本机三次热缓存读取中位数，只用于本项目工程取舍。",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## 下一步\n\n1. 所有模型只读取 `d003_validated_data_layer_v2`，不直接修改或覆盖官方 CSV。\n2. 同日资金、余额、收益和全量分位点不得直接作为同日预测特征；滚动统计严格按折内拟合并滞后。\n3. 在用户分析方案确定后，再由 split JSON 动态生成 cutoff 用户快照。\n4. 先锁定六个滚动窗口的统一 30 日口径，再开始版本化模型比较。",
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": "## 后续问题\n\n- 八月最终采用 `08-01..08-30` 还是保持 `08-02..08-31`？\n- 用户分析阶段是否需要将极端用户拆分为余额型、申购型和赎回型三类？\n- SHIBOR 与基金收益是否作为主模型特征，还是仅作为消融实验？",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "audit_metrics": records(metrics),
                "daily_users": records(daily_chart),
                "concentration": records(
                    concentration_chart[["metric", "group", "share_pct"]]
                ),
                "performance": records(
                    performance_chart[
                        ["case_label", "format", "scope", "median_seconds", "rows"]
                    ]
                ),
                "identities": records(identities),
                "anomalies": records(
                    anomaly_table[["report_date", "metric", "value", "robust_score"]]
                ),
                "leakage": records(leakage),
                "reconciliation": records(reconciliation),
                "audit_summary": records(summary),
            },
            "accessIssues": [],
        },
        "sources": [
            {
                "id": row["id"],
                "query": {
                    "engine": "duckdb",
                    "sql": (
                        f"SELECT * FROM read_csv_auto('{row['path']}', header = true)"
                        if row["path"].endswith(".csv")
                        else f"SELECT * FROM read_json_auto('{row['path']}')"
                    ),
                    "description": row["label"],
                    "path": row["path"],
                    "executed_at": generated_at,
                },
            }
            for row in sources
        ],
        "package_info": {
            "originUrl": "artifact://official-data-quality-audit",
            "controls": {"edit": False, "refresh": False},
        },
    }
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = args.output or (
        project_root / "data" / "derived" / AUDIT_VERSION / "report_artifact.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report artifact: {output}")
    artifact = build_artifact(project_root)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
