# Derived data

Generated datasets belong in versioned subfolders such as `d001_daily_aggregate/`. A generated dataset is not valid unless its folder records the source manifest and exact generator command.

Current canonical data foundation:

- `d002_official_csv_snapshot/`: byte-identical official CSV copy.
- `d003_validated_data_layer_v2/`: typed, date-sorted Parquet layer.
- `d004_data_quality_audit_v2/`: formal audit evidence, notebook, and HTML report.
- `d005_direct_feature_analysis_v1/`: direct feature associations and fold-safe feature catalog.
- `d006_daily_flow_feature_table_v1/`: canonical 427-day flow, balance, and rate feature table.
- `d007_user_behavior_segmentation_research_v6/`: leakage-safe rolling user behavior labels, active-user clustering diagnostics, and Chinese future-outcome research report.

