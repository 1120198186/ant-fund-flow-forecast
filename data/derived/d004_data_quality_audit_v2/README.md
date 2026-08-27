# Data quality audit

Complete evidence package for the five frozen official CSV files and the canonical
typed layer `d003_validated_data_layer_v2`.

Run with `scripts/run_data_quality_audit.py`. `REVIEW` means preserve the source
value and investigate modeling impact; it does not mean the row should be removed.
The HTML report is generated separately from the reviewed evidence tables.

Open `data_quality_report.html` for the browsable technical report. Its portable
package passed schema, source-provenance, block, chart and table validation. The
packaged runtime could only complete structural verification on this machine;
the installed Edge build did not satisfy its Chromium chart-extraction check.
