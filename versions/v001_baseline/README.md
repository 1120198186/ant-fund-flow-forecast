# v001 首榜基线

状态：`frozen`

## 方案

- 总申购、总赎回分别建立同星期与近28日水平融合的季节锚点。
- 主动申购、转账赎回、消费使用一次性30步 Ridge 残差模型；收益转入按历史万份收益与预测余额递推。
- 仅用2014年3–7月选择总量层融合权重，8月作为配置冻结后的准留出审计。
- 候选模型未通过跨月稳定性门槛，因此正式首榜结果回退为100%季节锚点；分量模型只保留为诊断产物。

## 初级结果

- 8月申购平均相对误差：`0.1075`；赎回平均相对误差：`0.2205`。
- 8月本地代理评分：线性`5.1785`、二次`4.1119`、三次`3.4232`。
- 2014年9月预测日均申购`258,427,629`分、日均赎回`261,424,354`分。
- 截止日后哨兵污染测试通过，最大预测差异为`0`。

## 线上提交

- 提交日期：`2026-08-25`
- 提交产物：`artifacts/submissions/initial_20260825/首榜初级提交.csv`
- 线上得分：`106.5001`
- 详细记录：`SUBMISSION_LOG.md`

## 复现

```powershell
.\.venv\Scripts\python.exe versions\v001_baseline\src\run_initial_model.py
.\.venv\Scripts\python.exe versions\v001_baseline\src\validate_initial_outputs.py
```

正式产物位于各类 `artifacts/*/initial_20260825/` 目录，均带清单和只读保护。完整中文报告见 `artifacts/metrics/initial_20260825/首榜初级结果报告.md`。

