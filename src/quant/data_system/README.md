## 目录分类

- `pv/`：基础量价数据
- `pv_derived/`：由基础量价派生的短中期量价特征
- `index_data/`：指数收益率数据
- `industry/`：行业分类
- `limit/`：涨跌停掩码
- `minute_vwaps/`：开盘后不同时间窗口的 VWAP
- `status/`：交易状态与上市状态
- `styles/raw/`：原始风格因子
- `styles/derived/`：修正并标准化后的风格因子
- `universe/`：股票池与指数成分掩码
- `fundamental/`：基本面数据

## `pv/` 基础量价数据

- `adj_factors`
  - 复权因子，用于把原始价格还原为后复权价格。
- `opens`
  - 未复权开盘价。
- `highs`
  - 未复权最高价。
- `lows`
  - 未复权最低价。
- `closes`
  - 未复权收盘价。
- `hfq_opens`
  - 后复权开盘价，定义为 `opens*adj_factors`。
- `hfq_highs`
  - 后复权最高价，定义为 `highs*adj_factors`。
- `hfq_lows`
  - 后复权最低价，定义为 `lows*adj_factors`。
- `hfq_closes`
  - 后复权收盘价，定义为 `closes*adj_factors`。
- `ctc_returns`
  - 收盘到收盘收益率，定义为 `hfq_closes_t/hfq_closes_{t-1}-1`。
- `cto_returns`
  - 昨收到今开收益率，定义为 `hfq_opens_t/hfq_closes_{t-1}-1`。
- `oto_returns`
  - 开盘到开盘收益率，定义为 `hfq_opens_t/hfq_opens_{t-1}-1`。
- `volumes`
  - 成交量。
- `amounts`
  - 成交额。
- `vwaps`
  - 日内成交量加权平均价，定义为 `amounts/volumes`。
- `turnovers`
  - 换手率，定义为 `amounts/float_market_caps`。
- `float_market_caps`
  - 流通市值。
- `total_market_caps`
  - 总市值。

## `pv_derived/` 量价派生特征

由 `quant_union/common/datalab.py` 中 `build_derived_pvs()` 构造。

- `ema5`
  - `hfq_closes` 的 5 日指数移动平均。
- `ema20`
  - `hfq_closes` 的 20 日指数移动平均。
- `adv5`
  - `volumes` 的 5 日均值。
- `adv20`
  - `volumes` 的 20 日均值。
- `vol5`
  - `ctc_returns` 的 5 日滚动标准差。
- `vol20`
  - `ctc_returns` 的 20 日滚动标准差。
- `pvc5`
  - `hfq_closes` 与 `volumes` 的 5 日滚动相关系数。
- `pvc20`
  - `hfq_closes` 与 `volumes` 的 20 日滚动相关系数。
- `te`
  - 成交效率，定义为 `abs(ctc_returns) / amounts`。

## `index_data/` 指数收益率

`index_data.pkl` 是一个按交易日对齐的指数收益率表，当前包含以下列：

- `sz50s_returns`
  - 上证 50 指数日收益率。
- `hs300s_returns`
  - 沪深 300 指数日收益率。
- `zz500s_returns`
  - 中证 500 指数日收益率。
- `zz800s_returns`
  - 中证 800 指数日收益率。
- `zz1000s_returns`
  - 中证 1000 指数日收益率。
- `mscias_returns`
  - MSCI A 指数日收益率。
- `zzhls_returns`
  - `zzhls` 指数日收益率。
- `zzhls_h_returns`
  - `zzhls_h` 指数日收益率。

`size_rets.pkl` 为`pd.Series`，定义为`hs300s_returns - zz1000s_returns`

## `industry/` 行业分类

- `industrys`
  - 个股所属行业分类矩阵。
  - 常用于 `group_rank`、`group_neutralize` 等截面分组操作。

## `limit/` 涨跌停掩码

以下字段都是布尔矩阵，`True` 表示该股票在该交易日触发对应涨跌停状态。

- `limit_up_ctc`
  - 以昨收到今收收益率为基准计算的收盘涨停掩码。
- `limit_down_ctc`
  - 以昨收到今收收益率为基准计算的收盘跌停掩码。
- `limit_up_cto`
  - 以昨收到今开收益率为基准计算的开盘涨停掩码。
- `limit_down_cto`
  - 以昨收到今开收益率为基准计算的开盘跌停掩码。

补充说明：

- 阈值按板块和 ST 状态区分：
  - 主板通常为 9.9%，ST 在特定时期为 4.9%
  - 创业板 2020-08-21 后通常为 19.9%
  - 科创板通常为 19.9%

## `minute_vwaps/` 开盘后 VWAP

以下字段均为从开盘后 09:31 开始累计到指定时点的 VWAP：

- `open_vwaps_1m`
  - 09:31 到 09:31 的 VWAP。
- `open_vwaps_5m`
  - 09:31 到 09:35 的 VWAP。
- `open_vwaps_10m`
  - 09:31 到 09:40 的 VWAP。
- `open_vwaps_15m`
  - 09:31 到 09:45 的 VWAP。
- `open_vwaps_30m`
  - 09:31 到 10:00 的 VWAP。

## `status/` 状态数据

- `tradables`
  - 可交易掩码，`True/1` 表示该日可正常交易。
- `st_stocks`
  - ST 掩码，`True/1` 表示该股票当日为 ST 状态。
- `days_since_ipos`
  - 距离上市日的天数。

## `styles/raw/` 原始风格因子

输入数据主要包括：

- `float_market_caps`
- `ctc_returns`
- `turnovers`

字段定义：

- `size`
  - `ln(float_market_caps)`。
- `nlsize`
  - `size^3` 对 `size` 做截面回归后的负残差。
- `mom`
  - 过去 140 日累计收益，排除最近 20 日。
- `rev1`
  - 单日反转，定义为 `-ctc_returns`。
- `rev3`
  - `rev1` 的 3 日滚动均值。
- `rev5`
  - `rev1` 的 5 日滚动均值。
- `vol20`
  - `ctc_returns` 的 20 日滚动标准差。
- `vol60`
  - `ctc_returns` 的 60 日滚动标准差。
- `vol240`
  - `ctc_returns` 的 240 日滚动标准差。
- `liq20`
  - `ln(turnovers 的 20 日滚动均值)`。
- `liq60`
  - `ln(turnovers 的 60 日滚动均值)`。
- `liq240`
  - `ln(turnovers 的 240 日滚动均值)`。
- `beta`
  - 相对市值加权市场收益率的 240 日滚动 Beta。

## `styles/derived/` 修正风格因子

由 `styles/raw/` 修正并标准化后得到，主要用于保证 `standards` 股票池内更稳定的覆盖度和可比性。

处理方式：

- 对原始风格因子做截断（winsorize）
- 再做市值加权截面标准化
- 部分长窗口因子会用较短窗口结果补缺

字段定义：

- `Size`
  - 修正并标准化后的 `size`。
- `Nlsize`
  - 修正并标准化后的 `nlsize`。
- `Mom`
  - 修正并标准化后的 `mom`。
- `Rev`
  - 由 `rev1`、`rev3`、`rev5` 合成的反转风格。
- `Vol`
  - 由 `vol20`、`vol60`、`vol240` 合成的波动率风格。
- `Liq`
  - 由 `liq20`、`liq60`、`liq240` 合成的流动性风格。
- `Beta`
  - 修正并标准化后的 `beta`。

## `universe/` 股票池与指数成分

以下字段均为布尔掩码，`True` 表示该股票在该日属于对应股票池或指数成分集合。

- `liquidity_amounts_60d_top75pct_min10M`
  - 基于 `amounts` 的 60 日滚动中位数做流动性筛选，保留横截面前 75% 且中位成交额有效的股票。
- `stables`
  - 稳定股票池，要求同时满足：
    - `abs(ctc_returns) < 0.21`
    - 当日可交易
    - 过去 20 日正常交易天数大于 10
    - 上市天数大于 60
    - 非 ST
- `universe_liq_div3y`
  - 红利流动性股票池，要求同时满足：
    - 非 ST
    - 可交易
    - 流通市值和成交额均不在横截面后 20%
    - 过去三个完整年度连续现金分红
- `standards`
  - `stables & liquidity_amounts_60d_top75pct_min10M`。
- `smalls`
  - `standards & (~hs300s) & (~zz500s)`。
- `sz50s`
  - 上证 50 成分股掩码。
- `hs300s`
  - 沪深 300 成分股掩码。
- `zz500s`
  - 中证 500 成分股掩码。
- `zz800s`
  - 中证 800 成分股掩码。
- `zz1000s`
  - 中证 1000 成分股掩码。
- `zzhls`
  - `zzhls` 指数成分股掩码。


## `fundamental/`

基本面数据由 [datahub.core.fundamental_updater.py]统一构建，用来把财报、分红、市值等输入整理成按交易日对齐的因子面板。

输入数据：

- `fundamentals.pkl`
  - 原始文件：`/home/workspace/common/product_pkl/fundamentals/fundamentals.pkl`。
  - 使用字段：
    - `code`
    - `date`
    - `quarter`
    - `updateDate`
    - `np`
    - `revenue`
    - `op`
    - `totalAssets`
    - `totalLiab`
    - `totalOwnerEquity`
    - `operatingTotalCost`
    - `longLiab`

预处理规则：

- 仅保留 `2015-01-01` 之后的数据。
- 删除 `code/date/updateDate` 缺失记录。
- 删除 `updateDate < date` 的未来占位行。
- 同一股票、同一公告日、同一报告期只保留最后一条。
- 公告统一映射到下一个交易日，避免前视。
- 利润表口径先拆单季，再滚成 TTM。

输出结果：

- `fundamental_factors/`
  - 默认目录：`/home/workspace/common/quant_data/fundamental_factors/`。
  - 每个 `pkl` 文件都是按交易日和股票代码对齐的因子矩阵。

字段定义：

- `ep`
  - 盈利收益率，`np_ttm / total_market_caps`。
- `bp`
  - 账面市值比，`totalOwnerEquity / total_market_caps`。
- `roe`
  - 净资产收益率，`np_ttm / totalOwnerEquity`。
- `roe_trend`
  - 当前 `roe` 减去去年同期 `roe`。
- `profit_margin`
  - 营业利润率，`op_ttm / revenue_ttm`。
- `cash_coverage`
  - 利润偿债覆盖度，`op_ttm / totalLiab`。
- `leverage`
  - 杠杆率，`totalLiab / totalAssets`。
- `asset_growth`
  - 总资产同比增速。
- `revenue_growth`
  - `revenue_ttm` 同比增速。
- `div_yield_ttm`
  - TTM 股息率，过去 243 个交易日每股现金分红 / 收盘价。
- `div_yield_yoy`
  - 当前 `div_yield_ttm` 减去去年同期 `div_yield_ttm`。
- `div_yield_stability`
  - 股息率稳定性，`-(std/mean)`，约 3 年窗口。
- `payout_ratio`
  - 派息率，过去 243 个交易日总分红 / `np_ttm`。
- `consecutive_div_years`
  - 向前连续有现金分红的年数。
- `earnings_growth`
  - `np_ttm` 同比增速。
- `roe_stability`
  - 最近 12 个季度 `roe` 的负标准差，数值越大越稳定。
- `net_margin`
  - 净利率，`np_ttm / revenue_ttm`。
- `asset_turnover`
  - 资产周转率，`revenue_ttm / totalAssets`。
- `operating_cost_ratio`
  - 营业成本率，`operatingTotalCost_ttm / revenue_ttm`。
- `cost_ratio_trend`
  - 当前营业成本率减去去年同期营业成本率。
- `long_term_debt_ratio`
  - 长期负债占比，`longLiab / totalLiab`。
- `ff_factors`
  - 日频 FF 风格收益序列，包括：
    - `ff_mkt`
    - `ff_smb`
    - `ff_hml`
    - `ff_rmw`
    - `ff_cma`
- `ff_factors_rolling`
  - `ff_factors` 的 `20d`、`60d` 滚动累计版本。

更新说明：

- 入口函数是 `FundamentalUpdater.update()`。
- 已有历史结果时走增量更新。
- 增量更新会以前次结果末日为续算起点，并向前回补 warmup 窗口后重算。
- 没有历史结果时从 2015 年开始全量构建。

## `futures/`

衍生品数据，待更新