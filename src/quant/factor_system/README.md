# factor_system

因子研发管理系统，核心模块为 `factor_hub`。

## 目录结构

```
factor_system/
├── dashboard/                # 前端 (Flask)
│   ├── app.py
│   └── templates/
├── factor_hub/
│   ├── main.py               # 入口，FactorManager
│   ├── core/                  # 核心引擎层
│   │   ├── config.py                  # 全局配置 (日期范围/硬性标准/评分规则)
│   │   ├── factor_loader.py           # 因子模块加载与校验
│   │   ├── factor_registry.py         # 因子注册表 (JSON)
│   │   ├── factor_selector.py         # 精选因子列表 (JSON)
│   │   ├── factor_criteria.py         # 硬性标准检查 (按 universe+domain+section)
│   │   ├── factor_scorer.py           # 综合评分 (按 universe+domain，函数式配置)
│   │   ├── factor_value_engine.py     # 因子值计算 (regular/super/minutes + decay + neutralize)
│   │   ├── factor_value_store.py      # 因子值存储 (pkl)
│   │   ├── factor_result_engine.py    # 回测引擎 (分组收益/多空/RankIC/precision)
│   │   ├── factor_result_store.py     # 回测结果存储
│   │   ├── factor_result_profiles.py  # 回测 profile 定义
│   │   ├── factor_performance_engine.py  # 绩效计算 (含 mono/monoir/precision)
│   │   └── factor_value_transformer.py   # 因子值变换 (子股票池/中性化)
│   ├── components/            # 组合组件层
│   │   ├── lib_interactor.py          # 核心交互器 (串联所有 core 组件 + filter_names + selector 代理 + rankics 缓存)
│   │   ├── factor_comparator.py       # 相关性分析
│   │   └── factor_simulator.py        # 因子回测器 (串联因子定义->因子值->回测结果->绩效指标)
│   ├── services/              # 服务层
│   │   ├── access_service.py          # 访问服务 (值/绩效/相关性，支持 **kwargs 过滤)
│   │   ├── research_service.py        # 研究服务 (evaluate 单个/批量)
│   │   ├── manage_service.py          # 管理服务 (submit/delete/update + criteria 检查)
│   │   └── select_service.py          # 精选服务 (add/remove/set)
│   ├── tools/                 # 工具层
│   │   ├── factor_plotter.py
│   │   ├── factor_spec_builder.py
│   │   ├── factor_spec_renderer.py
│   │   └── factor_template_builder.py
│   ├── plugins/               # 插件层
│   │   ├── factor_portrait.py         # 因子画像 (嵌套结构: section x period)
│   │   └── factor_weekly_report.py    # 因子周报
│   └── templates/             # 因子文件模板
│       ├── regular_factor_template.py
│       ├── super_factor_template.py
│       └── minute_factor_template.py
```

## 架构分层

```
FactorManager (main.py)
  ├── AccessService    → LibInteractor + FactorComparator + FactorPlotter + FactorPortraitStore + FactorScorer
  ├── ResearchService  → LibInteractor + FactorComparator + FactorSimulator + FactorPlotter + FactorTemplateBuilder
  ├── ManageService    → LibInteractor + FactorComparator + FactorSimulator + FactorPortraitBuilder + FactorPortraitStore + FactorCriteria
  ├── SelectService    → LibInteractor + FactorComparator
  └── FactorWeeklyReporter → LibInteractor + FactorPortraitStore + DataProvider

LibInteractor 内部串联:
  FactorRegistry + FactorSelector
  + FactorValueEngine → FactorValueStore
  + FactorResultEngine → FactorResultStore
  + FactorPerformanceEngine
  + filter_names(**kwargs) 统一过滤: tier/type/author/level/tag/domain/category
```

## 两层因子库

- **Pool**: 全量因子，`factor_registry.json`
- **Selected**: 精选子集，`selected_factors.json`（纯因子名列表）
- `filter_names(tier="selected")` 取精选，`tier="pool"` 取未选，`tier="all"`取全量
- prod_corr 默认只与同 universe + 同 domain + 已 selected 的因子比较

## 因子 META 必选字段

factor_name, author, level, domain

## portrait 结构

```python
portrait = {
    "raw":      {"all": dict, "weekly": DataFrame, "monthly": DataFrame, "yearly": DataFrame},
    "hs300s":   {"all": dict, "weekly": DataFrame, "monthly": DataFrame, "yearly": DataFrame},
    "zz1000s":  {同上},
    "complete": {同上},
}
```

## 硬性标准与评分

- 配置集中在 `core/config.py`
- 硬性标准: `CRITERIA[(universe, domain)]` → 按 section (raw/zz1000s/complete) 分别设阈值
- 评分: `SCORING[(universe, domain)]` → 函数，输入 perf dict 输出 float
- submit 时 criteria 不通过则拒绝
- get_performance 返回中自动包含 score 字段

## 数据

- 数据源: `/home/workspace/common/quant_data`
- 因子库根目录: `/home/workspace/common/quant/quant_factor`
  - `factor_registry.json` — 注册表
  - `selected_factors.json` — 精选列表
  - `factor_values/` — 因子值 pkl
  - `factor_results/{factor_name}/{profile_id}.pkl` — 回测结果
  - `factor_repo/` — 入库因子源文件
  - `factor_portraits/` — 因子画像 pkl

## 依赖

- `quant.quant_lib.analysis` — 因子算子库 (cs_rank, ts_decay_linear 等)
- `quant.quant_lib.numbafunc` — numba 加速截面运算
- `quant.quant_lib.minute_tools` — 分钟线因子引擎
- `quant.quant_lib.quantEnum` — 枚举类型 (DomainType, CategoryType, UniverseType)
