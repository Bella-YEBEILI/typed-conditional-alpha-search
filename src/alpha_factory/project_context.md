# alpha_factory 项目上下文

## 项目目标

alpha_factory 是一套因子挖掘、因子交互、遗传搜索和生产因子管理系统。

第一性原理：融合高频量价、日频量价、基本面、行业分组、条件变量以及后续可扩展的另类数据源，持续产出低相关、高质量、可进入生产库的因子。

当前主线：

- 受控束搜索：从原子字段出发，用 math/ts/group 逐层扩展，产出 seed 和 candidate。
- interaction：从 seed 库读取已剪枝 seed，做 condition/cross 交互，产物直接面向生产库，不再进入下一层 seed 搜索。
- genetic：强类型遗传规划搜索，从配置 atoms 初始化 population，批量变异、回测、选择下一代。
- refine：给定一个已知 candidate 公式，做局部变异，寻找相关性更低且仍满足 candidate 的版本。

本机版本不能连接公司内网真实数据，通常只在本机改代码，再复制到公司服务器测试。

## 主入口

主入口是 `alpha_factory.main.AlphaFactory`。

```python
from alpha_factory.main import AlphaFactory

af = AlphaFactory()

af.run_seed("/home/lxw/quant_lxw/alpha_configs/seed.py")
af.run_interaction("/home/lxw/quant_lxw/alpha_configs/interaction_condition.py")
af.run_interaction("/home/lxw/quant_lxw/alpha_configs/interaction_cross_single.py")
af.run_interaction("/home/lxw/quant_lxw/alpha_configs/interaction_cross_multi.py")
af.run_genetic("/home/lxw/quant_lxw/alpha_configs/genetic.py")
af.run_refine("group_rank(neg(ret1),sectors)", "/home/lxw/quant_lxw/alpha_configs/refine.py")

res = af.check_submission("group_rank(neg(ret1),sectors)")
res = af.submit("group_rank(neg(ret1),sectors)")
af.delete("spec_id")
af.optimize_product_exact(corr_threshold=0.5,apply=False)
```

所有 run 入口启动时会打印当前进程 PID，方便服务器上用 `top -p PID` 监控。

## 核心对象

核心数据结构在 `core/factor.py`。

- `Factor`：系统内流转的完整因子对象，包含 `meta/spec/node/result`。
- `FactorMeta`：记录来源信息，包括 `atom/layer/path/factory`。
- `FactorSpec`：记录公式信息，包括 `formula/decay/neutralize/formula_id/spec_id`。
- `FactorNode`：parser 解析后的表达式树、字段列表、算子列表。
- `FactorResult`：记录回测指标、candidate/seed 标记、打分、相关性和剪枝结果。

核心结果字段：

- 绩效：`ic/rankic/rankicir/longret/turnover/coverage/rankics`
- 标记：`is_candidate/is_seed`
- 打分：`perf_score/structure_score/batch_corr/prod_corr/fitness`
- candidate 剪枝：`candidate_prod_prune_passed/candidate_self_prune_passed`
- seed 剪枝：`seed_prune_passed`

## 字段与算子注册

字段注册表：`config/fld_registry.py`

字段元信息：

- `dtype`: `float/condition/group`
- `level_a`: 大类，例如 `daily_pv/fundamental/industry`
- `level_b`: 经济类别，例如 `ret/risk/liq/shape/value/operate/cashflow/debt/growth/improve`
- `horizon`: `point/short/mid/long`
- `dim`: `score/rank/zscore`

算子注册表：`config/ops_registry.py`

算子元信息：

- `factory`: `math/ts/group/condition/cross`
- `inputs`: 输入类型槽位
- `output`: 输出类型
- `directional`: 是否有方向性
- `order_preserving`: 是否保序

遗传规划依赖字段和算子注册表做强类型变异。

## Seed Mining

实现位置：`pipeline/seed_miner.py`

配置位置：`config/seed.py`

流程：

1. 对每个 atom 初始化 layer 0 输入。
2. 按配置逐层扩展 math/ts/group。
3. raw factors 做 IS 回测。
4. 根据 rankic 方向和 turnover 生成 adjusted factors，再做 IS 回测。
5. 标记 candidate 和 seed，计算 perf_score。
6. candidate 计算 structure_score 和 prod_corr，做 product prune 和 self prune，通过后尝试进入生产库。
7. seed 计算 structure_score、batch_corr、prod_corr 和 fitness，做 seed prune，通过后进入 accepted_seeds。
8. 每层结束后清空 value cache；非最后一层会缓存下一层输入 seed 的因子值。

终端进度中的 `C/S/A`：

- `C`: 当前 batch 已回测出的 candidate 数量。
- `S`: 当前 batch 已回测出的 seed 数量。
- `A`: 当前 batch 已完成回测总数。

## Interaction Mining

实现位置：`pipeline/interaction_miner.py`

配置位置：

- `config/interaction_condition.py`
- `config/interaction_cross_single.py`
- `config/interaction_cross_multi.py`

顶层配置使用 `seeds` 指定输入 seed/atom，不再使用 `source["atoms"]`。

```python
CONFIG = {
    "seeds":["neg(ret1)","to1"],
    "source":{
        "layers":[1,2,3],
        "paths":None,
        "factories":None,
        "seed_prune_passed":True,
        "low_rankic":None,
        "reprune":None,
        "limit":None,
    },
}
```

condition 策略会按 `seeds` 逐个独立循环运行：

- 每个 seed 单独拉取 seed 池。
- 每个 seed 单独使用完整 strategy limit。
- 每个 seed 单独做 candidate prune、写库和 run_record。
- 每个 seed 结束后刷新 product factors，避免影响下一个 seed 的相关性计算。

cross 策略仍按 `seeds` 合并运行，因为 cross 本身需要多个 seed/atom 之间形成 pair。

Interaction 流程：

1. 从 seed 库按 `seeds/layers/paths/factories/rankic/limit` 读取 seed。
2. 可选对读取出的 seed 再做一次 seed prune。
3. 清空 value cache，并缓存 seed 因子值。
4. 从 interaction `path` 表读取已完成 job，传给 strategy 去重和补抽。
5. strategy 生成 interaction factors，并在 factor 上挂载 `interaction_path`。
6. 对 generated factors 做 IS 回测、adjust、再回测。
7. raw 和 adjusted 结果写入 interaction 库。
8. candidate 计算 structure_score、prod_corr、fitness 和剪枝结果。
9. candidate 通过 product/self prune 后进入 ProductManager，尝试写入生产库。

## Interaction 数据库

数据库路径：

```text
excavate_data/interaction.sqlite
```

现在 interaction 库和 seed 库一样使用三张表：`spec/eval/path`。

### spec

- `spec_id`
- `formula_id`
- `formula`
- `decay`
- `neutralize`
- `created_at`
- `updated_at`

### eval

- `spec_id`
- `ic`
- `rankic`
- `rankicir`
- `longret`
- `turnover`
- `coverage`
- `perf_score`
- `structure_score`
- `prod_corr`
- `fitness`
- `is_candidate`
- `candidate_prod_prune_passed`
- `candidate_prod_max_corr`
- `candidate_prod_max_id`
- `candidate_self_prune_passed`
- `candidate_self_max_corr`
- `candidate_self_max_id`

### path

- `spec_id`
- `kind`: `condition/cross`
- `op`
- `params`
- `seed1`: seed1 的 `formula_id`
- `seed1_atom`
- `seed1_layer`
- `seed1_path`
- `seed1_factory`
- `seed2`: seed2 的 `formula_id`，condition 为 `None`
- `seed2_layer`
- `seed2_path`
- `seed2_factory`

去重 key 由 `kind/op/params/seed1/seed2` 组成。condition 的条件参数写入 `params`，`seed2` 相关字段为 `None`。

旧的 `condition_registry/cross_registry` 已废弃。

## Interaction Strategy

策略放在 `interaction_strategy/`，每个策略暴露：

```python
generated,records = generate(factors,done_jobs,params)
```

已有策略：

- `default_condition`: 对单个 seed 做 condition 扩展；按 seed 分配预算；已完成 job 会跳过并补抽。
- `default_cross_single`: 同一 atom 内 seed cross。
- `default_cross_multi`: 两个不同 atom 之间 cross，按 pair 相关性分 high_corr/low_corr。
- `lc_cross_single` / `lc_cross_multi`: 给与 product 相关性更低的 seed/pair 分配更多预算。
- `common_cross_multi`: 不区分高低相关，统一抽样配置。

strategy 生成的 factor 会携带 `interaction_path`，adjust 后该属性会被保留，最终由 `InteractionRepository.save_eval_batch()` 写入 `path` 表。

## Genetic Mining

实现位置：`pipeline/genetic_miner.py`

配置位置：`config/genetic.py`

核心思想：强类型遗传规划。

流程：

1. 从配置 atoms 初始化 population。
2. 每代从 population 变异生成 offspring。
3. 批量回测 raw factors。
4. adjust 后再次回测。
5. 标记 seed/candidate。
6. candidate 计算 prod_corr，若满足低相关和 candidate 标准，则尝试入生产库。
7. seed 写入 genetic.sqlite，并进入本次运行的 accepted seed pool。
8. 从旧 population 和本代 seed 中选择 survivors，进入下一代。

当前主要变异：

- `wrap_op`
- `replace_leaf`

`GeneticContext` 使用 `fld_registry` 和 `ops_registry` 过滤非法输入输出类型。

## Refine

实现位置：

- `pipeline/candidate_refiner.py`
- `genetic/local_mutator.py`

配置位置：`config/refine.py`

定位：对一个已知 candidate 公式做局部低相关改造。它不从零搜索，而是在保留主结构的前提下，寻找与生产库相关性更低且仍满足 candidate 的近邻公式。

入口：

```python
af.run_refine(formula, config_path)
```

命中条件：

- `is_candidate=True`
- signed `prod_corr <= corr_threshold`

命中后直接尝试进入 ProductManager。

## Product Flow

实现位置：

- `pipeline/product_manager.py`
- `repository/product_repository.py`

生产库路径：

```text
product_data/product.sqlite
```

生产库表：

- `spec`
- `product_is`
- `product_os`
- `product_full`

`product_is` 和 `product_os` 重复记录 `spec_id/formula/decay/neutralize` 以及各自区间的指标和 score。

`product_full` 额外记录：

- `max_corr`
- `max_corr_id`
- `avg_corr`

每次插入或删除 full product 后，需要刷新全库 full 相关性。

生产库 rankics：

```text
product_data/rankics/is/{spec_id}.npy
product_data/rankics/full/{spec_id}.npy
```

挖掘过程中参与相关性约束的 product factors 来自：

```sql
product_is INNER JOIN product_full
```

即只有 full 已入库的生产因子参与后续挖掘相关性约束。

## Repository 与持久化

所有数据库路径固定，不从外部传入。

### SeedRepository

位置：`repository/seed_repository.py`

数据库：

```text
excavate_data/seed.sqlite
```

表：

- `spec`
- `eval`
- `path`

### InteractionRepository

位置：`repository/interaction_repository.py`

数据库：

```text
excavate_data/interaction.sqlite
```

表：

- `spec`
- `eval`
- `path`

### GeneticRepository

位置：`repository/genetic_repository.py`

数据库：

```text
excavate_data/genetic.sqlite
```

表：

- `spec`
- `eval`
- `run_log`

### 挖掘侧 rankics

除生产库外，挖掘系统中所有需要持久化的 IS rankics 统一放在：

```text
excavate_data/rankics/{spec_id}.npy
```

不再使用：

```text
excavate_data/rankics/is/
excavate_data/rankics/full/
excavate_data/rankics/genetic/
```

full rankics 只属于生产库 `product_data/rankics/full/`。

## 日志与分析

### run_records.csv

位置：

```text
excavate_data/run_records.csv
```

由 `log_and_time/run_recorder.py` 维护。记录：

- `run_type`
- `config_path`
- `atom`
- `evaluated_count`
- `seed_count`
- `candidate_count`
- `final_seed_count`
- `full_product_count`
- `started_at`
- `ended_at`
- `total_sec`

### all_factors.sqlite

位置：

```text
excavate_data/all_factors.sqlite
```

由 `log_and_time/all_factor_recorder.py` 维护。用于统一观察 seed/interaction/genetic/refine 的因子质量和出货情况。

## 插件工具

`plugins/db_interactor.py` 提供便捷读取：

- `read_seed(...)`
- `read_interaction(kind=None,...)`
- `read_intersection(kind=None,...)`
- `read_genetic(...)`
- `read_excavate(...)`
- `read_product(...)`

`read_interaction(kind="condition",seed_atom="to1")` 会通过 interaction `path.seed1_atom` 筛选 condition 结果。

## 常见维护注意

- 公司服务器 Python 版本偏旧，代码中避免使用 `dict|None` 这类语法，优先使用 `Union[...]` 或不写复杂类型。
- 本机开发版本不能连接公司内网真实数据，通常只改代码，不做真实数据测试。
- 修改 SQLite 表结构后，旧库不会自动变干净；需要重建数据库。
- `structure_score` 会影响生产相关性放行逻辑，product_is/product_os 中不应长期为空。
- `check_submission()` 不写库；`submit()` 写三张 product 表并刷新 full 相关性。
- `run_refine()` 适合处理“已经是 candidate 但与生产库相关性偏高”的公式。
