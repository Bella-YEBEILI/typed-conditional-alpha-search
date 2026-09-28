# QuantaAlpha 使用说明

当前主要有两个入口脚本：

- `run.sh`：因子挖掘主入口
- `backtest_library.sh`：独立回测主入口，用已有因子库逐因子回测

两个脚本都会先读取项目根目录的 `.env`，再按 `.env` 中的配置激活 Python 运行环境。

## 1. `run.sh`：因子挖掘

`run.sh` 用来启动一轮完整的因子挖掘实验，最终会生成因子库 JSON。

### 用法

```bash
bash run.sh "<初始方向>"
bash run.sh "<初始方向>" "<因子库后缀>"
bash run.sh "<初始方向>" "<因子库后缀>" "<数据域>"
bash run.sh "<初始方向>" "<因子库后缀>" "<数据域>" "<数据模式>"
```

### 参数说明

- `初始方向`：必填，传给挖掘流程的研究方向描述。
- `因子库后缀`：可选，用来控制因子库文件名。例如 `pv_1` 对应 `data/factorlib/all_factors_library_pv_1.json`。
- `数据域`：可选，常见值为 `pv`、`fundamental`、`minutes`。
- `数据模式`：可选，常见值为 `daily`、`fundamental`、`minutes`。

### 示例

```bash
bash run.sh "price-volume factor mining"
bash run.sh "momentum reversal factors" "pv_1"
bash run.sh "intraday momentum" "min_1" "minutes" "minutes"
```

### 主要输出

- 因子库 JSON：默认在 `data/factorlib/`。
- 运行时目录：默认在 `data/runtime/<experiment_id>/`。
- 单因子导出和中间产物：默认在 `data/runtime/<experiment_id>/tq_candidates/`。

### 当前评估口径

当前挖掘评估已按 `configs/tq_upstream_alignment.yaml` 对齐 TQ：

- 挖掘阶段同时计算三组检查：`raw`、`zz1000s`、`complete`。
- `raw` 使用原始因子值。
- `zz1000s` 使用原始因子值先做 `zz1000s` 子域遮罩后的结果。
- `complete` 含义：先行业中性化，再风格回归中性化。
- 因子是否通过检查只看 `raw && zz1000s`。
- 若 `raw && zz1000s` 通过但 `complete` 不通过，则自动把生成代码中的 `META["tag"]` 设为 `"smart_styles"`。
- 若三组都通过，则 `META["tag"]` 为空字符串。
- 交易约束也参与计算：`tradables`、`limit_up_cto`、`limit_down_cto`、`limit_up_ctc`、`limit_down_ctc`。

## 2. `backtest_library.sh`：独立回测

`backtest_library.sh` 用来对已有因子库做逐因子独立回测。

它实际调用：

```bash
python -m quantaalpha.backtest.run_library_backtest
```

### 用法

```bash
bash backtest_library.sh "<因子库名或路径>"
bash backtest_library.sh "<因子库名或路径>" "<结果后缀>"
```

### 参数说明

- `因子库名或路径`：必填，可以是后缀 `pv_1`，也可以是完整文件名或绝对路径。
- `结果后缀`：可选，用来控制输出目录后缀。
- 独立回测固定只回测因子库中 `check_passed=true` 的因子。
- 独立回测固定生成：
  - `*_diagnostics.png`
  - `*_quality.json`
  - `*_report.ipynb`
- `python -m quantaalpha.backtest.run_library_backtest` 这个 Python 入口也已经固定为同样行为，不再建议传旧的布尔开关。

### 示例

```bash
bash backtest_library.sh "pv_1"
bash backtest_library.sh "pv_1" "rerun_20260416"
```

### 输出目录规则

默认输出目录格式：

```text
data/tq_upstream/candidates/<library_stem>[_<结果后缀>]/
```

### 每个因子的主要产物

必有产物：

- `*_summary.json`
- `*_factor_value.pkl`
- `*_factor_result.json`
- `*_factor_result.pkl`
- `*.py`

- `*_quality.json`
- `*_diagnostics.png`
- `*_report.ipynb`

批量汇总文件：

- `<library_stem>_batch_summary.json`

### `factor_result.pkl` 当前包含的字段

- `long_rets`
- `long_turnovers`
- `long_nums`
- `short_rets`
- `short_turnovers`
- `short_nums`
- `group_rets`
- `rankics`
- `coverages`

### 当前独立回测口径

当前独立回测已经和 TQ 对齐，包括：

- `stables_o1_o2`
- `stables_c1_c2`
- `tradables` 交易过滤
- 涨跌停约束：`limit_up_cto`、`limit_down_cto`、`limit_up_ctc`、`limit_down_ctc`
- 收益、换手、持仓数、RankIC、分组收益、绩效公式与 TQ 一致

## 3. Notebook 报告说明

独立回测会固定额外生成一个轻量 `ipynb` 报告。

这个 notebook 不是重新执行回测，而是把已生成的回测产物组织成一个便于复查的 notebook，通常包含：

- summary 指标表
- `factor_result` 字段列表
- 重新加载 `summary`、`factor_value`、`factor_result.pkl` 的代码单元
- 可选的 quality report 读取单元
- 可选的 diagnostics 图片引用

## 4. 注意事项

- 两个脚本都依赖 `.env`。
- 独立回测默认使用 `configs/tq_upstream_alignment.yaml`。
- 当前远程仓库路径：`/home/workspace/users/liwei/tqstrategyserver/QuantaAlpha`。
- 当前远程 Python 环境：`/home/workspace/users/liwei/.venv/bin/activate`。

## 5. 常用命令

因子挖掘：

```bash
bash run.sh "price-volume factor mining" "pv_1"
```

独立回测：

```bash
bash backtest_library.sh "pv_1" "rerun_20260416"
```

这条命令的含义是：

- 回测 `all_factors_library_pv_1.json`
- 输出目录后缀为 `rerun_20260416`
- 只回测 `check_passed=true` 的因子
- 固定生成图、quality report 和 notebook 报告
