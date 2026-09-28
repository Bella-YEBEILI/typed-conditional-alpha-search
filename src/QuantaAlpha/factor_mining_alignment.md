# 因子挖掘链路对齐方案（pv / minutes / joint）

> 目标：让 minutes 和 joint 的挖掘链路收敛到 pv 的"表达式→模板渲染"稳定路径上，同时**保留**三个 domain 天然的数据字段、算子库、因子模板差异。

---

## 1. 为什么 pv 稳

pv 之所以稳，不是因为 prompt 写得特别好，而是因为 **LLM 的输出面足够窄**：

```
LLM 只产出一条表达式字符串  e.g. "ts_return(hfq_closes, 5)"
        │
        ▼
validate_expression_against_registry  （纯静态校验，零 LLM 参与）
        │
        ▼
code_template.render(expression, factor_name, data_needed)
        │
        ▼
    factor.py 模块（LLM 从未写一行 python）
```

关键文件：
- `quantaalpha/factors/coder/template.jinjia2` — 模板
- `quantaalpha/factors/factor_template/regular_factor_template.py` — 运行时基准模板
- `quantaalpha/factors/coder/evolving_strategy.py:34 _infer_expression_data_needed` — 静态校验
- `quantaalpha/factors/coder/evolving_strategy.py:1402 FactorParsingStrategy.implement_one_task` — LLM→expr→render 主路径
- `quantaalpha/factors/coder/evolving_strategy.py:1558 FactorRunningStrategy.implement_one_task` — 无需 LLM 的直通渲染路径

---

## 2. minutes / joint 现状

### minutes 现状（`evolving_strategy.py:1243 FactorMultiProcessEvolvingStrategy.implement_one_task`）

minutes 其实已经有**结构化**路径，不是我一开始以为的单正则：

```
 ┌─ _build_deterministic_minute_spec   （单正则，当前只覆盖 safe_div(ts_return(...), daily)）
 │        │ 命中
 │        ▼
 │   _render_structured_minute_module  ──→ minute_code_template.render(...)  ✅ 稳
 │        │ 不命中
 │        ▼
 ├─ LLM-produce-JSON-spec (6 次重试)
 │        │ 校验通过
 │        ▼
 │   _render_structured_minute_module  ──→ minute_code_template.render(...)  ✅ 稳
 │        │ 6 次都失败
 │        ▼
 └─ _render_structured_minute_failure_module  ──→ 渲染一个会 raise 的模块给 evaluator 当反馈
```

**LLM 产出的是 JSON spec 而不是 python 代码**，feature kinds 有：`snapshot / snapshot_return / window_aggregate / intraday_pct_aggregate / engine_run`（`evolving_strategy.py:763-877`）。

### joint 现状

joint 走的就是 `FactorMultiProcessEvolvingStrategy.implement_one_task` 同一条分支，但：
- `_build_deterministic_minute_spec` 的正则基本打不中（joint 的表达式形状更复杂）；
- 最终更多依赖 **LLM 产出 JSON spec** 分支，受 `validate_joint_pv_minutes_contract` 约束（要求 level=minutes、pv/minute 字段都必须出现、禁止裸 OHLC、用 `hfq_*` 替代）；
- 整体失败率仍比 pv 高。

### 真正的稳定性差距

| | pv | minutes | joint |
|---|---|---|---|
| LLM 输出面 | 一行 expression | JSON spec（5 种 kind） | JSON spec + 强契约 |
| 静态校验 | 无分支，一次过 | 多种拒绝理由，需要 LLM 重试 | 多种拒绝理由 + 契约校验 |
| Python 代码由谁写 | **模板** | 模板（spec 成功时）/ LLM（fallback） | 模板（spec 成功时）/ LLM（fallback） |
| fallback 路径 | 无需 fallback | LLM 全量写 python（`implement_one_task:1299`） | 同左 |
| 首次通过率（经验） | 高 | 中 | 低 |

---

## 3. 必须保留的模板差异

这几项是 domain 的本质属性，**对齐方案不会抹平**：

### 3.1 factor_template（运行时基准）
| 文件 | 签名 | 用于 |
|---|---|---|
| `regular_factor_template.py` | `calc_factor(data_ctx)` | pv / fundamental |
| `minute_factor_template.py` | `prepare_minute_datas()` + `calc_factor(data_ctx, minute_ctx)` | minutes / joint |
| `super_factor_template.py` | `calc_factor(data_ctx, factor_ctx)` | 因子组合（正交于三个 domain） |

### 3.2 jinja 渲染模板
| 文件 | 渲染参数 | 目的 |
|---|---|---|
| `coder/template.jinjia2` | `expression / factor_name / data_needed` | pv daily 纯表达式模块 |
| `coder/minute_template.jinjia2` | `factor_name / original_expression / final_expression / data_needed_literal / prepare_blocks / render_failure_message_literal` | minutes / joint 模块（带 `prepare_minute_datas` 块） |
| （缺失）`coder/joint_template.jinjia2` | 建议新增，见 §4 Stage 3 | joint 专用（可选，也可以继续复用 minute 模板 + 契约校验） |

### 3.3 数据字段白名单（`data_domains.py`）
- `PV_FIELDS` = 7 个（`hfq_opens/hfq_closes/hfq_highs/hfq_lows/volumes/vwap/turnover`）
- `MINUTE_FIELDS` = 11 个（`opens/highs/lows/closes/volumes/amounts/ratios/vwaps/turnovers/returns/lreturns`）
- `FUNDAMENTAL_FIELDS` = 23 个
- joint = `PV_FIELDS ∪ MINUTE_FIELDS`，并额外禁止裸 OHLC（`RAW_DAILY_OHLC_FIELDS`）

### 3.4 算子库（`alignment/registry.py`）
- pv / fundamental：`quantaalpha.factors.accelerated_ops`
- minutes：`accelerated_ops ∪ MinuteFactorEngine._OPS`（minute 算子在 3D 阶段用，降维后继续用 daily 算子）
- joint：同 minutes

### 3.5 契约校验
- pv：无额外契约
- minutes：`combined_domain_contract.infer_module_level / _validate_minute_engine_calls`
- joint：`validate_joint_pv_minutes_contract`（必须 level=minutes、pv/minute 字段都出现、禁止裸 OHLC、`data_needed` 必须覆盖 `data_ctx[...]`）

---

## 4. 对齐方案（分阶段）

按风险从低到高、按收益从小到大分四步。每一步都能独立上线。

### Stage 1 — Prompt 层形状对齐（纯 prompt 改动，零执行路径风险）

**目的**：让三个 domain 的 prompt overlay 覆盖整齐、把硬编码 joint 约束挪到 overlay。

**改动清单**：

1. 补齐缺失的 overlay 文件：
   - `quantaalpha/factors/coder/prompts_domain_pv.yaml` （新建，对标现有 `prompts_domain_minutes.yaml`）
   - `quantaalpha/factors/coder/prompts_domain_fundamental.yaml` （新建）
   - `quantaalpha/factors/coder/qa_prompts_domain_minutes.yaml` （新建，对标现有 `qa_prompts_domain_pv.yaml`）
2. 新建 `quantaalpha/factors/coder/prompts_domain_joint.yaml` 和 `quantaalpha/factors/coder/qa_prompts_domain_joint.yaml`，把 `coder/prompt_utils.py:63-79` 里 `is_joint_pv_minutes_run(domains)` 分支中硬编码的 joint 约束文本搬过来。
3. 扩展 `DomainPromptProxy._domain_prompt_files` 识别虚拟 domain "joint"：当 `pv` 和 `minutes` 同时激活时，额外加载 `*_domain_joint.yaml`。
4. `coder/prompt_utils.py:build_runtime_registry_constraints()` 保留动态白名单拼接（它需要运行时 domain 列表），但把**所有写死的文本约束**全部移出到 overlay。

**验证**：
```bash
# 跑 pv / minutes / joint 各 3 轮，对比三轮生成的 prompt dump 是否结构一致
python -m quantaalpha.pipeline.factor_mining --config ... --step_n 3
```

### Stage 2 — 代码层 `_PromptProxy` 去重（纯重构）

现在有 3 份几乎一样的 proxy 类：
- `factors/feedback.py:34 _PromptProxy`
- `factors/proposal.py:210 _PromptProxy`
- `factors/coder/prompt_utils.py:95 DomainPromptProxy`

**改动**：
- 把 `coder/prompt_utils.py` 的 `DomainPromptProxy` 提到 `factors/alignment/prompt_proxy.py`（新文件），另外两处全部 import 复用，删除重复实现。
- `_resolve_prompt_mode` / `resolve_prompt_mode` 也合并到这里。

**验证**：运行现有测试；diff 生成的 prompt 内容应完全一致。

### Stage 3 — 核心：扩大 minutes/joint 的确定性渲染覆盖率

**这是稳定性的真正杠杆点。**

目前 `_build_deterministic_minute_spec`（`evolving_strategy.py:905`）只覆盖一个正则，90% 情况依赖 LLM 产 JSON spec。把它升级成**通用 expression → minute_feature_spec 编译器**。

**改动**：

1. 新文件 `quantaalpha/factors/coder/minute_spec_compiler.py`：
   - 输入：`factor_expression` 字符串 + 当前 domain（minutes / joint）
   - 处理：用 `coder/factor_ast.parse_expression` 解析成 AST → 识别哪些子树是"raw minute fields 上的计算"，把这些子树改写成 `minute_features`（kind=`engine_run` 或 `snapshot_return` 等）→ 剩余部分用占位符写进 `final_expression`
   - 输出：合法的 spec（和 LLM 当前应产的一致）或抛出精确错误

2. 在 `_try_render_structured_minute_module:1152` 之前先调新编译器：
   ```python
   compiled_spec = compile_expression_to_minute_spec(target_task, domains)
   if compiled_spec is not None:
       return _render_structured_minute_module(target_task, compiled_spec), None
   # 旧的单正则 + LLM spec 路径保留作为兜底
   ```

3. 让上游的 `FactorMiningHypothesisGen` / proposal 改成**只产 expression**（和 pv 一致），不再让 LLM 直接产 spec。spec 由编译器从 expression 推导。

4. joint 复用同一套编译器。编译出的 spec 自带 `data_needed`（所有 `hfq_*` / `vwap` / `turnover`）和 `final_expression`，自然满足 `validate_joint_pv_minutes_contract`，无需 LLM 理解契约。

5. 保留但弱化 `implement_one_task:1299` 的 LLM 全量 python 分支作为最终兜底（仅在编译器 + LLM spec 都失败时触发）。

**预期收益**：minutes / joint 首次通过率靠近 pv，重试次数显著下降。

**验证**：
- 单测编译器：`tests/factors/coder/test_minute_spec_compiler.py`（覆盖 20 个常见 minute / joint 表达式形状）
- 跑 100 因子的 mining 对比，统计 "不经过 LLM 写 python 就产出合法模块" 的占比应 ≥90%。

### Stage 4 — 提案层也对齐到 expression 口径

Stage 3 做完后，proposal 应统一：

- 三个 domain 都产出 `factor_expression`（字符串），不再产 python 代码片段或 JSON spec。
- `FactorMiningHypothesisGen` 走同一个 `potential_direction_transformation` prompt，只是注入的算子/字段白名单不同（来自 `alignment/registry.py`）。
- `FactorParsingStrategy.implement_one_task`（目前只处理 pv）扩成 domain-aware：pv 走 `code_template`，minutes/joint 走 `minute_spec_compiler` + `minute_code_template`。这样 `FactorMultiProcessEvolvingStrategy`（目前 minute/joint 的主入口）可以被彻底弃用或只保留 fallback 语义。

---

## 5. 注意事项 / 不要踩的坑

1. **`regular_factor_template.py` 的 `pasteurization=False`，`template.jinjia2` 渲染时是 `True`**。这不是 bug，是 runtime 期望：daily pv 批量挖掘时要遮罩 universe。新做 minute/joint 对齐时**不要复制粘贴** `template.jinjia2` 的 header，要以对应 domain 的 `factor_template/*.py` 为准。当前 `minute_template.jinjia2` 写的是 `pasteurization: False`，和 `minute_factor_template.py` 一致，正确。

2. **minute_template 的 `prepare_minute_datas` 内部打开 h5 句柄**（`handle = h5py.File(mfe.h5_path, "r")`），所有 `load_single_minute(handle, ...)` 调用必须在 `with` 块里。新 kind 时别忘了缩进。

3. **joint 禁止裸 OHLC**：`RAW_DAILY_OHLC_FIELDS = {"opens","highs","lows","closes"}`，daily 侧必须写 `hfq_opens/hfq_closes/hfq_highs/hfq_lows`。`minute_features` 的 `inputs` 参数里的 `opens/closes` 是 minute 字段，合法。这两套同名字段是 joint 复杂度的主要来源，编译器要显式区分。

4. **`MinuteFactorEngine` 算子一元/二元 arity 严格**（`_validate_minute_engine_calls`）：`mean/std/sum/min/max/log/abs/...` 1 输入；`add/sub/mul/div/pct/corr` 2 输入。编译器生成 `engine_run` 节点时必须匹配，否则会被 contract 拒掉。

5. **`pct_change` 必须传 `fill_method=None`**（见 `coder/prompts_domain_minutes.yaml:24`）。编译器如果生成 pct_change 调用，要显式带这个参数。

6. **`universe` 只能用 `standards`**（minute），`stables`（super factor），`standards`（pv）。**不要发明 `csi300` 之类**。编译器生成的模板必须写 `"universe": "standards"`。

7. **direction 种子文件命名**：pv 支持三种备选名（`original_direction_daily.json` / `original_direction_pv.json` / `original_direction.json`，见 `factor_mining.py:50-66`）。新增 domain 时沿用这个命名策略，实验目录不要混用。

8. **`factor_template/` 不能删**。虽然 jinja 模板已经覆盖生成路径，但 `FactorWorkspace(template_folder_path=upstream_factor_template_path())` 仍会把这三个文件作为**workspace 基线**注入运行时（`factors/experiment.py:53`）。生成的 `factor.py` 会覆盖它们，但导入和目录结构依赖这些基线文件存在。

9. **不要在生成的模块里留 `if __name__ == "__main__":`**（`coder/prompts_domain_minutes.yaml:28`）—— TQ factor_hub 加载时会失败。编译器必须避免这种输出。

---

## 6. 改动优先级与工作量估算

| Stage | 改动点 | 预估工作量 | 风险 | 稳定性收益 |
|---|---|---|---|---|
| 1 | Prompt overlay 补齐 + joint 硬编码搬出 | 0.5 天 | 几乎零 | 低（为后面铺路） |
| 2 | `_PromptProxy` 三合一 | 0.5 天 | 低（纯重构） | 零（代码质量） |
| 3 | **minute/joint expression 编译器** | 3–5 天 | 中（新组件 + 改主调用） | **高** |
| 4 | Proposal/Parsing 层 expression 口径统一 | 1–2 天 | 中 | 中 |

**建议顺序**：1 → 2 → 3 → 4。Stage 3 是这次对齐的核心。

---

## 7. 验收标准

对齐完成后应满足：

- [ ] 三个 domain 的 prompt overlay 文件覆盖一致（基础 + 每域 overlay + joint 虚拟 overlay）
- [ ] 全仓库只有一份 `DomainPromptProxy`
- [ ] `FactorMiningHypothesisGen` / proposal 对三个 domain 都输出 `factor_expression` 字符串
- [ ] minute / joint 的 100 因子挖掘批次中，"不触发 LLM 写 python"（即编译器或 LLM JSON spec 命中）的比例 ≥ 90%
- [ ] `FactorMultiProcessEvolvingStrategy.implement_one_task:1299-1378`（LLM 全量写 python 分支）的命中日志计数 ≤ 5%
- [ ] factor_template 的三个 `.py` 文件保持不动；jinja 模板保持分三个（daily / minute / 可选 joint）
- [ ] 既有 pv mining 的通过率不回退
## Execution Notes (2026-04-23)

- Stage 1 is partially complete in code: coder-side domain overlays for `pv`, `minutes`, and `joint` are now present and loaded through `DomainPromptProxy`.
- The old assumption that `minutes/joint` still fall back to free-form Python generation after structured rendering is no longer true. The current chain returns a template-based failure module to evaluator feedback instead.
- Stage 3 should be executed incrementally: first expand `expression -> minute spec` compilation coverage for common minute/joint shapes, then keep the existing LLM JSON-spec path only as a fallback.
- The practical target is chain alignment, not identical class reuse: `pv` remains `expression -> daily template`, while `minutes/joint` should converge to `expression -> minute spec -> minute template`.
