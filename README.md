# AI辅助条件因子挖掘与研究管理系统

这是拉普拉斯奖学金实践创新类的本地代码审阅材料。

## 范围与阅读顺序
1. src/alpha_factory：条件因子挖掘主系统，先读main.py和project_context.md。
2. src/alpha_factory/config：字段类型、算子注册和搜索配置。
3. src/alpha_factory/core、pipeline、genetic：解析、计算、评估、条件交互和遗传搜索。
4. src/QuantaAlpha/quantaalpha：LLM辅助假设生成、规划、编码、反馈与演化；README.md记载已有入口。
5. src/quant/factor_system：FactorHub因子评估、筛选、注册和结果管理；src/quant其他目录保留其数据及策略配套工程。
6. experiments：既有实验与条件合成审计脚本。脚本保留历史绝对路径，使用前须适配项目根目录与输出目录。

## 数据与输入输出
AlphaFactory的数据管理器读取src/alpha_factory/factory_data/data下的npy面板，维度为日期×股票；axis/dates.pkl、axis/stocks.pkl记录轴信息。字段清单以config/fld_registry.py为准。回测还依赖standards、tradables、limit_up_cto、trade_returns等字段。原始行情与公司数据未打包。
种子表达式和配置输入后，解析器生成表达式树，算子执行得到因子面板，再生成IC、RankIC、RankICIR、换手率和覆盖率等指标。seed、interaction、genetic与product各类SQLite存储保存公式、规格和评价记录。

## 核心环境与操作
建议在获授权的独立研究环境使用Python 3.10或以上版本；核心依赖列于requirements-core.txt，版本未锁定。QuantaAlpha扩展依赖见src/QuantaAlpha/requirements.txt。此快照未在本次整理中运行真实数据挖掘。
在本目录配置环境后，将src加入PYTHONPATH。准备一致的数据字段和轴，检查各配置的并发数及输出目录，再使用以下既有API：
```python
from alpha_factory.main import AlphaFactory
af = AlphaFactory()
af.run_seed("src/alpha_factory/config/seed.py")
af.run_interaction("src/alpha_factory/config/interaction_condition.py")
af.run_genetic("src/alpha_factory/config/genetic.py")
```
这些调用会开展挖掘并写入本地结果库，不是无数据演示。QuantaAlpha需另行配置获授权的模型接口与数据环境；未包含实际.env、密钥和访问凭证。部分配套模块依赖公司内部数据接口，因此本包可供源码审阅，但不承诺开箱运行全流程。
