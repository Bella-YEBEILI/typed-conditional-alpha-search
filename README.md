# AI辅助条件因子挖掘与研究管理系统

这是拉普拉斯奖学金实践创新类的本地代码审阅材料。整理日期为2026年9月28日，来源为LXW工作区的quant_anian副本，不表示已同步公司服务器最新版本。申请人叶蓓莉。保留现有工程结构，未修改原研究工程。

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

## 评估与限制
候选生成数量、去重公式数量、通过筛选数量、最终入库数量分别统计，不能互相替代。研究效果应由原始日志、固定样本划分、样本外指标和相关性结果支持。当前代码的样本切分在core/result_engine.py中采用索引；迁移数据时须对齐实际日期。生产筛选可能使用OSChecker和FullChecker，不应直接等同于论文独立留出样本协议。
本次排除了标注preview/placeholder的实验生成器与结果，未将其作为真实业绩证明。论文与现有本地实验版本尚未逐项对应，申请文字未采用其中的收益或提升百分比。

## 个人贡献与第三方来源
申请人申报的贡献为条件因子搜索方案、字段与算子约束、实验组织、系统集成及应用。工程含开源框架和配套代码，目录中存在某文件不等于申请人独立原创该文件。QuantaAlpha依赖RD-Agent等第三方组件；保留已有版权文字，未新授予开源许可。具体个人开发边界应结合提交记录及指导人或单位证明确认。

## 文件追溯
MANIFEST.json记录所收录源文件的相对路径、字节数、SHA-256与复制状态；EXCLUSIONS.txt记录未收录文件。以扩展名及目录筛选排除数据、缓存、模型凭证和重复归档；文本凭证模式做了基础替换，不代表完成公司代码外发授权。README与依赖提示为此次新增的说明文件。
