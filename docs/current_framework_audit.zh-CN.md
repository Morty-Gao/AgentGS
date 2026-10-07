# AgentGS 重构前运行框架调研

调研日期：2026-10-07。以工作目录中的实际源码为依据，静态阅读入口、调度、规划、skills、工具实现、数据解析、服务管理与评测。没有调用付费模型、启动旧服务或重新运行完整评测。本文不将源码存在视为端到端验收通过。

## 1. 当前有三个不同的项目状态

- 新工作仓库：Ubuntu-24.04 `/home/gaosong/Code/Project/AgentGS`。现有 uv 项目、下载 CLI、数据与验证记录；`src/agentgs/__init__.py` 仍是打印 Hello 的入口。旧 agent 框架没有迁入。
- 本次主要源码依据：Windows `D:/gaosong/Desktop/大创项目（新)/GeoToolAgent`。
- 另有交付副本 `D:/gaosong/Desktop/大创项目（新)/AgentGS_主项目_20261006`。它与旧主项目并不完全相同：注册工具分别为101与110，runtime、catalog、skills 和规划代码也有差异。不可把两个目录的文件任意拼装成新版本。

## 2. 实际运行拓扑

用户或评测器 -> FastAPI agent 服务(app.py，默认50005) -> scripts/retrieve_tool_info.py 调度 -> HTTP 工具客户端 -> FastAPI工具服务(ToolsAgent/main.py，默认60002) -> 工具注册表 -> 各领域工具实现 -> 数据文件/生成产物/外部后端。

最终工具证据送往 LLM 生成最终答案；chat_trace 会返回 trace、调用使用信息等，并尝试写入 SQLite 历史。

旧 `start_all.ps1` / `manage_local.ps1` 启动 agent 与 tools 两个进程，解释器为 Windows `.venv/Scripts/python.exe`。其中 GeoSPARQL 工具另需有效的 `GEOSPARQL_ENDPOINT`；工具服务只是该后端的 HTTP 调用者，两服务启动并不自动证明空间 RDF 后端就绪。

入口 app.py 同时负责网页静态资源、聊天、trace、历史、KG查询和运行配置注入。导入时构造 app 并初始化数据库。SQLite 使用 `data/chat_history.db` 相对路径，工作目录会影响实际落盘位置。

## 3. 真正的 harness 在哪里

`scripts/retrieve_tool_info.py` 约1700行，实际承载运行时核心：请求构造、配置与模型初始化、路线选择、KG检索、Skill规划、工具链校验、参数绑定、执行与重试、产物登记、最终答案生成和trace/usage封装。

因此当前 `scripts/` 混有生产运行时库。不能把 scripts 整体视为可随时丢弃或搬动的 pipeline。

主要分支顺序：
1. 非严格评测模式下，识别问题中的显式工具链，直接执行。
2. 识别到 GeoSPARQL 路线时，可提前走紧凑查询路径；配置默认关闭 GeoSPARQL LLM 规划，使用规则生成查询。
3. tool_only 路径尝试直接的 LLM geoscience 路由。
4. 一般路径加载 KG，检索候选工具；skill_guided 额外检索 Skill、展开能力、扩充工具候选。
5. skill_guided 且规则已推导出工具链时，schema_route 可以直接采用该链；否则调用 LLM 生成工具计划。
6. 非严格模式下，规划失败可以回退到规则推导并验证的工具链，后面还有选取首个检索工具的兜底。
7. 执行工具链，失败时反馈、重规划；已成功且与新计划一致的前缀可复用。
8. 以工具输出生成最终答案。即使规划使用规则，最终答案通常仍要求模型调用；工具证据存在不等于整个请求能够离线完成。

`GEOTOOL_EVAL_REQUIRE_LLM` 控制其中一些规则捷径和回退是否允许。当前 `.env` 文件设置 skill_guided；Config 无配置时默认 tool_only。配置文件里未设置的标志仍可能由启动进程环境注入，静态检查不能确定正在运行的进程实际值。

这是一套 KG/Skill/LLM/领域规则协同的工具链执行器。不能仅根据 skill_guided 标签或 trace 中出现 skill_plan 就断言工具链由 LLM 自主规划。

## 4. 参数生成与执行

`execute_tool()` 先取 KG schema，再用实时 catalog 的输入/输出描述覆盖旧索引信息，随后叠加科学工具与遥感契约。

普通工具先调用 `prepare_tool_input()` 尝试通过问题、输入资源、上游输出与 ArtifactRegistry 组装参数。参数足够时直接跳过 LLM；不足时由模型生成，再交给契约层修正或绑定。科学工具和 shared bridge 路线有更严格的模型参数约束。

`app/tools/tool_input_contracts.py` 约1607行，其中单个 `prepare_tool_input()` 约984行，混合大量领域分支。这是一个重要的运行时知识模块，而非简单 JSON 格式化器。

工具调用经 `app/tools/tool_runner.py` 使用 HTTP 发往 `/run-func`。工具名称和参数以字符串传输；许多工具采用 `tool(input_str) -> str` 约定，结果再解析为 JSON 或文本。ToolsAgent/tool_runner.py 的 `run_task_in_process` 实际使用 ThreadPoolExecutor，并没有使用进程池。

执行反馈层有工具尝试预算，模型客户端另有 HTTP 重试，harness 还有重规划预算，评测器又有请求重试。重构需要一个可追踪的预算体系与副作用策略，不能把各层尝试数独立扩大。成功前缀复用已经存在，需保留其语义。

## 5. Skills、能力图与工具知识图谱

运行时 Skill 定义来自 `app/skills/library.py` 的26项定义，加 mapping_extensions 的2项扩展，共28项。生成目录 `skills/` 有28份 SKILL.md/geoskill.yaml；library.json、hypergraph.json 是派生描述。直接编辑生成的 Markdown 不会自动修改当前运行时行为。

`app/skills/planning.py`：
- SkillRetriever 按 token overlap、IDF权重、关键词短语、工具提示、领域与数据类型加分排序；这部分不是向量检索。
- GeoConstraintChecker 检查输入资源与地理约束。
- SkillPlanner 将 Skill 工作流展开为能力步骤。
- ToolGrounder 将能力匹配到可执行工具及其输入约束。

`hypergraph.py` 与 `shared_bridge.py` 分别负责能力关系、Skill组合及跨工具证据绑定。生成的 hypergraph.json 记录28个Skill超边、38个能力、110个工具映射。

KG 工具检索是另外一层：`scripts/custom_kg_retrievers.py` 基于 LlamaIndex，实现向量相似度、图关系扩展以及领域/规则提示。索引目录为 KG/geotoolkg，配置当前使用 FastEmbed 多语言模型；factory 也支持远程模型与用于诊断的 local_hash。构建和读取索引时需保证 embedding 模型与维度一致。

另一个具体不一致：实时注册表有110工具，但 `scripts/build_geotoolkg.py` 的 `_load_geo_frames()` 仅使用101项 GeoEval catalog。该文件虽然定义了6个 geoscience rows，实际构建函数没有合并它们，也未合并3个 GeoTools。不能把“KG生成脚本”和“完整运行工具注册表”视为完全同步。

## 6. 工具组织与科学计算边界

实时工具数：101项领域工具，加6项 Geoscience 和3项 Geo 工具，共110项。

| 领域 | catalog 工具数 |
|---|---:|
| DeepEarth | 5 |
| GeoBigData | 1 |
| MineralResources | 4 |
| ClimateChange | 4 |
| EnergyResources | 10 |
| OceanCarbon | 10 |
| DeepTimeBiosphere | 14 |
| CCS | 15 |
| RemoteSensing | 30 |
| NaturalHazard | 8 |

各领域目录主要是 tool_name_dict 注册包装，实际大量逻辑集中于 `ToolsAgent/ToolsFuns/GeoEval/runtime.py`，约2057行；空间处理又依赖 scientific_spatial 等共享模块。

工具实现不能统一理解为现场运行完整科学算法：有的实际处理栅格/矢量，有的过滤聚合工作簿，有的查询外部后端，有的读取预先准备的 derived 样本表。例如地球模型工具读取 location_model_samples.csv；能源与海洋若干工具读取 derived 的容量、匹配点、剖面或聚合摘要。这些结果的适用范围由输入数据与构建流程决定。

因此重构必须把原始下载、预处理产物、工具现场计算、外部服务查询分别标记，保存来源与参数。规则和派生表有其作用，但不能将它们的执行记录当作未发生的完整算法重算。

## 7. 数据路径和在线/离线耦合

| 旧位置 | 当前作用 |
|---|---|
| data/benchmarks | benchmark输入资源，路径搜索根之一 |
| data/answer_formats/geotooleval_v1_1.jsonl | 按规范化问题匹配最终答案格式；运行时与评测格式耦合 |
| data/models/fastembed | 本地embedding模型缓存 |
| data/chat_history.db | 可变聊天历史，不是源码或原始数据 |
| datasets/deep_earth_toolqa/data | 模型样本、USGS source_snapshot、派生地球事件表 |
| datasets/energy_resources_toolqa/data | 能源/矿产相关原始和派生数据 |
| datasets/ocean_carbon_toolqa/data | SOCAT/GLODAP/SNAPO等派生输入 |
| _dataset_work/habitable_earth_dataset/processed | 矿产/气候相关处理产物 |
| dataset_work/extracted、dataset_work/数据集下载 | 部分工具读取的原始材料与工作簿 |
| 上级 GeoToolAgent数据集 | 旧环境配置的数据根，自然灾害工作簿等 |
| outputs/geotool_runtime | 工具运行时产物及后续工具可能使用的输入 |

旧 .env 仍有 Windows 绝对数据根。另有两套路径解析：ArtifactRegistry 的资源解析与工具 runtime 的 `_resolve_path()`。它们搜索多个目录；后者还存在文件名 token/数字/相似度匹配。这种方式让旧项目“尽量找到文件”，但同名数据、版本分叉或目录移动时可能选到不正确的资源。

特别注意在线依赖：runtime.py 通过 importlib 动态加载 `scripts/generate_energy_resources_toolqa.py` 和 `scripts/generate_ocean_carbon_toolqa.py` 的解析函数。删除这些“生成脚本”会破坏线上工具。应先将公共解析逻辑抽入 src 包，再让 pipeline 与工具共同调用。

新仓库现有 download 脚本只负责下载，符合此前要求；上述 derived/processed 资源与 KG 索引的构建仍需要独立 prepare/build pipeline。数据文件已下载或复制不代表旧框架的路径与格式自动兼容。

## 8. 配置、依赖与第三方代码

旧核心 requirements 主要是 LlamaIndex、FastAPI、OpenAI/httpx/aiohttp、pandas/openpyxl、NumPy、FastEmbed；另有 ToolsAgent requirements 与庞大的 scientific requirements。

scientific requirements 的 numpy<2、scipy<1.16 范围，与新项目当前 NumPy2.5.3/Scipy1.18.1 不一致。旧安装说明 Python3.11，新仓库当前 Python3.13。这个事实说明不能直接合并旧依赖文件；它不等价于已验证新环境运行失败。需按实际使用工具确定主依赖、可选科学后端和平台约束。

旧目录是 `.vendor/`，包含 GMT、SAIPy、SeisBench、PyGMI、安装器、wheels、测试资源和 WSL科学后端等。本次对主 app/ToolsAgent（排除历史 geose_geogpt 子工程）、scripts 与 deployment 的所查源码进行名称/路径检索，未找到这些 vendor 名称的直接引用。这只支持“不能默认全量迁入”的结论，不能证明整目录无用：外部命令、插件加载、旧实验和测试仍需单独追踪。

未来可下载源码、安装器、wheel与构建二进制应由 setup/build pipeline 管理版本与校验和；确实维护的第三方源码补丁需保留来源、许可证与补丁记录。此项目前未删除或迁入。

## 9. 评测与证据

`evaluation/run_full_evaluation.py` 是独立评测客户端：从 Excel 读取题目、输入资源、标准答案/工具链，调用 agent `/chat_trace`，保存每次HTTP尝试、raw trace、结果与进度，并计算指标。

所读请求 payload 包含问题、资源提示、模型与规划模式；标准答案在评测侧评分和结果记录中使用，没有出现在该请求 payload。不能仅因为评测文件包含 gold_answer 就认定 agent 收到了答案。与此同时，agent 运行时读取 answer_formats 的问题匹配格式约束，后续应明确其是公开输出格式协议还是评测专用配置。

现有 evaluation 中混有完整评测器、固定子集/不同baseline及结果分析代码。它们不是产品核心 harness，不能全部平移进核心包。

指标重构必须区分工具链命中、实际执行完成、最终答案正确性、模型调用过程和科学计算来源；一个总分不能替代这些证据。

## 10. 已定位的结构问题

1. 核心运行时在 scripts，而 ToolsAgent runtime 动态依赖数据生成脚本，职责边界倒置。
2. 注册表、KG索引、Skill映射、scientific/RS schema覆盖并存，101/110工具范围不一致。
3. 单文件包含多领域规则，多个千行文件共同决定一次请求，维护与迁移困难。
4. 请求参数写入全局 os.environ，运行中还修改 LlamaIndex Settings；并发请求可能互相影响配置。
5. 部分 API 捕获错误后返回普通JSON，没有明确HTTP错误状态；下游必须额外识别内容级错误。
6. ToolsAgent 上传配置在多个新建 Config() 实例上设置，Config不是单例，后续新实例不保留设置；至少这段传递机制不可靠，需根据实际使用方再修复。
7. 硬编码旧数据路径、递归与模糊资源定位，不适合可复现流水线。
8. 导入模块产生数据库初始化、输出目录创建、模型配置等副作用。
9. 多层超时和重试缺少统一请求预算，非幂等工具不能无差别重试。
10. 旧主项目与交付副本已分叉，必须固定迁移基线。

## 11. 建议的新边界（仅建议，尚未实施）

src/agentgs 下分 config、agent、llm、skills、retrieval、tools、artifacts、api、evaluation 等职责。agent 专门负责计划/执行状态机与预算；tools按领域分实现，独立 registry/schema；共享数据解析放包内，不依赖 scripts。

保留用户约定：所有测试放 src/tests/ 并按模块划分；scripts/ 放按职责划分的 CLI，如 download、prepare、build、evaluate。无需为 scripts 另写镜像测试；核心模块可用有价值的契约与流程测试。

原始数据按 data/<dataset>/ 组织，派生数据在同一数据集下明确区分 derived/processed（具体布局仍按用户后续要求）；生成索引、运行输出、历史数据库与缓存不能与源码混放。图与文档需要确定一个权威源，其余统一生成。

建议先固定工具/Skill/数据清单与迁移基线，再抽出共享解析与结构化工具契约，最后迁移 harness、API 与评测。迁移过程中保留旧证据、成功前缀复用与领域约束，避免同时改目录和算法而无法定位行为变化。

## 12. 验证边界

本次做了源码静态追踪、配置非敏感项读取、注册/生成图数量检查、新仓库状态确认和旧/交付副本文件对比。没有更改旧项目，没有重构生产代码，没有付费模型调用，没有确认旧两服务或所有110工具在新WSL环境已运行通过。
