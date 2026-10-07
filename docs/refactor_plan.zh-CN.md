# AgentGS 全面重构计划

日期：2026-10-07。状态：讨论稿；本轮只制定方案，不迁移生产代码、不安装第三方工具。依据旧 GeoToolAgent 实际源码和本仓库现状，承接 current_framework_audit.zh-CN.md。

## 一、总体决策

采纳用户的 scripts/tools、skills、client、agent、runtime、evaluation、cli 和 React 展示模块划分。增加一个很小的 contracts 公共契约层，以及 config 配置层；它们不依赖具体 agent、调度器、HTTP 框架或科学工具。

建议展示模块使用正确拼写 visualization，命令使用 `agentgs ui`。若用户要求保留 visulization，可统一替换目录名；不要同时维护两个名字。

主工作仓库 `/home/gaosong/Code/Project/AgentGS`；旧主项目只作为源材料。默认以旧主项目110工具注册面与28Skill作为迁移清单，记录交付副本101工具的差异；迁移目标是保持有效科学能力与可解释行为，不复制所有历史回退和错误实现。废弃项应记录去向，不静默丢弃。

## 二、依赖方向

contracts <- client/skills/agent/runtime/evaluation/visualization
config <- 各应用模块的启动装配
agent -> contracts、client、skills 的接口与自身检索实现；禁止导入 runtime 实现、CLI、Web。
runtime -> contracts 与执行适配器；通过 AgentFactory/AgentRunner 注入 agent，不硬编码某一个算法。
skills -> contracts；使用能力接口请求工具/模型/产物，不直接启动进程、不读全局环境、不导入具体 scheduler。
evaluation、visualization -> TaskService/事件与结果契约，不调用 agent 私有函数。
cli -> 统一装配各模块，是依赖具体实现的组合入口。

基础层只包含确实共享的协议与模型，禁止变成无边界的 utils/common。类型系统用于接口契约、运行时校验和可序列化数据；不为了泛化给每个类建立继承体系。

## 三、目录蓝图

```text
AgentGS/
├── pyproject.toml
├── uv.lock
├── llm.toml                      # provider/model/角色配置，无秘密
├── runtime.toml                  # 队列、预算、资源与沙箱配置
├── .env                         # 本机环境变量，不入Git
├── .env.example                 # 变量名与占位值
├── docs/
├── scripts/
│   ├── download/                # 保留已有22组下载CLI与清单
│   ├── tools/
│   │   ├── _common.py
│   │   ├── manifests/           # 安装源、版本、平台、hash、许可证
│   │   ├── install_bwrap.py
│   │   ├── install_proot.py
│   │   ├── install_<tool>.py
│   │   ├── install_all.sh
│   │   └── install_all.ps1
│   ├── prepare/
│   │   ├── prepare_deep_earth.py
│   │   ├── prepare_energy.py
│   │   ├── prepare_ocean.py
│   │   ├── prepare_geosparql.py
│   │   └── prepare_<dataset>.py
│   ├── build/
│   │   ├── build_tool_index.py
│   │   ├── build_skill_index.py
│   │   ├── export_skill_docs.py
│   │   └── build_frontend.py
│   └── evaluation/
│       ├── run_geotooleval.py
│       └── summarize_results.py
├── src/
│   ├── agentgs/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── contracts/
│   │   │   ├── task.py
│   │   │   ├── agent.py
│   │   │   ├── skill.py
│   │   │   ├── tool.py
│   │   │   ├── artifact.py
│   │   │   └── events.py
│   │   ├── config/
│   │   │   ├── paths.py
│   │   │   ├── settings.py
│   │   │   └── env.py
│   │   ├── client/
│   │   │   ├── base.py
│   │   │   ├── models.py
│   │   │   ├── config.py
│   │   │   ├── openai_compatible.py
│   │   │   ├── adapters/
│   │   │   └── usage.py
│   │   ├── skills/
│   │   │   ├── base.py
│   │   │   ├── models.py
│   │   │   ├── registry.py
│   │   │   ├── validation.py
│   │   │   ├── earthquake_catalog/
│   │   │   │   ├── skill.py
│   │   │   │   ├── models.py
│   │   │   │   └── prompts/
│   │   │   ├── geosparql_query/
│   │   │   └── <其他具体Skill>/
│   │   ├── agent/
│   │   │   ├── base.py
│   │   │   ├── engine.py
│   │   │   ├── state.py
│   │   │   ├── planning/
│   │   │   ├── retrieval/
│   │   │   │   ├── base.py
│   │   │   │   ├── skills.py
│   │   │   │   ├── tools.py
│   │   │   │   └── graph.py
│   │   │   ├── execution/
│   │   │   │   ├── bindings.py
│   │   │   │   ├── feedback.py
│   │   │   │   └── recovery.py
│   │   │   └── answering/
│   │   ├── runtime/
│   │   │   ├── service.py
│   │   │   ├── scheduler.py
│   │   │   ├── worker.py
│   │   │   ├── resources.py
│   │   │   ├── tools/
│   │   │   │   ├── registry.py
│   │   │   │   ├── executor.py
│   │   │   │   ├── adapters/
│   │   │   │   └── domains/     # 内部科学工具按领域组织
│   │   │   ├── sandbox/
│   │   │   │   ├── base.py
│   │   │   │   ├── policy.py
│   │   │   │   ├── bwrap.py
│   │   │   │   ├── proot.py
│   │   │   │   └── windows.py
│   │   │   ├── artifacts/
│   │   │   ├── storage/
│   │   │   └── backends/       # 单任务进程；未来池化等适配
│   │   ├── datasets/
│   │   │   ├── registry.py
│   │   │   └── <领域>/         # 在线/离线共享的解析与变换函数
│   │   ├── evaluation/
│   │   │   ├── models.py
│   │   │   ├── runner.py
│   │   │   ├── datasets/
│   │   │   ├── metrics/
│   │   │   └── reports/
│   │   ├── cli/
│   │   │   ├── app.py
│   │   │   └── commands/
│   │   └── visualization/
│   │       ├── backend/
│   │       │   ├── app.py
│   │       │   ├── routes/
│   │       │   └── dependencies.py
│   │       └── frontend/
│   │           ├── package.json
│   │           ├── package-lock.json
│   │           ├── vite.config.ts
│   │           └── src/
│   │               ├── api/
│   │               ├── pages/
│   │               ├── components/
│   │               └── types/
│   └── tests/                   # 所有自动化测试，按模块划分
│       ├── contracts/
│       ├── client/
│       ├── skills/
│       ├── agent/
│       ├── runtime/
│       ├── evaluation/
│       ├── visualization/       # 含前端测试，前端配置指向这里
│       └── fixtures/
├── tools/                       # 整体Git忽略
│   ├── bin/                     # 激活后的二进制或启动器
│   ├── packages/<name>/<version>/<platform>/
│   ├── envs/                    # 独立uv环境，避免科学栈冲突
│   ├── share/                   # 必需支持文件，不塞进bin
│   └── state/                   # 安装收据、能力探测
├── data/<dataset>/{raw,processed}/
├── outputs/tasks/<task_id>/
├── outputs/evaluations/<run_id>/
└── tmp/
```

目录是目标职责蓝图，不一次性创建空模块。新增 datasets 层是因为旧能源/海洋工具直接加载生成脚本：解析逻辑必须有包内归属，不能重新依赖 scripts。

## 四、Skill与Tool契约

SkillSpec：稳定id、版本、说明、输入输出schema、能力标签、约束、所需工具/数据集与prompt资源。SkillContext：只给该Skill需要的模型端口、ToolExecutor、ArtifactStore、事件端口和执行预算。Skill具有 validate_input、build_plan/prepare、execute、validate_output 等必要接口；纯prompt Skill或组合Skill可以提供默认实现，不强迫每个Skill实现无用方法。

Skill不内置向量库或KG算法。它提供可检索的descriptor和约束；agent/retrieval中的SkillRetriever接口实现关键词、向量或混合检索，返回统一SkillMatch。组合能力的扩展、选择和bridge规划放在agent/planning；具体Skill执行与领域约束在skills内部。

Tool是原子操作；Skill是任务能力或工作流。28Skill不应机械展开成110个工具Skill。内部工具实现按领域放runtime/tools/domains；某个Skill专属的绑定/校验留在其目录，跨Skill可复用操作只保留一份。

ToolSpec：稳定id、输入输出schema、effects、资源需求、backend/data依赖、timeout和retry_policy。ToolCall/ToolResult结构化，错误分validation、transport、dependency_missing、domain、timeout、resource_limit等；科学指标中的error字段不能自动当作执行失败。迁移期只在旧工具边界保留str->str适配器，新内核不继续字符串拼JSON。

ArtifactRef含id、类型、来源、hash/元数据；结果不能靠扫描上一条回答文本找路径。数据定位通过dataset_id/version/relative_path和清单；拒绝模糊文件名搜索。CRS、nodata、时间窗、场景配对与单位等科学约束保留并逐域迁移。

## 五、agent 与 runtime 的标准边界

AgentRunner提供异步run(TaskInput, AgentContext)->AgentResult；Context注入ModelClient、SkillRegistry、Retriever、ToolExecutor、ArtifactStore、EventSink和预算/取消状态。agent不读取.env、不创建工作目录、不启动bwrap、不修改LlamaIndex全局Settings。

Runtime TaskService提供 submit/get/cancel/list/subscribe/result；接收可序列化TaskSpec、分配task_id、排队、装配依赖、创建工作区、启动worker、持久化事件与结果。CLI、Web和evaluation均调用这个服务接口。本地直接实现先完成，未来远程和池化通过backend替换。

agent中保留规划/选择策略，runtime执行策略要求的操作。Runtime的ToolExecutor做实际调用/资源隔离；agent接收结果并决定修复或重规划。RetrievalRequest/Result同样通过接口，索引算法在agent，索引文件读取和资源租赁由运行时提供基础能力。

Event为可序列化的版本化模型：task_id、run_id、sequence、时间、类型、payload；包含plan.created、model.call、skill.started、tool.completed、artifact.created、task.failed等。进程间传事件数据，不传Python回调对象。应用侧可将事件绑定回调；普通展示/日志订阅者失败不能改变任务结果，需要影响执行的验证钩子必须显式声明契约。

## 六、任务调度、资源和沙箱

首版每个执行中的task一个独立worker进程，进程启动采用spawn兼容模式；模型请求用async，科学/代码操作交给可控子进程。禁止用不可终止的线程承担需要硬限制的代码执行。池化接口预留，首版不同时开发多套调度器。

状态：queued -> preparing -> running -> succeeded/failed/cancelled。资源不足维持排队；资源限额失败记录独立错误原因。task内部Skill/Tool有单独step状态。取消/超时要终止整棵进程树，不仅停止等待HTTP响应。

资源配置：max_concurrent_tasks、max_concurrent_tool_processes、每任务与全局memory预算、RSS软阈值、wall time、模型调用预算、子进程数、输出字节数。BLAS/OMP线程数在worker启动时限制，避免任务并发乘以内部并行。支持task资源profile，重型地理模型不套用轻工具的内存参数。并发与内存具体默认值在目标机实测后确定。

RSS用于进程树观测和软阈值处理。Linux cgroup v2 memory.max/pids.max提供可用时的强限制，但memory.max限制的是cgroup计费内存，并非精确RSS；没有委托权限时记录监测模式与限制能力。RLIMIT_AS约束虚拟地址空间，不能冒充RSS上限。

代码执行沙箱和agentworker分开：模型凭据仅在宿主/agentworker模型客户端使用，生成代码进程采用环境变量白名单，不继承.env密钥。输入只读挂载，输出只允许task目录，显式网络策略。普通注册科学工具也有资源/输出约束，但不将所有LLM网络请求塞进代码沙箱。

Linux：探测实际运行能力后首选bwrap。定义最小只读运行环境、独立tmp、进程/网络策略及允许输出挂载；不把整个repo含.env挂进去。可用的cgroup限制由supervisor施加。PRoot作为兼容回退，额外配进程监督；它不能提供与bwrap等价的网络与内核隔离。TaskPolicy要求的能力若回退后无法满足则拒绝该代码步骤，不能静默降级。

Windows：用户当前项目优先通过WSL Ubuntu-24.04后端执行Linux科学工具和生成代码；保留原生backend，用Job Object管理进程树/内存与超时，结合restricted token或AppContainer及工作区ACL。Job Object本身不是文件/网络沙箱，commit limit不等于RSS。若原生隔离能力未满足任务要求，路由到WSL或报告unsupported；跨平台支持不要求所有第三方工具都有原生Windows构建。

## 七、第三方工具安装

scripts/tools/install_<name>.py保持与download CLI类似风格：项目解释器、pyproject根定位、项目根相对默认输出、--help/--version/--verify/--force按职责提供。安装流程幂等：探测平台 -> 检查已安装版本与hash -> 下载暂存 -> 校验 -> 解包/构建 -> 原子安装 -> 激活bin入口 -> 写收据并执行最小功能探测。

manifest记录来源、固定版本、平台/架构、sha256、许可证、构建配方和系统依赖；拒绝未经记录的latest。仅当必要才下载源码构建，源码暂存tmp/并由配方重现，不提交vendor。工具自身运行必需的库/数据放packages/share；tools/bin仅承载可执行入口。Windows不依赖必须管理员权限的软链接，可使用.exe或.cmd启动器。

Python主框架依赖由uv.lock管理，不放tools/bin。需要不同Python版本或科学库范围的外部Python工具使用tools/envs隔离环境。bwrap的可运行性、某些系统库并不由复制一个二进制解决；doctor应检测并报告宿主要求，不让安装脚本悄悄修改系统。

安装all按声明的profile依赖顺序执行，默认只安装启用能力需要的工具；当前.vendor没有建立调用证据的工具先不迁入。用安装/构建清单保留确有用途的第三方补丁与来源，而不是全量拷贝历史目录。

## 八、LLM配置与client

client统一OpenAI兼容消息、工具调用、结构化输出、流式响应、错误和usage模型；使用一个复用连接的异步客户端，provider适配器封装参数差异。首版对接OpenAI兼容chat completions，协议端点差异不泄漏进agent。embedding另外定义端口，供检索使用。

llm.toml分providers、models、roles；provider定义base_url/api_key引用、协议和限流；model定义远端model id、context_window、max_output_tokens、supported_efforts、工具/JSON/图像/stream能力；role将planner/argument/answer/embedding映射到模型。

环境引用语法建议`${VAR}`，只对显式字段插值，未定义变量启动校验失败，禁止执行表达式。当前尚未确认用户所指grok仓库，配置方案只按已给要求设计，不声称复刻其格式。示意（数字仅为语法示例，不是任何实际模型参数）：

```toml
schema_version = 1
[providers.primary]
protocol = "openai_compatible"
base_url = "${LLM_BASE_URL}"
api_key = "${LLM_API_KEY}"
[models.planner]
provider = "primary"
model = "${LLM_PLANNER_MODEL}"
context_window = 128000
max_output_tokens = 8192
supported_efforts = ["low", "medium", "high"]
default_effort = "medium"
[models.planner.capabilities]
tool_calling = true
streaming = true
structured_output = true
[roles]
planner = "planner"
answer = "planner"
```

.env保存本地变量，llm.toml不存明文秘密；.env.example只有占位说明。启动时加载一次不可变Settings，进程环境优先于.env；不通过每个HTTP请求改os.environ。模型effort不支持时应明确报错或按已声明policy处理，不能静默忽略；支持extra_body的provider细节由适配器承接。

模型context元数据用于预算，tokenizer未知时标记估计而非声称精确。usage未返回保持unknown/nullable，不记为零。重试仅处理合适的传输/限流错误，遵守全局调用预算与取消；模型重试不自动重放已成功工具。

开发默认从明确project_root/llm.toml读配置；repo root通过pyproject识别，支持--project/--llm-config。安装成包后不假设源码文件的parents能找到用户项目。

## 九、CLI、展示与评测

CLI建议Typer+Rich：agentgs run、task list/show/cancel、skills list/show、tools list/doctor、config validate、evaluate run/report、ui serve。脚本负责安装/下载/构建流程；CLI调用src中的应用服务，不重复算法。JSON模式给自动化，Rich输出给交互，日志走stderr，稳定退出码。

展示：FastAPI backend提供task资源、产物、结果与SSE事件订阅；首版单向进度使用SSE，确需双向交互时再加WebSocket。React+TypeScript+Vite frontend只调用公开API，包含任务队列、规划/执行时间线、产物预览、模型usage与评测报告。事件sequence支持断线重连；SQLite存任务元数据/事件，较大trace与产物放outputs。前端不接触apikey和agent私有对象，后端通过TaskService注入。

Pydantic/JSON Schema生成前端类型，避免三套Task模型。frontend源与Node锁文件入Git；node_modules和dist忽略，构建脚本负责打包。wheel包含所需prompt资源与已构建静态资源，不能遗漏非.py文件。所有测试包含前端测试集中在src/tests对应模块，避免散落另建tests。

评测TaskInput只含问题/输入资源/公开输出要求。gold_answer、gold_chain放评测私有对象，不传agent。benchmark输出格式通过明确OutputContract输入，不保留生产代码按题目偷偷查评测jsonl的耦合。

指标分开记录任务运行状态、工具链匹配、答案评分、领域证据/来源、模型usage/重试/耗时。保存数据版本、配置脱敏快照、prompt/Skill/tool版本、代码revision与原始尝试；断点续跑不得覆盖既有失败证据。评测调用统一TaskService，未来远程通过同一契约适配。

## 十、旧代码迁移映射

| 旧来源 | 新职责 |
|---|---|
| scripts/retrieve_tool_info.py | agent engine/planning/execution/answering，runtime装配 |
| scripts/custom_kg_retrievers.py | agent/retrieval |
| scripts/generate_kg_index.py、build_geotoolkg.py | scripts/build薄CLI + agent/retrieval构建函数 |
| app/skills/library、models、planning、hypergraph | skills规格与agent检索/能力规划，分离算法 |
| app/skills/shared_bridge.py | agent/planning组合与agent/execution绑定 |
| app/skills/artifacts.py | contracts的引用模型 + runtime/artifacts存储与解析 |
| app/tools/tool_input_contracts.py | 按领域拆进skills绑定或共享工具契约，禁止搬成同样巨型文件 |
| app/tools/execution_feedback.py | 结构化ToolResult、runtime尝试状态与agent恢复策略 |
| app/llms/api_clients、usage_tracker | client |
| ToolsAgent/ToolsFuns/GeoEval/runtime.py | runtime/tools/domains |
| tool registry/catalog/scientific schema | 一个ToolSpec权威源，其余由它导出 |
| generate_energy/ocean中的共享解析 | datasets；prepare CLI调用它 |
| app.py、chat_store、旧web | visualization/backend、runtime/storage、React前端 |
| evaluation各runner/scorer | evaluation核心 + scripts/evaluation薄CLI |
| 旧vendor、安装包、二进制 | scripts/tools可重现安装流程，输出tools |

原始下载与派生数据分开；prepare记录原始输入hash、参数和处理版本；build记录embedding/schema版本。工具明确报告来源类型raw_compute/derived_lookup/external_backend，查询派生表不能冒充现场完整模型计算。

## 十一、实施阶段与验收

0. 固定迁移清单：110工具、28Skill、数据与外部backend依赖；记载两个旧版本差异，得到每项能力的迁移状态。保留旧项目，只写新仓库。
1. 基础：contracts、config、client、TyperCLI、uv依赖分组与secret模板。验收配置校验、环境引用、模型能力检查、包安装及资源包含；mock验证client协议，无需付费调用。
2. Runtime纵向链路：TaskService -> taskworker ->一个实际简单工具->artifact/result，做queue/cancel/events/resource与bwrap探测。验收多个任务隔离、进程树取消、限制能力与回退记录；具体RSS阈值通过实测选定。
3. Skill+agent闭环：先选真实地震目录任务和一条可多步验证的空间/栅格任务，不迁移全部巨型模块。打通检索->计划->绑定->工具执行->最终答案，观测每个决定来源。验收真实算法输出、约束错误、恢复与成功前缀复用。
4. 全领域迁移与pipeline：逐域替换旧runtime，数据清单/prepare/build、第三方安装profile并行按职责推进。每项工具依赖要有可用/缺失明确状态；KG/Skill索引由统一registry生成，消除101/110不一致。
5. Evaluation：导入真实样本，核验评分与断点证据；全量付费评测是单独实验，不作为日常重构测试默认执行。
6. React展示与交付：API契约驱动前端、SSE重连、错误与产物展示；WSL/Linux主路径验收及Windows路由/原生能力边界。前端构建、uv安装、prompt资源、脚本help/verify人工检查，清洁环境复现后才能称交付可用。

每阶段交付可运行闭环，不先搭一堆空抽象再一次性搬旧代码。核心测试全部src/tests，包含有意义的流程与领域契约；scripts不单独写镜像测试，按用户要求人工验证CLI与实际输出。采用依赖边界测试避免agent偷偷依赖runtime/展示。

## 十二、Git与依赖管理

源码、prompt、非敏感配置、清单、uv.lock、frontend锁文件和小型测试fixture入Git。忽略tools、.env、node_modules、dist、数据、输出、tmp、forks与cache；大原始/派生文件和索引通过流水线构建。测试fixture可在src/tests/fixtures包含精简原始输入，但不能携带生产评测gold进入agent。

uv分主框架、scientific、visualization、dev等可选extras/groups，不把历史TensorFlow/GemPy/PyGMI等默认塞入核心。确需互斥科学栈的工具独立env。旧numpy/scipy限制与新环境冲突需逐工具验证再定版本，迁移前不盲目降级整个仓库。

## 十三、仍待确认但不阻碍框架设计

用户所指grok的config.toml仓库/文件尚未明确。确认后对齐实际语法；provider/model分离、环境引用与能力校验的要求已经纳入。默认提案是保留原有能力清单、以WSL/Linux为科学运行主路径、Windows通过WSL桥接并逐步提供可满足契约的原生能力。展示目录建议visualization。

## 技术依据

- Typer子命令与Rich：https://typer.tiangolo.com/tutorial/subcommands/ 与 https://typer.tiangolo.com/tutorial/printing/
- bubblewrap隔离依赖调用方定义策略：https://github.com/containers/bubblewrap/blob/main/README.md
- PRoot基于ptrace并共享宿主内核资源：https://proot-me.github.io/
- cgroup v2资源语义：https://docs.kernel.org/admin-guide/cgroup-v2.html
- Windows Job Objects：https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
- Windows restricted tokens：https://learn.microsoft.com/en-us/windows/win32/secauthz/restricted-tokens
