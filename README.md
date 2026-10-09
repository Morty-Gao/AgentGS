# AgentGS

标准 Python 地学 Agent 项目。当前范围为 **101 项 GeoToolEval 工具、24 个 Skill、31 个 Capability**，默认执行原版的分阶段流程。核心源码以原主项目冻结版本为基线；生产包独立于旧源码目录。

## 当前重构状态

101 项工具均有实际处理函数，已在科学复现环境逐项执行成功。100 项科学工具使用同一输入与旧源码对照一致；GeoQuestions 按用户授权保留 40 条物化结果的 source_id / 查询结构匹配协议，另有 125 项独立回放对照。已迁入能源、海洋共享解析器、空间/栅格科学操作及相应数据定位。

默认 Agent 已连接 Skill 检索、KG 检索、规划、桥接、grounding、参数绑定、原子反馈、重规划、成功前缀复用、最终模型回答和非修改性验证。规划通过标准 function call 提交 `submit_tool_plan`，每个科学工具使用独立的 JSON 参数 schema；已移除 `algorithm=prototype` 入口。24 项 Skill 是动态能力声明，不代表 24 条固定工具链。

已验收的独立对照包括 341 项确定性组件决策、16 项原 KG/SDK 检索、298 项参数绑定、65 组科学结果和产物、2,412 项原 FAS 分数。保存实验答案的 28 组 original / compatibility 复评一致。真实 TaskService/worker 已验证失败恢复、六步空间链和跨领域产物传递。

**尚未完成旧 harness 所有入口、模式与桥接组合的同响应回放验收，因此 `runtime_equivalent=false` 保留。** 这表示全链路等价证据尚不充分；迁移清单不再把已实现模块标成占位。完整证据与未验收范围见 [algorithm_alignment.json](docs/algorithm_alignment.json) 和 [migration_inventory.json](docs/migration_inventory.json)。

## 十领域抽样交付验收

按用户最新要求，本地完成10个领域各1题的实际Agent抽样，全部任务成功、冻结FAS=1；模型gpt-5.6-terra，共22个模型阶段。样题为Q61、Q106、Q141、Q199、Q287、Q347、Q447、Q471、Q496、Q544，其中遥感/自然灾害包含多步链。科学复现环境完整测试714项通过、无跳过。结果汇总为 `outputs/verification/delivery.json`，逐题请求、trace、用量和评分保留在 `domains-1-5.json` 与 `domains-6-10.json`。

603题Agent端到端实验按用户决定在服务器运行。本地的2412项评分oracle检查是离线评分验证，未执行603题模型实验；`runtime_equivalent=false`保留为更广泛形式验收边界，不阻碍本次约定的十领域抽样交付。

## 环境准备

运行路径为 Linux / WSL，科学复现环境使用 Python 3.12，NumPy 1.26.4、SciPy 1.15.3 等原分发版本。默认开发环境与复现环境分别管理。

```bash
cd /home/gaosong/Code/Project/AgentGS
uv sync --locked --extra visualization
.venv/bin/python scripts/tools/install_reproduction.py --python 3.12
.venv/bin/python scripts/tools/install_reproduction.py --verify
```

整理结果统一放在 `data/`：单文件直接保存，多文件按数据集目录管理。超过 64 MiB 的多文件数据集额外压缩为 ZIP，由对应 prepare 脚本负责校验和展开；展开结果也在 `data/`，原始下载与临时文件仅放在 `data/raw/`。GeoToolEval 使用刻意纳入 Git 的 `data/geotooleval.jsonl`，保留原工作簿的题目、答案、资源、工具链和格式列。其余可生成数据通过 `scripts/prepare/` 从可用原始来源准备，默认不会使用 Git 管理的结果快照。

```bash
tools/envs/reproduction/bin/python scripts/prepare/prepare_all.py
# 使用已下载的固定原始数据与本地原附件
tools/envs/reproduction/bin/python scripts/prepare/prepare_all.py --offline
# 首次使用保留的原附件时明确指定来源
tools/envs/reproduction/bin/python scripts/prepare/prepare_all.py --source-root data
```

各入口命名为 `prepare_<领域>.py`，总入口为 `prepare_all.py`；GeoQuestions 固定 40 条数据由独立准备脚本生成并被 Git 忽略。回答格式直接保存在 `data/geotooleval.jsonl` 中，无需额外准备。矿产从 MRDS 按原 8,000 个记录 ID 重建，海洋和能源从原始文件计算，遥感保留固定 49 个目录的原始影像内容。GISTEMP 年表由 NASA 原始月表生成；World Bank 旧原始文件缺失时直接从当前 API ZIP 重建。来源更新允许改变科学结果，每次记录实际源文件哈希。BioDeepTime 原子集/手工区界、CCS 手工空间资产及原灾害表仍需要提供原输入。路径、来源与可选的显式快照恢复模式见 [prepare/README.md](scripts/prepare/README.md)。

Git 保留 GeoToolEval、GEM 压缩 JSONL、固定数据参考资料、暂不能完整重建的少量原始资产、深地固定输入以及用户提供的冻结 KG JSON。可生成的处理结果、公开原始下载、大型模型及缓存均被忽略；来源哈希与选择参数放在准备脚本的 `manifests/`。数据准备通过不等同于真实模型端到端实验通过。

用户提供的 `geotoolkg.zip` 和 `fastembed.zip` 已恢复到 `config.toml` 声明的 `data/geotoolkg/` 与 `data/raw/models/fastembed/`。索引覆盖全部 101 个工具，缓存模型离线生成 384 维向量；十领域检索通过，抽查五条索引关系的重新编码向量与原向量完全一致。来源包和逐文件 SHA-256 保存在 `data/geotoolkg/import_manifest.json`。轻量 KG JSON 可纳入 Git，大型 embedding 权重仍为本地资源；部署到其他机器时需迁移该缓存。

本机 `.env` 已配置用户提供的模型连接，选择 `gpt-5.6-terra`，密钥被 Git 忽略。2026-10-08 服务器连接验收确认模型列表可访问，Q287 的真实 Agent 完成规划、`EnergyCapacityValue` 工具调用与最终回答，输出 `620.399 MW`；2 个模型阶段，FAS=1、PAS=0.8，严格参数成功门未通过。报告保存在 `outputs/verification/live-server-20261008-q287.json`。这是一题连接和执行验收，尚未启动服务器的 603 题完整实验。项目依赖包含 HTTPX 的 SOCKS 支持，适用于本机代理环境。

随后按用户要求，服务器用 `gpt-5.6-terra` 将十领域各一题同时并行执行，固定选择 Q61、Q106、Q141、Q199、Q287、Q347、Q447、Q471、Q496、Q544。2026-10-08 本次十题全部执行完成，8 题 FAS=1，平均 FAS=0.855，共 22 个模型阶段、耗时约 52 秒；worker 事件证实峰值同时运行 10 个进程。Q106 最终回答省略五条自治市记录，FAS=0.216667；Q199 返回正确温度但省略 age 和结构字段，FAS=0.333333。严格参数成功门通过 2/10，不将执行成功视为所有评分项通过。完整回答、工具产物、事件、模型用量和评分保存在 `outputs/verification/parallel-domains-20261008.json`，未按成绩重跑或替换样题。

十领域抽样的独立辅助脚本已移除，历史报告保留。新抽样实验统一使用 `agentgs evaluate benchmark --ids` 指定题号，并使用新的输出目录。

用户随后授权启动服务器全部 603 题的真实 Agent 实验，输出目录为 `outputs/evaluations/server-603-max128-20261008T145211Z/`，模型 `gpt-5.6-terra`。本次使用运行器上限 128 并发，每题限时 600 秒、RSS 上限 2,048 MB，总 RSS 预算 256 GiB。评测入口采用有界并行调度；运行资源通过 CLI 参数指定。`experiment.log` 和 `experiment.events.jsonl` 持续记录进度，`progress.json` 保存当前计数，每题的 `attempt_*` 保存公开请求、task ID、完整结果、事件、stderr 和评分，产物保存在对应 `outputs/tasks/<task_id>/`。`launch.json` 记录监督进程与执行进程，`exit.json` 在执行进程结束时记录退出状态。

本轮已于 2026-10-08 完成，监督进程耗时 653.176 秒，运行进程退出码 0。603 题均保留首次 attempt：512 题执行成功、91 题失败（85 题算法返回未完成、5 题最终回答生成 TypeError、1 题 worker 事件 JSONDecodeError），没有超时或 RSS 超限。平均 FAS=0.698711，十领域宏平均 FAS=0.621294，333/603 题 FAS=1；严格参数成功门通过 48/603，执行成功、答案评分与参数评分分别报告。1386 个模型阶段共报告 4,123,350 tokens，其中一题用量记录不完整，不将缺失用量补零。`run_audit.json` 已核对 603 个题号、每题六类记录、603 条预测、运行前后的源码/数据身份，以及 worker 时间线的 128 并发峰值；日志和报告未发现模型密钥。完整评分保存在 `summary.json`，未自动重跑失败题。

```bash
# 新实验：监督进程保存完整控制台日志及退出状态。
tools/envs/reproduction/bin/python scripts/evaluation/run_logged_benchmark.py --output outputs/evaluations/server-603-new --concurrency 128 --timeout 600 --rss-limit-mb 2048
# 实验结束或中断后，使用相同源码、数据和配置续跑尚未完成的题目：
tools/envs/reproduction/bin/python -m agentgs evaluate benchmark --output outputs/evaluations/server-603-max128-20261008T145211Z --concurrency 128 --timeout 600 --rss-limit-mb 2048
```

默认续跑保留失败结果；需要重试失败题时显式加 `--retry-failed`，原 attempt 仍保留。若中断后留下 `.run.lock`，先确认 `launch.json` 的执行进程已退出，再移除该锁；运行中的目录不得并发续跑。十领域抽样与本次全量实验分别保留，均不作为旧 harness 所有模式完全等价的证据。

用户确认的最新版 `GeoToolEval_固定回答格式.xlsx` 已一次性转换为 `data/geotooleval.jsonl`，包含 603 题、11 个原始字段，完整保留“固定回答格式”和“格式说明”；全部单元格值与类型已逐字段核对。每条记录额外包含 `question_sha256` 和 `answer_format`，供运行时直接提取公开回答要求；不再维护独立的回答格式文件。英文通用提示词保存在 `src/agentgs/datasets/geotooleval.py`。原文件 SHA256、领域题数、输出哈希及另外两个工作表的内容统一保存在 `data/metadata/geotooleval.json`（保留公式文本及缓存值，不执行公式）。主数据及来源说明均可由 Git 跟踪，不保留转换脚本或工作簿副本。

## 最新数据与算法版本

内嵌数据以用户最后上传的 `GeoToolEval_固定回答格式.xlsx` 为准。与前次上传相比，新增两列回答格式字段，75 题的标准工具调用明细发生变化；原题文本、答案及其他原始字段一致。来源与变更记录保存在 `data/metadata/geotooleval.json`。工作簿中的格式列按原文保留，运行时和评测读取同一个 JSONL，回答格式配置与公开列一致性及题目哈希在加载时校验。

GeoQuestions 只按公开 numeric source_id 或提交的 SPARQL 结构定位物化结果，符号 topic 不充当 ID。查询匹配保留原权重、阈值和歧义规则，未匹配返回失败。结果标记 `derived_lookup`、`answer_lookup=true`，不声称执行了 RDF 端点。

`config.toml` 的 `[agent]` 管理算法开关、预算与检索路径；`llm.toml` 独立存储模型调用信息（provider、model 和角色映射）；运行资源与沙箱通过 CLI 参数指定；`.env` 保存本机连接信息并被 Git 忽略。模型模式需要 `LLM_BASE_URL`（含 `/v1`）、`LLM_API_KEY`、`LLM_MODEL`。用量缺失、失败或中断时保留 unknown，成本和综合分不可据此补零。

## 工具规划消融

`planning_mode="tool_only"` 使用同一 AgentEngine，按 SciToolAgent 的分阶段结构执行：工具 KG 检索 → 模型提交工具链 → 逐步生成参数、执行并修复 → 模型回答。该模式不加载 Skill 定义、Capability 扩展或 planner，不使用能力补全、共享桥接、Skill grounding 或对应提示，规划阶段记录为 `tool_planning`。工具范围由 `contracts/tool_scope.json` 独立声明；任务级模式覆盖在 worker 初始化之前解析。

两组共享 101 个工具、标准 function call、KG 检索、工具级领域 seed/提示、公开资源绑定、产物追踪、单工具修复、重规划、回答格式和评分。工具级规则不读取 Skill/Capability 声明。消融测量整个 Skill/Capability 规划层及其候选扩展、补全和桥接的收益；这是在 AgentGS 框架内对齐 SciToolAgent 的工具层基线，不是旧代码的严格复现。

旧 `require_llm` 参数已移除，配置和任务 JSON 使用以下字段；问答模式始终需要模型，参数和最终回答始终由模型生成。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `allow_rule_planning` | `false` | 是否允许规则直接路由、规划失败回退或修正工具链；不控制模型是否调用 |
| `plan_attempts` | `3` | 整链规划总次数，包括执行失败后的重规划；与规则开关独立 |
| `tool_attempts` | `3` | 每个工具实例的参数与执行修复总次数 |
| `argument_attempts` | `3` | 原生规划调用的协议修复预算及参数请求的传输重试上限 |
| `model_attempts` | `4` | 模型请求的传输重试预算 |

旧 `require_llm=true` 的规划策略对应 `allow_rule_planning=false`；旧 `require_llm=false` 对应 `allow_rule_planning=true`。本次默认统一关闭规则规划；若要保留旧配置的规则辅助，需要显式开启。规划次数单独指定，不再因为关闭规则规划而降为一次。`shared_bridge` 在工具模式下有效值固定为 `false`。

正式对比只改变模式，使用相同题目、模型和预算，并分别保存到新实验目录：

```bash
tools/envs/reproduction/bin/python -m agentgs evaluate benchmark --planning-mode skill_guided --no-rule-planning --plan-attempts 3 --tool-attempts 3 --output outputs/evaluations/full-new
tools/envs/reproduction/bin/python -m agentgs evaluate benchmark --planning-mode tool_only --no-rule-planning --plan-attempts 3 --tool-attempts 3 --output outputs/evaluations/tool-only-new
```

日志监督脚本也支持上述模式和预算参数。实验协议记录实际生效的任务覆盖、恢复预算及工具/规划声明哈希；每个 AgentResult 的 `provenance.effective_algorithm` 保存有效配置。旧参数、源码或协议的实验不能混入新目录续跑。

运行时存储对历史任务做只读字段转换，保留旧规则策略及原规划预算，历史结果仍可查看；转换不会覆盖原数据库记录，也不会接受旧参数作为新的任务输入。

## 启动与示例

`run`、`evaluate run`、`evaluate benchmark` 和 `ui serve` 使用以下运行参数，不读取运行配置文件。未传参数时使用代码默认值：

| CLI 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--concurrency` | 2 | 最大并发任务数（1–128） |
| `--model-concurrency` | 8 | 同一项目内并发模型 HTTP 请求上限 |
| `--tool-concurrency` | 4 | 同一项目内并发工具执行上限 |
| `--queue-capacity` | 1024 | 已接收但尚未结束的任务上限，满时 API 返回 429 |
| `--model-timeout` | 180 | 单次模型调用总期限，包含排队、重试和退避（秒） |
| `--tool-timeout` | 120 | 单次工具调用总期限，包含排队和计算（秒） |
| `--timeout` | 300 | 每题总期限，包含排队、准备和执行（秒） |
| `--rss-limit-mb` | 1024 | 每题 RSS 内存上限（MB） |
| `--max-total-rss-mb` | 4096 | 全部 worker 的 RSS 总预算（MB） |
| `--code-timeout` | 20 | 代码步骤超时秒数 |
| `--sandbox` | `auto` | `auto`、`bwrap`、`proot` 或 `disabled` |
| `--allow-proot` / `--no-allow-proot` | 禁用 | 显式允许 PRoot 后端 |
| `--worker-python` | 自动选择 | 指定 worker Python 的绝对路径或项目相对路径 |

显式设置并发或每题 RSS 上限时，未指定的总 RSS 预算会自动提高到至少“并发 × 每题 RSS”；显式指定 `--max-total-rss-mb` 则使用该值。worker 默认优先使用项目的科学复现环境，缺失时使用当前 Python。

任务、模型和工具的额度通过 `outputs/runtime/coordination.sqlite3` 共享。相同项目根目录的多个 runtime 实例和 worker 使用同一额度，活动实例必须使用一致的并发配置；跨实例可以查询、等待和取消任务。科学内置工具默认通过独立子进程计算，超时或取消时回收进程树。同步扩展 handler 在线程中执行，异步 handler 直接等待；线程中的同步函数取消后会等待其退出，因此需要硬终止的计算应使用进程后端。

Python 运行时 API 的提交、查询和事件读取均需要 `await`，可以用异步上下文管理器关闭服务：

```python
from pathlib import Path
from agentgs.config.settings import RuntimeSettings
from agentgs.contracts.models import TaskInput, ToolCall
from agentgs.runtime.service import TaskService

async def run_example():
    settings = RuntimeSettings(max_concurrent_tasks=4, max_concurrent_tool_calls=2)
    async with TaskService(Path.cwd(), settings) as runtime:
        request = TaskInput(
            parallel_tools=True,
            calls=[
                ToolCall(id="mean", tool="RemoteSensingMean", arguments={"x": [1, 3]}),
                ToolCall(id="temperature", tool="RemoteSensingKelvinToCelsius", arguments={"kelvin": 273.15}),
            ],
        )
        record = await runtime.submit(request)
        result = await runtime.wait(record.spec.id)
        events = await runtime.events(record.spec.id)
        return result, events
```

`parallel_tools=true` 允许显式调用中的独立步骤并发；`ToolCall.depends_on` 用调用 ID 声明依赖，依赖失败时不执行下游。循环依赖和未知 ID 在执行前拒绝。现有问答工具链仍按已规划的数据依赖顺序执行，模型返回参数的原生 Function Calling 协议保持不变。每个调用独立登记产物，并发批次逐个保存已完成结果，中断后保留部分结果与调用记录。

`llm.toml` 的 provider 可配置 `max_concurrent_requests`、`requests_per_minute` 和 `tokens_per_minute`；每次 HTTP 重试也受约束，退避期间不占请求额度。TPM 使用保守的输入估计和最大输出预算预留，有用量回执时按实际 `total_tokens` 结算，没有回执时保留预留值。

显式工具调用无需模型密钥：

```bash
tools/envs/reproduction/bin/python -m agentgs --help
tools/envs/reproduction/bin/python -m agentgs run --tool DeepEarthQueryUSGSCatalog --arguments '{"year":2011,"operation":"largest"}'
tools/envs/reproduction/bin/python -m agentgs run --tool GeoBigDataExecuteGeoSPARQL --arguments '{"source_id":591}'
tools/envs/reproduction/bin/python -m agentgs ui serve
```

浏览器访问 `http://127.0.0.1:8000`。任务、事件、失败记录与产物位于 `outputs/`。每次工具调用有独立产物目录，后续调用可读取同一任务的已产生文件。ExecutePython 是显式内部沙箱操作，未纳入 101 项领域工具或模型候选列表。原生 Windows 科学运行尚未验收，使用 WSL。

## 验收与评测

```bash
tools/envs/reproduction/bin/python -m pytest -q
.venv/bin/ruff check src scripts
.venv/bin/ruff format --check src scripts
```

历史基线采集、工具覆盖和 worker 辅助验收脚本已移除；既有对照记录与测试 fixture 保留。生产代码不依赖旧源码目录。

已有预测可以离线评分；原 FAS、复合指标及历史兼容层分别报告：

评测以 `Conversation` 为对话记录，执行信息通过 `execution_events()` 从标准 `tool_calls` / `tool_call_id` 消息关系解析。每次 benchmark 尝试保存 `conversation.json` 和 `messages.jsonl`，保留原始参数、运行时绑定参数、reasoning 和模型请求事件。离线预测可提供 `messages` 列表或完整 `conversation` 对象；仅有旧 `trace` 的预测需要先取得原始消息。

当前执行评分协议为 `geotooleval-conversation-v2`：只统计当前用户轮次的领域调用，包含失败重试，排除规划调用和对照分支；并行回复按调用声明顺序评估。参数按实际执行输入评分，模型用量优先从对话中的请求记录推导；请求未完成、失败或重试用量不完整时，成本分和综合分保持未知。答案 FAS 规则保留。新协议实验使用新目录，旧实验可通过保存的消息离线重评分。

```bash
tools/envs/reproduction/bin/python -m agentgs evaluate score predictions.jsonl --allow-partial --output outputs/evaluations/scored.json
```

小规模真实模型实验使用同一 benchmark 入口，结果保留每次失败，不按分数选择重跑：

```bash
tools/envs/reproduction/bin/python -m agentgs evaluate benchmark --ids 61,86 --concurrency 2 --timeout 240 --output outputs/evaluations/small-run
```

603 题实验入口为 `agentgs evaluate benchmark`；省略 `--ids` 时评测全部题目。续跑检查源码、数据和配置身份；不同版本使用新目录。标准答案仅用于 evaluator，不进入 Agent 请求。

实际模型验收使用 gpt-5.6-terra：Q61 FAS=1，Q86 FAS=0.75（218.015 的最终取舍与冻结答案不同）；Geo 参数修复后的 Q106/Q111 均 FAS=1。失败尝试和超时报告同时保留于 `outputs/verification/`，这些小样本不证明全部实验性能复现。

## 前端与安装包

```bash
tools/envs/reproduction/bin/python scripts/build/build_api_types.py
tools/envs/reproduction/bin/python scripts/build/build_frontend.py
uv build
```

产物为 `dist/agentgs-0.2.0-py3-none-any.whl` 与 `dist/agentgs-0.2.0.tar.gz`。wheel 包含工具目录、声明、GeoQuestions 选择清单与前端静态资源；sdist 包含环境约束、准备脚本、算法契约、测试及非敏感配置。原始数据、密钥、工具环境及 node_modules 不打包。

## 服务器运行603题

本地已准备可迁移数据包 `dist/agentgs-data-final.tar` 与逐文件校验清单 `dist/agentgs-data-final.manifest.json`。它只包含实际runtime/evaluator所需的冻结数据、模型、KG及最新格式版本，没有密钥、旧源码和本地抽样夹具。将源码sdist、该数据包及本机单独配置的API信息传到服务器：

```bash
tar -xzf agentgs-0.2.0.tar.gz
cd agentgs-0.2.0
tar -xf ../agentgs-data-final.tar
uv sync --locked --extra visualization
.venv/bin/python scripts/tools/install_reproduction.py --python 3.12
cp .env.example .env
# 在服务器填写 .env 的 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL
tools/envs/reproduction/bin/python -m agentgs config validate
tools/envs/reproduction/bin/python -m agentgs evaluate benchmark --output outputs/evaluations/server-603
```

最后一条是真实603题模型实验，按用户决定在服务器启动。既有结果默认续跑；`--retry-failed`显式保留重试attempt，数据/代码/模型配置改变时使用新输出目录。

## 抽样数据流审计

审计报告为 `outputs/verification/leakage-audit.json`。10题请求未发现标准答案、标准链或评分反馈注入；22个重建用户提示全部匹配原记录SHA256。重建消息保存在 `outputs/verification/audited-prompts/`，原始HTTP请求正文未留存，审计不新增网络调用。

这组抽样属于工程回归。GeoQuestions Q106经授权读取官方物化答案和参考查询，不能用作独立未知答案/RDF几何推理成绩；海洋Q347读取10行benchmark派生缓存，来源是原SOCAT数据而非私有标签，但未在运行时重算完整原网格。Q61/Q106此前用于调试，格式文件也包含上游事后诊断修订，10题满分不代表严格留出集泛化准确率。

通用科学资源解析已拒绝访问评测标签/历史评分、回答格式文件及Geo回放数据目录；专用Geo快照工具保留已授权访问，真实自然灾害表保持可读。实际样题未发现访问这些私有文件，此边界仍不是完整OS沙箱。

GEM 工作簿已完整转为 `data/gem.jsonl.gz` 并由 Git 保留；`prepare_gem.py` 展开到 `data/gem.jsonl`，能源准备直接读取 JSONL。新工作簿可用 `.venv/bin/python scripts/prepare/prepare_gem.py --workbook data/raw/gem/文件名.xlsx` 导入。固定查询名单、区域和地理参考值位于 `data/geoquestions/profile.json` 与 `data/reference/`，由 Git 管理。
