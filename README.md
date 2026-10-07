# AgentGS

基于类型契约的地学 agent 与 task runtime。工作仓库位于 Ubuntu-24.04 的 `/home/gaosong/Code/Project/AgentGS`。

## 当前实现范围

- `contracts`：任务、工具、产物、结果与事件模型。
- `client`：provider/model 分离的 OpenAI 兼容异步调用，能力/effort校验、连接复用、有限重试、未知usage保留为null。首版使用非流式chat completions。
- `skills`：独立描述、prompt与准备/执行接口。内置地震目录与地理计算两项能力。
- `agent`：可替换检索接口、关键词Skill选择、模型工具调用循环与显式工具计划。
- `runtime`：独立任务worker、并发队列、取消/超时、RSS监测、SQLite事件、产物校验和、bwrap与显式PRoot兼容回退。
- `evaluation`：同一TaskService运行JSONL用例，gold不进入运行请求；首版提供工具链与答案精确匹配，尚未替代旧FAS评测器。
- `cli`：Typer/Rich子命令；`visualization`：FastAPI、SSE与React任务控制台。

本版迁入旧110工具中的8项：6个Geoscience工具和2个USGS工具；新增ExecutePython。其中USGS Query为部分迁移（旧聚合接口尚未补齐）。其余102项及其派生数据/索引仍见 `docs/migration_inventory.json` 的pending状态。未导入整个旧runtime或vendor，也未声称全量科学工具/旧分数已复现。

## 安装与离线执行

```bash
cd /home/gaosong/Code/Project/AgentGS
uv sync --locked --extra visualization
uv run --locked agentgs --help
uv run --locked agentgs run --tool GeoHaversineDistance   --arguments '{"lat1":0,"lon1":0,"lat2":0,"lon2":1}'
uv run --locked agentgs run --tool DeepEarthQueryUSGSCatalog   --arguments '{"start_year":2010,"end_year":2020,"min_magnitude":8,"limit":10}'
```

工具参数已经采用类型schema，旧工具的字符串输入不能原样套用。USGS读取已下载原始 `data/usgs/usgs_2010_2020_m7.csv`。结果、事件和产物存储于outputs；工具结果带来源类型，目录查询不冒充实时数据。

## 模型配置

复制 `.env.example` 为 `.env` 并填写 `LLM_BASE_URL`、`LLM_API_KEY` 和 `LLM_MODEL`。`llm.toml` 只引用 `${VAR}`，不存秘密。配置按 `.env` 后进程环境覆盖加载，不修改进程全局变量。

```bash
uv run --locked agentgs config validate
uv run --locked agentgs run '查询2010到2020年震级至少8的地震数量'
```

按实际模型修改context/output预算、supported_efforts及tool_calling能力。支持`--model`选择配置模型、`--effort`选择声明的effort。provider可设置`output_token_parameter="max_completion_tokens"`，默认max_tokens。当前配置语法是本项目方案，不声称复刻未明确的grok项目。没有配置模型时显式工具任务仍然可运行。

## 外部工具与代码沙箱

```bash
.venv/bin/python scripts/tools/install_node.py
.venv/bin/python scripts/tools/install_bwrap.py
.venv/bin/python scripts/tools/install_proot.py  # 可选
uv run --locked agentgs tools doctor
uv run --locked agentgs run --tool ExecutePython --arguments '{"code":"print(1 + 1)"}'
```

安装结果放tools，manifest记录固定版本、平台与hash。已提供Ubuntu24.04 x64 bwrap/PRoot及Linux/Windows x64/arm64 Node清单；其它发行版/架构不冒充已验证支持。无sudo或系统包修改。

代码进程只挂载必要运行环境和task代码目录，环境变量使用白名单；bwrap路径隔离网络，启动前功能探测。PRoot是弱兼容环境，只有runtime.toml明确allow_proot=true才允许回退，结果标明network_isolated=false。缺失后端不会直接在宿主执行代码。

RSS上限由进程树watchdog实施，采样可能存在瞬时超限；当前没有cgroup强制限制或Windows原生AppContainer。Windows主路径通过WSL，已提供无shell拼接的CLI桥接：

```powershell
agentgs wsl -- run --tool GeoMagnitudeToEnergy --arguments '{"magnitude":7}'
```

原生Windows科学工具与沙箱尚未端到端验收。并发/资源参数见runtime.toml。初版每task独立进程，未实现池化、跨runtime全局配额、崩溃自动续跑。

## 数据处理、索引与展示

```bash
.venv/bin/python scripts/prepare/prepare_usgs.py
.venv/bin/python scripts/build/build_tool_index.py
.venv/bin/python scripts/build/build_skill_index.py
.venv/bin/python scripts/build/build_api_types.py
.venv/bin/python scripts/build/build_frontend.py
uv run --locked --extra visualization agentgs ui serve
```

浏览器访问 `http://127.0.0.1:8000`。前端类型由Pydantic序列化schema生成；显示真实任务状态、执行事件、结果与产物，无演示统计。JSONL评测可运行：

```bash
.venv/bin/python scripts/evaluation/run_suite.py --cases docs/examples/offline_suite.jsonl
```

下载层只下载原始字节；prepare写独立处理产物。现有download脚本与此前已下载数据保留；其余领域prepare/build随工具迁移补齐。

## 验证与目录约定

```bash
uv run --locked --extra visualization pytest -q
uv run --locked ruff check src scripts/tools scripts/build scripts/prepare scripts/evaluation
uv build
```

所有自动化测试在src/tests按模块划分；scripts通过人工CLI/实际输出验证，不另写镜像测试。前端执行TypeScript检查和Vite构建。wheel包含prompt和已构建静态资源，排除node_modules与测试目录；构建前运行frontend脚本。

模型工具循环通过mock HTTP验证，没有执行付费模型实验。旧项目只读保留，全部工作在新仓库；代码迁移计划见docs/refactor_plan.zh-CN.md。
