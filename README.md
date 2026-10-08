# AgentGS

标准 Python 地学 Agent 项目。当前范围为 **101 项 GeoToolEval 工具、24 个 Skill、31 个 Capability**，默认执行原版的分阶段流程。核心源码以原主项目冻结版本为基线；生产包独立于旧源码目录。

## 当前重构状态

101 项工具均有实际处理函数，已在科学复现环境逐项执行成功。100 项科学工具使用同一输入与旧源码对照一致；GeoQuestions 按用户授权保留 40 条物化结果的 source_id / 查询结构匹配协议，另有 125 项独立回放对照。已迁入能源、海洋共享解析器、空间/栅格科学操作及相应数据定位。

默认 Agent 已连接 Skill 检索、KG 检索、规划、桥接、grounding、参数绑定、原子反馈、重规划、成功前缀复用、最终模型回答和非修改性验证。原 function-calling 原型仅在显式选择 `algorithm=prototype` 时使用。24 项 Skill 是动态能力声明，不代表 24 条固定工具链。

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

原始数据保留在 `data/`，不包含在 wheel 中。当前工作区已准备数据；新工作区可从原项目导入冻结输入、模型和 KG，按 SHA256 校验，不导入旧运行代码：

```bash
tools/envs/reproduction/bin/python scripts/prepare/import_reference_data.py --source /mnt/d/gaosong/Desktop/GeoToolAgent
tools/envs/reproduction/bin/python scripts/prepare/prepare_usgs.py
tools/envs/reproduction/bin/python scripts/prepare/prepare_geoquestions.py
tools/envs/reproduction/bin/python scripts/prepare/prepare_geotooleval.py --source /mnt/d/gaosong/Desktop/GeoToolAgent --contracts-source '/mnt/d/gaosong/Desktop/大创项目（新)/AgentGS_主项目_20261006'
```

GeoToolEval 原始工作簿和自然灾害表由 `scripts/download/download_geotooleval.py --source-dir <原数据目录>` 导入，清单校验原字节。`import_reference_data.py` 保留原 derived 表、完整 GEM/MRDS 输入与模型缓存；尚未把所有历史 QA/derived 生成脚本迁为重建入口，当前复现依赖冻结输入导入。

## 最新数据与算法版本

最新公开回答合同为 v1.2，含 2026-10-07 的 Q106/Q111 修订，SHA256 为 `a9cdf8fef4f82021cd6b5f5afa55ef9a262ea7ca1def3e2dcaeae0c0bdb60c25`。文件来源与选择记录在 `data/geotooleval/version_manifest.json`。保留发布的 603 题 v1.1 扩列工作簿，其前九列与原题表完全相同；当前运行提示使用更新的 v1.2 公开合同。旧文件及之前的验证结果均保留。

GeoQuestions 只按公开 numeric source_id 或提交的 SPARQL 结构定位物化结果，符号 topic 不充当 ID。查询匹配保留原权重、阈值和歧义规则，未匹配返回失败。结果标记 `derived_lookup`、`answer_lookup=true`，不声称执行了 RDF 端点。

`agent.toml` 固定算法开关与预算；`llm.toml` 管理 provider、model 和角色；`.env` 保存本机连接信息并被 Git 忽略。模型模式需要 `LLM_BASE_URL`（含 `/v1`）、`LLM_API_KEY`、`LLM_MODEL`。用量缺失、失败或中断时保留 unknown，成本和综合分不可据此补零。

## 启动与示例

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
tools/envs/reproduction/bin/python scripts/build/audit_algorithm_alignment.py --output outputs/alignment/current.json
tools/envs/reproduction/bin/python scripts/evaluation/verify_tool_coverage.py --output outputs/verification/tools-new
tools/envs/reproduction/bin/python scripts/evaluation/validate_workers.py --output outputs/verification/workers-new
tools/envs/reproduction/bin/python scripts/evaluation/replay_reference_scores.py
```

工具覆盖验证使用 reference / synthetic 参数，属于工具验收，不是模型规划实验。旧科学源码对照可加 `--source /mnt/d/gaosong/Desktop/GeoToolAgent`；仅验收脚本读旧源码，生产代码没有该依赖。`audit_algorithm_alignment.py --require-runtime-parity` 在完整旧 harness 回放验收前返回非零。

已有预测可以离线评分；原 FAS、复合指标及历史兼容层分别报告：

```bash
tools/envs/reproduction/bin/python -m agentgs evaluate score predictions.jsonl --allow-partial --output outputs/evaluations/scored.json
```

有界真实模型验收最多 5 题、每题最多 12 个模型阶段，结果保留每次失败，不按分数选择重跑：

```bash
tools/envs/reproduction/bin/python scripts/evaluation/validate_live.py --ids 61,86 --max-model-calls 8 --timeout 240 --output outputs/verification/live-new.json
```

603 题实验入口为 `agentgs evaluate benchmark`，可先用 `--ids 61,86 --output outputs/evaluations/small-run`。完整实验没有在本次验收中启动。续跑检查源码、数据和配置身份；不同版本使用新目录。标准答案仅用于 evaluator，不进入 Agent 请求。

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
