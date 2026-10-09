# AgentGS

Standard Python geoscience agent. The current scope is **101 GeoToolEval tools, 27 Skills, and 34 Capabilities**, with explicit task-requirement selection and staged execution. The production package is self-contained.

## Refactor status

Every one of the 101 tools has a real handler and has run successfully in the scientific reproduction environment. One hundred scientific tools match the old source on the same inputs. GeoQuestions keeps the authorized protocol that matches 40 materialized rows by `source_id` or query structure, plus 125 independent replay checks. Energy and ocean share parsers, and spatial and raster operations use the current data layout.

The default agent connects skill retrieval, task-requirement selection, KG retrieval, planning, bridging, grounding, argument binding, atomic feedback, replanning, reuse of successful instances, the final model answer, and non-mutating validation. Native `select_capabilities` calls adopt relevant operations and approve proposed support; `submit_tool_plan` declares execution instances. Each scientific tool has its own JSON argument schema. Skills describe operation scopes and dependencies, including a typed hazard load-filter-count bridge composition.

Checked independent comparisons include 341 deterministic component decisions, 16 original KG/SDK retrievals, 298 argument bindings, 65 groups of scientific results and artifacts, and 2,412 original FAS scores. Twenty-eight saved experiment answers rescore the same under original and compatibility scoring. A real TaskService/worker has verified failure recovery, a six-step spatial chain, and cross-domain artifact passing.

**Replay of every old harness entry, mode, and bridge combination is not finished, so `runtime_equivalent=false` stays.** That means full-chain equivalence evidence is still incomplete. The migration inventory no longer marks implemented modules as placeholders. Evidence and the unfinished scope are in [algorithm_alignment.json](docs/algorithm_alignment.json) and [migration_inventory.json](docs/migration_inventory.json).

## Ten-domain sample delivery

A local sample of one question from each of the 10 domains completed successfully with frozen FAS=1 on `gpt-5.6-terra`, 22 model stages in total. The questions were Q61, Q106, Q141, Q199, Q287, Q347, Q447, Q471, Q496, and Q544. Remote sensing and natural hazards use multi-step chains. The scientific reproduction environment passed 714 tests with no skips. The summary is `outputs/verification/delivery.json`. Per-question requests, traces, usage, and scores are in `domains-1-5.json` and `domains-6-10.json`.

The 603-question agent experiment runs on the server, as decided. The local 2,412-score oracle checks scoring offline and does not run the 603-question model experiment. `runtime_equivalent=false` remains the boundary for broader formal acceptance and does not block this agreed ten-domain sample.

## Setup

The runtime path is Linux or WSL. The scientific reproduction environment uses Python 3.12 and the original pins, including NumPy 1.26.4 and SciPy 1.15.3. The default development environment and the reproduction environment are managed separately.

```bash
cd /home/gaosong/Code/Project/AgentGS
uv sync --locked --extra visualization
.venv/bin/python scripts/tools/install_reproduction.py --python 3.12
.venv/bin/python scripts/tools/install_reproduction.py --verify
```

Prepared results live in `data/`. A single-file dataset is one file. A multi-file dataset is a directory. A multi-file dataset larger than 64 MiB is also stored as a ZIP. The matching prepare script checks and unpacks it. Unpacked results stay in `data/`. Raw downloads and temporary files stay in `data/raw/`. GeoToolEval is the Git-tracked `data/geotooleval.jsonl`, with the workbook question, answer, resource, tool chain, and format columns stored under English names. Other generated data is built by `scripts/prepare/` from the available raw sources. Preparation does not fall back to a Git-managed result snapshot.

```bash
tools/envs/reproduction/bin/python scripts/prepare/prepare_all.py
# Use downloaded raw inputs and retained local attachments only.
tools/envs/reproduction/bin/python scripts/prepare/prepare_all.py --offline
# Point at retained attachments on first use.
tools/envs/reproduction/bin/python scripts/prepare/prepare_all.py --source-root data
```

Each domain entry is `prepare_<domain>.py`. The aggregate entry is `prepare_all.py`. The fixed 40 GeoQuestions rows are produced by their own prepare script and are Git-ignored. Answer formats are stored on each `data/geotooleval.jsonl` record, so they need no extra preparation. Minerals are rebuilt from MRDS for the original 8,000 record IDs. Ocean and energy are computed from raw files. Remote sensing keeps the original pixels in a fixed set of 49 directories. The GISTEMP annual series is built from the NASA monthly table. If the old World Bank file is missing, the current API ZIP is used. A newer source may change scientific values. Every run records the source-file hashes actually used. BioDeepTime observations and hand-drawn zones, CCS spatial assets, and the original hazard table still require their retained inputs. Paths, sources, and the optional explicit snapshot restore mode are in [prepare/README.md](scripts/prepare/README.md).

Git keeps GeoToolEval, the compressed GEM JSONL, fixed reference data, a few raw assets that cannot yet be rebuilt completely, the deep-earth inputs, and the frozen KG JSON supplied for the project. Generated results, public downloads, large models, and caches are ignored. Source hashes and selection parameters live in the prepare `manifests/`. A successful prepare run is not a successful end-to-end model experiment.

The tool graph lives in `data/geotoolkg/`; `config.toml` locates the local ONNX model. Run `uv run scripts/prepare/prepare_embedding.py` on a CPU worker to prepare `embedding/tools.npz`, `questions.npz`, and `contexts.npz`. Queries use the same public question/resource builder as execution. Context vectors cover the graph expansion strings. The preparation receipt records counts, model files, graph and vector-file hashes. Benchmark resume identities include the vector files.

Each `TaskService` owns one loopback embedding service, initialized and warmed once on its first retrieved-mode scientific task. Isolated task processes reuse that encoder and its prepared vectors. Previously unseen texts are encoded on CPU and retained in a bounded in-memory cache. Closing the runtime closes the service; `data_only`, explicit tool calls and `all_registered` planning do not require this encoder. Independent runtime launches each own their encoder. This keeps question cancellation and process isolation while avoiding per-question ONNX startup.

The local `.env` holds the supplied model connection and selects `gpt-5.6-terra`. Secrets are Git-ignored. On 2026-10-08 the server connection check could list models. The live agent for Q287 planned, called `EnergyCapacityValue`, and answered `620.399 MW` in 2 model stages, with FAS=1 and PAS=0.8. The strict argument-success gate did not pass. The report is `outputs/verification/live-server-20261008-q287.json`. That was a one-question connection and execution check, not the full 603-question server experiment. HTTPX SOCKS support is included for the local proxy.

The server then ran one question from each domain in parallel on `gpt-5.6-terra`: Q61, Q106, Q141, Q199, Q287, Q347, Q447, Q471, Q496, and Q544. On 2026-10-08 all ten finished. Eight had FAS=1, mean FAS was 0.855, and there were 22 model stages in about 52 seconds. Worker events showed a peak of 10 processes at once. Q106 omitted five municipality records and scored FAS=0.216667. Q199 returned the correct temperature but omitted the age and structure fields and scored FAS=0.333333. The strict argument-success gate passed 2/10. Execution success is not treated as success on every score. Full answers, tool artifacts, events, model usage, and scores are in `outputs/verification/parallel-domains-20261008.json`. Questions were not rerun or replaced based on the score.

The separate ten-domain helper script has been removed. Historical reports remain. New samples use `agentgs evaluate benchmark --ids` and a new output directory.

The full 603-question live experiment was then authorized. Its output directory is `outputs/evaluations/server-603-max128-20261008T145211Z/`, model `gpt-5.6-terra`. The runner cap was 128 concurrent tasks, 600 seconds and 2,048 MB RSS per question, and 256 GiB total RSS. The benchmark entry uses bounded parallel scheduling. Runtime resources are CLI arguments. `experiment.log` and `experiment.events.jsonl` record progress. `progress.json` stores the current counts. Each `attempt_*` stores the public request, task ID, full result, events, stderr, and scores. Artifacts are under `outputs/tasks/<task_id>/`. `launch.json` records the supervisor and the execution process. `exit.json` records the exit status when the execution process ends.

That run finished on 2026-10-08. The supervisor took 653.176 seconds and the runner exited 0. All 603 questions kept their first attempt: 512 succeeded and 91 failed (85 algorithm incomplete, 5 final-answer TypeError, 1 worker-event JSONDecodeError). There were no timeouts and no RSS limit hits. Mean FAS was 0.698711. The ten-domain macro mean FAS was 0.621294. 333/603 questions had FAS=1. The strict argument-success gate passed 48/603. Execution success, answer score, and argument score are reported separately. 1,386 model stages reported 4,123,350 tokens. One question had incomplete usage, and missing usage was not filled with zero. `run_audit.json` checked all 603 IDs, six record types per question, 603 predictions, source and data identity before and after the run, and a worker-timeline peak of 128. Logs and reports do not contain the model key. Full scores are in `summary.json`. Failed questions were not rerun automatically.

```bash
# New experiment. The supervisor stores the full console log and the exit status.
tools/envs/reproduction/bin/python scripts/evaluation/run_logged_benchmark.py --output outputs/evaluations/server-603-new --concurrency 128 --timeout 600 --rss-limit-mb 2048
# After the run ends or is interrupted, resume unfinished questions with the same source, data, and config:
tools/envs/reproduction/bin/python -m agentgs evaluate benchmark --output outputs/evaluations/server-603-max128-20261008T145211Z --concurrency 128 --timeout 600 --rss-limit-mb 2048
```

A default resume keeps failed results. `--retry-failed` retries failed questions and keeps the original attempt. If an interrupt leaves `.run.lock`, confirm that the process in `launch.json` has exited before removing the lock. A directory that is still running must not be resumed concurrently. The ten-domain sample and this full run are kept separately. Neither is evidence that every old harness mode is equivalent.

The confirmed `GeoToolEval_fixed_answer_format.xlsx` was converted once to `data/geotooleval.jsonl`, 603 questions. The 11 workbook fields use English names: `question_id`, `category_id`, `question`, `expected_answer`, `gold_chain_text`, `chain_mode`, `answer_type`, `input_resource`, `gold_steps`, `answer_template`, and `format_notes`. Controlled values, format notes, and resource labels are English. Each record also has `question_sha256` and `answer_format` so the runtime can read the public answer requirement directly. There is no separate answer-format file. The shared English system prompt is in `src/agentgs/datasets/geotooleval.py`. The source SHA256, domain counts, output hashes, and the other two worksheets are in `data/metadata/geotooleval.json`. Formula text and cached values are kept, and formulas are not executed. The main data and the source note are Git-tracked. The converter and the workbook copy are not kept.

## Current data and algorithm version

The embedded data follows the last uploaded `GeoToolEval_fixed_answer_format.xlsx`. Compared with the previous upload, two answer-format columns were added and the standard tool-call detail changed for 75 questions. Question text, answers, and the other original fields were unchanged. The source and the change record are in `data/metadata/geotooleval.json`. Format templates and notes are written in English. Runtime and evaluation read the same JSONL. Loading checks that the answer-format settings match the public columns and the question hash.

GeoQuestions locates a materialized result only by the public numeric `source_id` or by the submitted SPARQL structure. A symbolic topic is not an ID. Query matching keeps the original weights, thresholds, and ambiguity rules. A miss returns failure. Results are marked `derived_lookup` and `answer_lookup=true`. They do not claim that an RDF endpoint was executed.

`[agent]` in `config.toml` holds algorithm switches, budgets, and retrieval paths. `llm.toml` stores the provider and the single model used for planning, argument filling, and the final answer. Runtime resources and the sandbox are CLI arguments. `.env` holds the local connection and is Git-ignored. Model mode needs `LLM_BASE_URL` (including `/v1`), `LLM_API_KEY`, and `LLM_MODEL`. Missing, failed, or interrupted usage stays unknown. Cost and the combined score are not filled with zero.

## Tool-planning ablation

For composable A1–A6 module interventions inside normal `skill_guided` mode,
including dynamic planning instructions and the equal-information bipartite
control, see [Module ablations](docs/module-ablations.md). These interventions
are separate from the `tool_only` and `data_only` baselines below.

`planning_mode="tool_only"` uses the same AgentEngine and the SciToolAgent stage order: tool KG retrieval, the model submits a tool chain, arguments are generated and repaired step by step, then the model answers. This mode does not load Skill definitions, Capability expansion, or the planner. It does not use capability completion, the shared bridge, Skill grounding, or those prompts. The planning stage is recorded as `tool_planning`. The tool scope is declared separately in `contracts/tool_scope.json`. A task-level mode override is resolved before the worker starts.

Both modes share the 101 tools, standard function calls, KG retrieval, tool-level domain seeds and prompts, public resource binding, artifact tracking, single-tool repair, replanning, answer formats, and scoring. Tool-level rules do not read Skill or Capability declarations. The ablation measures the Skill/Capability planning layer, including candidate expansion, completion, and bridging. It is a tool-layer baseline aligned with SciToolAgent inside AgentGS, not a strict replay of the old code.

The old `require_llm` parameter is gone. Config and task JSON use the fields below. Question answering always needs the model. Arguments and the final answer are always produced by the model.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `allow_rule_planning` | `false` | Allow rules to route directly, fall back after a planning failure, or revise the tool chain. This does not control whether the model is called. |
| `plan_attempts` | `3` | Total full-chain planning attempts, including replanning after execution failure. Independent of the rule switch. |
| `tool_attempts` | `3` | Argument and execution repair attempts for each tool instance. |
| `argument_attempts` | `3` | Shared protocol-repair budget within a planning stage; independent of HTTP retries. |
| `model_attempts` | `4` | Transport-retry budget for model requests. |
| `command_attempts` | `20` | Bash calls allowed in `data_only`. Ignored by other modes. |

The old `require_llm=true` planning policy maps to `allow_rule_planning=false`. The old `require_llm=false` maps to `allow_rule_planning=true`. The current default turns rule planning off. Turn it on explicitly to keep rule assistance from an old config. Planning attempts are set on their own and are not reduced to one when rule planning is off. `shared_bridge` is fixed to `false` for `tool_only` and `data_only`. `data_only` also fixes `allow_rule_planning` to `false`.

`planning_mode="data_only"` does not load Skills, the tool KG, or the scientific tool registry. The model has only `bash`. On Linux, bwrap bind-mounts the allowed directories and files read-only at their original relative paths under `/data`. It does not copy them, change their format, or pack them per question. Symlinks at the top of the data tree are rejected. `geotooleval.jsonl`, `metadata/`, `geoquestions/`, `raw/`, and `geotoolkg/` are not mounted. `/work` for that question is writable and kept across calls. `/tmp` and `/dev/shm` are temporary. The network is blocked by a separate network namespace. Host environment variables are not passed in. The interpreter and virtual environment are available read-only. If the container cannot mount a private `/proc`, an empty `/proc` is used. PID and network namespaces are still kept. The host `/proc` is not bind-mounted, and execution does not fall back to PRoot or host Bash.

Bash and scientific tools share the runtime tool quota, queue deadline, cancellation, and task RSS supervision. They record a standard `ToolResult`, events, and JSON artifacts. A non-zero exit is a failure. stdout and stderr are kept, at most 16 KiB per stream. Extra output is still consumed so the command can finish. Exploration count is limited by `command_attempts` and ignored by other modes. The final answer uses the shared format check. The comparable metric for this experiment is the answer score FAS. Tool-chain match is summarized as `not_applicable`.

The desktop demo uses `--sandbox auto`. On macOS, `sandbox-exec` uses the system policy to limit file access and block the network when the capability probe succeeds. If it is missing or the probe fails, the demo uses local Bash. Windows uses Git Bash, and the existing WSL entry can provide Linux isolation. Desktop local paths are not remapped to `/data`. The model receives the real data and work directories. A local fallback records `isolation=native_demo`, `data_mount=native_unprotected`, and `network_isolated=false`. It does not guarantee a read-only mount, hidden host files, or a blocked network. It is for the demo and is not the Linux isolation experiment.

A formal comparison changes only the mode, uses the same questions and model, and writes each run to a new directory. `data_only` can use a larger per-question timeout:

```bash
uv run agentgs evaluate benchmark --benchmark-scope common_data --planning-mode skill_guided --no-rule-planning --plan-attempts 3 --tool-attempts 3 --output outputs/evaluations/full-new
uv run agentgs evaluate benchmark --benchmark-scope common_data --planning-mode tool_only --no-rule-planning --plan-attempts 3 --tool-attempts 3 --output outputs/evaluations/tool-only-new
uv run agentgs evaluate benchmark --benchmark-scope common_data --planning-mode data_only --no-rule-planning --command-attempts 20 --timeout 900 --output outputs/evaluations/data-only-new
```

The logged supervisor accepts the same mode and budget arguments. The experiment protocol records the task overrides that actually applied, the resume budget, and the hashes of the tool and planning declarations. Each AgentResult stores the effective config in `provenance.effective_algorithm`. An experiment with old parameters, source, or protocol cannot resume into a new directory.

Runtime storage converts historical tasks on read. It keeps the old rule policy and the original planning budget so old results can still be viewed. Conversion does not overwrite the stored record and does not accept old parameters as a new task input.

## Run and examples

`run`, `evaluate run`, `evaluate benchmark`, and `ui serve` take the runtime arguments below. They do not read a runtime config file. Omitted arguments use the code defaults:

| CLI argument | Default | Purpose |
| --- | --- | --- |
| `--concurrency` | 2 | Maximum concurrent tasks (1–128) |
| `--model-concurrency` | 16 | Concurrent model HTTP requests in one project |
| `--tool-concurrency` | 64 | Concurrent tool executions in one project |
| `--sandbox-concurrency` | 16 | Concurrent sandbox calls, also bounded by the shared tool quota |
| `--sandbox-disk-limit-mb` | 2048 | Persistent work plus per-call scratch budget (MiB) |
| `--queue-capacity` | 1024 | Accepted tasks that have not finished. The API returns 429 when full. |
| `--model-timeout` | 180 | Deadline for one model call, including queue, retries, and backoff (seconds) |
| `--tool-timeout` | 120 | Deadline for one tool call, including queue and compute (seconds) |
| `--timeout` | 300 | Deadline for one question, including queue, setup, and execution (seconds) |
| `--rss-limit-mb` | 1024 | RSS limit per question (MB) |
| `--max-total-rss-mb` | 4096 | Total RSS budget for workers in the same environment (MB) |
| `--code-timeout` | 20 | Timeout for a code step (seconds) |
| `--sandbox` | `auto` | Linux `auto`/`bwrap` requires bwrap. Desktop `auto` uses the system sandbox or the demo fallback. `disabled` allows local execution on the desktop only. |
| `--allow-proot` / `--no-allow-proot` | disabled | Kept for the flag. Model code and Bash do not use PRoot. |
| `--worker-python` | automatic | Absolute or project-relative path to the worker Python |

If concurrency or the per-question RSS limit is set and the total RSS budget is not, the total budget is raised to at least concurrency times the per-question RSS. An explicit `--max-total-rss-mb` is used as given. The worker prefers the project's scientific reproduction environment and otherwise uses the current Python.

Task, model, and tool quotas are shared through `outputs/runtime/coordination/<worker-identity>/coordination.sqlite3`. Runtime instances in the same project and worker environment share one quota, and live instances must use the same concurrency settings. Different machines or PID namespaces use their own quotas so a lease from another pod is not treated as a dead process. Instances can query, wait for, and cancel tasks across the shared store. Built-in scientific tools call their Python handlers directly and do not enter the sandbox. A synchronous handler runs in a thread. An asynchronous handler is awaited directly. After cancellation, a synchronous function is allowed to exit. For a hard stop at tool level, set `RuntimeSettings(tool_backend="process")` in the Python API. Task workers and Bash child processes remain under runtime supervision. Cancelling Bash reaps the process tree. Nested sandbox execution reuses the tool quota it already holds.

Sandbox calls acquire the tool and sandbox quotas atomically. Calls sharing a persistent work directory wait for that directory before acquiring a compute quota. Compute quotas are released before result serialization and artifact publication. Quota release wakes waiters in the same process; other processes retry with a bounded delay. Events distinguish `tool.queued`, `tool.started`, and `tool.completed`, with wait and execution durations. Task recovery, queue capacity, and RSS totals use the same worker identity; tasks from another worker or with unknown ownership are not terminated using local PID information. Finished dead workers no longer reserve RSS.

Linux `/tmp` and `/dev/shm` are backed by per-call scratch directories on the work filesystem and removed after child processes stop. They do not consume unaccounted tmpfs memory. A watchdog checks process-tree RSS and the combined work/scratch size before, during, and after execution. Disk accounting defaults to 2048 MiB, 10000 directory entries (`RuntimeSettings.sandbox_max_files`), and 64 directory levels. It does not follow symlinks or open special files. Exceeding a limit produces `resource_limit`; persistent `/work` files remain available for inspection. These are monitored limits rather than filesystem quotas. Desktop demo paths retain their declared isolation limits and have not been validated on native macOS/Windows.

Submit, query, and event reads on the Python runtime API all need `await`. An async context manager closes the service:

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

`parallel_tools=true` lets independent steps in an explicit call run concurrently. `ToolCall.depends_on` names call IDs. A failed dependency skips the downstream call. Cycles and unknown IDs are rejected before execution. Existing question-answering chains still run in the planned data-dependency order. The native function-calling protocol for model-returned arguments is unchanged. Each call records its own artifacts. A concurrent batch saves finished results one by one, and an interrupt keeps the partial results and the call record.

A provider in `llm.toml` can set `max_concurrent_requests`, `requests_per_minute`, and `tokens_per_minute`. Each HTTP retry is constrained, and backoff does not consume request quota. TPM reserves a conservative input estimate and the maximum output budget. When a usage receipt arrives, the reservation is settled with the actual `total_tokens`. Without a receipt, the reservation is kept.

An explicit tool call does not need a model key:

```bash
tools/envs/reproduction/bin/python -m agentgs --help
tools/envs/reproduction/bin/python -m agentgs run --tool DeepEarthQueryUSGSCatalog --arguments '{"year":2011,"operation":"largest"}'
tools/envs/reproduction/bin/python -m agentgs run --tool GeoBigDataExecuteGeoSPARQL --arguments '{"source_id":591}'
tools/envs/reproduction/bin/python -m agentgs ui serve
```

Open `http://127.0.0.1:8000`. Tasks, events, failure records, and artifacts are under `outputs/`. Each tool call has its own artifact directory, and later calls can read files produced in the same task. ExecutePython is an explicit internal sandbox operation. It is not one of the 101 domain tools and is not a model candidate. Native Windows scientific execution is not accepted. Use WSL.

The [visualization console guide](src/agentgs/visualization/README.md) covers frontend builds, planning modes, tool chains, conversation and model-request inspection, artifact previews, and the paginated task API.

## Checks and evaluation

The agent uses a [four-role message protocol](docs/message-protocol.md). Its system
prompt belongs to the framework; dataset instructions are appended in a dedicated
block. Stage reminders travel with tool feedback. Bash exploration returns the
model's final answer directly, without a second generation pass.

```bash
tools/envs/reproduction/bin/python -m pytest -q
.venv/bin/ruff check src scripts
.venv/bin/ruff format --check src scripts
```

Historical baseline collection, tool-coverage, and worker helper scripts have been removed. Existing comparison records and test fixtures remain. Production code does not depend on the old source tree.

Saved predictions can be scored offline. The primary FAS uses `geotooleval-answer-v2`: it removes common answer wrappers, parses DMS coordinates and calendar dates, compares quantities with compatible units and reference precision, and resolves explicit choice conclusions or unique equivalent option values. Ambiguous scalar answers and conflicting options do not receive credit. Typed Earth, ocean-summary, and paleo records bind values to their requested fields; query context such as the dlnVs sample depth is not an additional answer value. Scalar scientific text and labelled distance/bearing pairs are also parsed numerically.

Each result includes `answer_score`, `legacy_answer_score`, and an independent `answer_format` diagnostic. Reports include `legacy_micro_fas`, `format_valid_rate`, and `answer_review_count`. Format mismatch alone does not fail a completed agent execution; missing answers, unresolved values, and incomplete execution still do. These runtime statuses do not prove factual correctness. Historical lexical FAS remains available in `evaluation/metrics/legacy_answers.py`; optional historical compatibility scores are separate from the primary FAS.

Unsupported structured-record families are explicitly marked `lexical_record_fallback` and `needs_review`: their scores remain lexical overlap, not semantic or artifact verification. Free-text items use normalized lexical equality with duplicate-aware F1; arbitrary paraphrases and the scientific validity of generated files are not judged by a model. No additional model calls are made during scoring. The historical sample results elsewhere in this README use the original FAS and must not be compared directly to the new protocol without rescoring.

Evaluation uses a `Conversation`. Execution is parsed from standard `tool_calls` / `tool_call_id` relations by `execution_events()`. Each benchmark attempt saves `conversation.json` and `messages.jsonl`, including the original arguments, runtime-bound arguments, reasoning, and model-request events. An offline prediction can provide a `messages` list or a full `conversation` object. A prediction that only has an old `trace` needs the original messages first.

The current scoring protocol is `geotooleval-conversation-v3`. It counts domain calls in the current user turn, including failed retries, and excludes planning calls and comparison branches. Parallel replies are scored in declared call order. Arguments are scored from the inputs that actually ran. Model usage is derived from request records in the conversation when possible. If a request is unfinished, failed, or its retry usage is incomplete, the cost score and the combined score stay unknown. Composite metrics use the primary answer FAS. A new-protocol experiment uses a new directory. An old experiment can be rescored offline from the saved messages.

```bash
tools/envs/reproduction/bin/python -m agentgs evaluate score predictions.jsonl --allow-partial --output outputs/evaluations/scored.json
```

A small live model experiment uses the same benchmark entry and keeps every failure. It does not rerun questions because of the score:

```bash
tools/envs/reproduction/bin/python -m agentgs evaluate benchmark --ids 61,86 --concurrency 2 --timeout 240 --output outputs/evaluations/small-run
```

The 603-question entry is `agentgs evaluate benchmark`. Omitting `--ids` evaluates every question. Resume checks source, data, and config identity. A different version uses a new directory. Gold answers are used only by the evaluator and are not sent to the agent.

Live model checks used gpt-5.6-terra. Q61 had FAS=1. Q86 had FAS=0.75 because the final rounding of 218.015 differed from the frozen answer. After the Geo argument fix, Q106 and Q111 both had FAS=1. Failed attempts and timeout reports remain in `outputs/verification/`. These small samples do not show that the full experiment reproduces performance.

## Frontend and packages

```bash
tools/envs/reproduction/bin/python scripts/build/build_api_types.py
tools/envs/reproduction/bin/python scripts/build/build_frontend.py
uv build
```

The products are `dist/agentgs-0.2.0-py3-none-any.whl` and `dist/agentgs-0.2.0.tar.gz`. The wheel contains the tool catalog, declarations, the GeoQuestions selection list, and frontend static files. The sdist contains environment constraints, prepare scripts, algorithm contracts, tests, and non-secret config. Raw data, secrets, tool environments, and `node_modules` are not packaged.

## Run 603 questions on a server

The portable data package `dist/agentgs-data-final.tar` and the per-file checksum list `dist/agentgs-data-final.manifest.json` are prepared locally. The package contains only the frozen data, model, KG, and current format version required by the runtime and the evaluator. It has no secrets, old source, or local sample fixtures. Copy the source sdist, this package, and the separately configured API settings to the server:

```bash
tar -xzf agentgs-0.2.0.tar.gz
cd agentgs-0.2.0
tar -xf ../agentgs-data-final.tar
uv sync --locked --extra visualization
.venv/bin/python scripts/tools/install_reproduction.py --python 3.12
cp .env.example .env
# On the server, set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL in .env.
tools/envs/reproduction/bin/python -m agentgs config validate
tools/envs/reproduction/bin/python -m agentgs evaluate benchmark --output outputs/evaluations/server-603
```

The last command is the live 603-question model experiment and is started on the server when requested. Existing results resume by default. `--retry-failed` keeps a retry attempt. A change of data, code, or model config uses a new output directory.

## Sample data-flow audit

The audit report is `outputs/verification/leakage-audit.json`. The 10 sampled requests did not inject gold answers, gold chains, or score feedback. All 22 rebuilt user prompts match the recorded SHA256. Rebuilt messages are in `outputs/verification/audited-prompts/`. The original HTTP request bodies were not kept. The audit adds no network calls.

This sample is an engineering regression. GeoQuestions Q106 is authorized to read the official materialized answer and reference query, so it is not an independent unknown-answer or RDF geometry score. Ocean Q347 reads a 10-row benchmark-derived cache from the original SOCAT data, not from private labels, and does not recompute the full original grid at runtime. Q61 and Q106 were used while debugging, and the format file also contains later upstream diagnostic edits. A perfect score on these 10 questions is not a strict held-out generalization accuracy.

Generic scientific resource resolution refuses evaluation labels, historical scores, answer-format files, and the Geo replay data directory. The dedicated Geo snapshot tool keeps its authorized access, and the real natural-hazard table stays readable. The sample did not access those private files. This boundary is still not a complete OS sandbox.

The GEM workbook is fully converted to Git-tracked `data/gem.jsonl.gz`. `prepare_gem.py` expands it to `data/gem.jsonl`, and energy preparation reads the JSONL directly. A new workbook can be imported with `.venv/bin/python scripts/prepare/prepare_gem.py --workbook data/raw/gem/<file>.xlsx`. The fixed query list, regions, and geographic reference values are in `data/geoquestions/profile.json` and `data/reference/`, and they are Git-managed.
