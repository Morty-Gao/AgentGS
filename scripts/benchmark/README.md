# Fork benchmark adapters

Run these commands on a CPU worker from the AgentGS project root. All three read
the project's `llm.toml`. `--llm` on the Earth-Agent and GeoAgent commands
selects another provider/model configuration. `agentgs/run.py` uses the project
file directly.

```sh
uv run scripts/benchmark/earthagent/run.py --concurrency 4 --output outputs/evaluations/earth-new/predictions.jsonl
uv run scripts/benchmark/geoagent/run.py --concurrency 4 --output outputs/evaluations/geo-new/predictions.jsonl
uv run scripts/benchmark/agentgs/run.py --concurrency 4 --output outputs/evaluations/agentgs/predictions.jsonl
```

`agentgs/run.py` runs the project agent in `src/agentgs`. Earth-Agent and GeoAgent
keep their own loops. AgentGS reads `llm.toml` and `config.toml`. `--planning-mode`
accepts `skill_guided` (the default), `tool_only`, or `data_only`. `data_only`
forces rule planning off. Omitted runtime flags keep `RuntimeSettings`, including
concurrency 2 when `--concurrency` is omitted. `--plan-attempts`, `--tool-attempts`,
and `--command-attempts` stay unset unless passed, so `config.toml` supplies them.
`--answer-contracts` is on by default. Gold stays out of the runtime request.

A non-empty answer from a finished run is `status=succeeded`, including a
Not Completed result that still returned an answer. Timeouts, worker crashes,
and empty answers retry. Answer scores do not. Generation does not calculate FAS.

Omitting `--ids` selects all 603 unique GeoToolEval questions in ten domains.
`--ids 61,106` selects a sample. Labels and gold chains are unavailable to the
prompt builders. The 101 scientific tools and prepared inputs come from AgentGS;
its planning or answer-generation policies are not used by either fork.

Earth-Agent retains a ReAct tool/observation loop. GeoAgent retains executable
prefix nodes, P-UCB selection, independent candidate samples, API token
probabilities, stepwise simulation rewards, bounded repairs and replanning of
remaining work from a verified prefix. Every code-generation stage receives
actual tool schemas and result descriptions. Prefix observations are passed to
continuation and repair requests. GeoAgent requires API `logprobs` and JSON
schema output support and fails explicitly if the provider cannot supply them.
Search budgets can be set with `--rollouts` (default 200), `--top-k`,
`--repairs`, and `--timeout`. Earth-Agent's `--max-steps` counts model turns
and defaults to 200.

## Evidence and scoring

For `predictions.jsonl`, the durable run directory is `predictions.run/` next to
the output file. It contains a manifest, lock, summary and one directory per
question and attempt. Each attempt saves its result, canonical conversation,
model requests/responses, reasoning, logprobs, token usage, and errors. GeoAgent
also saves the search tree, pruned branches, selected program, each executed
program, stdout/stderr and scientific tool receipts. Each execution has its own
artifact directory. Earth-Agent artifacts live inside the attempt directory.
Tool outputs are kept in full, and scoring messages use the arguments that
actually reached the scientific adapter.

Both adapters extract explicit final-answer markers case-insensitively, accepting
indentation, Markdown headings/bold text, code fences, full-width colons, and JSON
answer fields. Multiline answers and explicit units are preserved. Empty or
conflicting JSON answers are rejected. `answer_extraction` records how the answer
was obtained; the original model response or program stdout remains in the trace.

Earth-Agent accepts plain text at its final assistant turn. GeoAgent also accepts
unmarked output when the program completes the final planned step successfully:
only the newly printed suffix after the verified prefix is used. Intermediate
prints, failed executions, and inconsistent prefix output cannot trigger this
fallback. It does not consult reference answers or pick values out of logs.
AgentGS stores the answer returned by its own answering stage and does not scan
the trace for a marker.

Predictions contain `conversation`, so generation does not calculate FAS.
Score each method with its own evaluation entry. Those commands apply the
method's answer and tool-call rules, then the shared metrics:

```sh
uv run scripts/evaluation/agentgs/score.py outputs/evaluations/agentgs/predictions.jsonl --allow-partial --output outputs/evaluations/agentgs/scores.json
uv run scripts/evaluation/earthagent/score.py outputs/evaluations/earth-new/predictions.jsonl --allow-partial --output outputs/evaluations/earth-new/scores.json
uv run scripts/evaluation/geoagent/score.py outputs/evaluations/geo-new/predictions.jsonl --allow-partial --output outputs/evaluations/geo-new/scores.json
```

GeoAgent scores the tool calls from the selected final execution; exploratory
branches remain in its trace and their model usage is counted. A failed search
keeps the final attempted execution and failure evidence. `status=succeeded`
means an answer was produced through the execution protocol; it does not mean
that the answer matches the reference. Use FAS and the other scoring metrics to
assess answer quality. Failed questions make the runner exit with status 1.

## Retry and resume

Failed questions automatically retry three times by default, for at most four
attempts. `--retries 0` disables automatic retries. Retries use the same worker
slot, with bounded backoff, and preserve every attempt before starting the next.
The final conversation counts model usage from all attempts; missing usage stays
unknown. Domain-call scoring uses the final execution. Answer scores never
trigger retries. `automatic_attempts_used` and `attempt_directories` record the
attempt count and receipts. Interruption recovery continues any pending retries
without resetting their budget.

Repeat the same command to retain completed and failed results and run only
unfinished questions. `--retry-failed` starts a new retry group for exhausted
failures while preserving their
earlier attempts in `predictions.run/attempts.jsonl`. The prediction file is an
atomic snapshot with one latest record per question, suitable for the normal
scorer after retries. Concurrent writers to the
same output are refused. Interrupted final JSONL fragments are repaired on
resume. A dead lock on the same worker can be recovered; locks from another
worker require verifying that worker first.

Resume checks source hashes, dataset hash, prepared-file sizes/timestamps, model,
endpoint and execution settings. Changed identities require a new directory.
Old outputs without a manifest cannot be silently combined with this protocol.
The prepared-file metadata check detects ordinary changes; it is not a
cryptographic receipt for every scientific input payload.
