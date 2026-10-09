# Execution and experimental controls

## Plans and scientific validation

The planning function accepts `steps`, not a list of tool names. Each step has
an `id`, `tool`, `purpose`, `depends_on`, `arguments`, and `input_bindings`.
Bindings identify an earlier `step_id` and a path within its observed JSON output.
Optional `skill_id` and `capability_id` identify the declared implementation scope.
Unknown/forward dependencies and duplicate IDs are rejected before execution.

```json
{"steps":[
  {"id":"mean","tool":"RemoteSensingMean","purpose":"Mean of the supplied values",
   "arguments":{"x":[1,3]},"depends_on":[],"input_bindings":{}},
  {"id":"difference","tool":"RemoteSensingDifference","purpose":"Compare with the reference",
   "arguments":{"b":1},"depends_on":["mean"],
   "input_bindings":{"a":{"step_id":"mean","path":["value"]}}}
]}
```

Capability prerequisites retain their Skill scope. Expansion computes dependency
closure and a topological order with cycle and missing-declaration diagnostics.
Lexical support retrieval is heuristic; unmatched query words are diagnostics,
not proof of missing scientific operations. Retrieved strategies are candidates:
model selection adopts scoped goal operations before expansion, and model
approval is required for proposed support additions. No-match retrieval or an
empty adoption falls back to tool planning.

Grounding rejects missing required capabilities, invalid scoped ordering, and
invalid explicit instance dependencies. Model-supplied prerequisites can satisfy
these checks even with automatic predecessor expansion disabled. Schema presence
is recorded separately from validation: concrete argument types and references
are validated at execution, not certified from prose descriptions. Successful
execution or evidence consistency does not establish a scientifically correct
answer; benchmark answer scoring remains independent.

Replanning does not reuse steps by tool-name prefix. Reuse requires complete
numeric arguments, unchanged instance semantics and verified upstream reuse keys.
Only explicitly allowlisted pure scalar operations are reusable; file-backed,
implicit-input and external-data computations are rerun. Rewriting a plan may
insert steps while preserving existing identities and bindings, but cannot
silently remove or reorder declared instances.

## Ablations and candidate access

[Module ablations](module-ablations.md) defines the composable interventions.
`tool_only` removes the entire Skill/Capability layer; it is not an isolated
representation ablation. `data_only` is the Bash system baseline.

For an additional equal-candidate comparison, set `[agent] candidate_pool =
"all_registered"` in every compared run. Both scientific modes then receive the
same registered tool definitions, independent of Skill-driven retrieval expansion.
The default `retrieved` setting retains each system's retrieval policy. Large
candidate sets must fit the configured context window; required contracts are not
silently truncated to fit.

`routing_hints` and `binding_assistance` expose the shared heuristic aids.
`routing_hints=false` disables inferred prompt hints and the KG retriever's
domain seed injection for that request, without mutating a shared retriever.
`allow_rule_planning=false` alone does not mean that all deterministic assistance
is disabled. Keep these controls identical across the relevant comparisons.

Prepared embeddings cover the public question plus resource locator, tool
functionality relations, and graph-expansion contexts. Every mode shares the same
input construction and encoder. Runtime sessions warm the CPU encoder once and
serve isolated task workers through loopback; dynamic misses are memoized.
Vector-file hashes are part of resume identity, so rebuilding the cache requires
a new experimental output directory. Scientific source-data gaps remain recorded
task failures and are not filled with inferred values.

Bridge direct-path comparisons are opt-in via `bridge_diagnostics=true` and are
off in normal experiments. Diagnostic executor requests have a separate counter.

## Data availability

Use `--benchmark-scope common_data` for all modes when comparing against Bash.
This predeclared selection excludes `GeoBigDataTools` materialized-query lookup
tasks, whose snapshot service is unavailable inside Bash. The protocol records
the selected IDs, exclusion reasons and access strata. `data_only` with `full`
scope rejects a selection containing these tasks instead of scoring an unequal
information condition as an agent failure. Scientific-tool-only comparisons can
still use `full` and retain the explicit lookup stratum.
Offline `evaluate score` accepts the same `--benchmark-scope`; completeness and
expected domain counts use the declared population rather than the excluded tasks.

This policy addresses the known snapshot asymmetry; resource presence and actual
task solvability still require dataset preflight. It does not claim that every
remaining task has identical difficulty with a shell and a scientific tool.

## Prompts, budgets and measurements

Shared final-answer rules live in the framework system prompt. Dataset rules
remain in its final `dataset-instructions` block. Stage control appears in
`system-reminder`; observations and structured constraints appear separately in
`runtime-context`. JSON contracts and input bindings are never cut in half.

The client checks the complete request and reserved output against `context_window`
using a conservative UTF-8 byte bound plus framing, not a claimed exact tokenizer
count. It can omit complete earlier user turns and older tool-result bodies while
preserving current tool-call pairs and the latest feedback. Omission is explicit.
If mandatory context still does not fit, the request fails before reaching the API.
Actual transport payloads remain in the audit log after any context reduction.

The default per-task caps are 200 model requests and 262144 total tokens. HTTP
attempts count separately; a response's reported usage replaces its conservative
reservation, while unknown consumption remains conservatively reserved. Configure
these caps explicitly for experiments. HTTP retry, native-protocol repair, tool
execution repair and replanning have distinct scopes. Transport exhaustion and
budget failures do not trigger scientific execution replay.

Bash budget exhaustion sets `action_budget_exhausted`; it does not alone decide
completion. Both normal and budget-limited final answers use the same evidence
validation. `answer_evidence_status` describes consistency with observed outputs,
not ground-truth correctness. `provenance.execution_cost` separates regular and
diagnostic executor requests from model requests; transport attempts and token
accounting are recorded under `provenance.budget`.
