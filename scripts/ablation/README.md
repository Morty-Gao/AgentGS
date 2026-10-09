# Module ablation experiment

This directory defines a **purposefully selected, paired diagnostic study** of
the module controls. It does not estimate unbiased performance over all 603
GeoToolEval questions. `tool_only`, `data_only`, Earth-Agent and GeoAgent belong
to separate system comparisons, not this module table.

## Questions and hypotheses

| Cell | Change from Full | Interpretation |
| --- | --- | --- |
| Full | All modules enabled | Paired reference |
| A1 | Flat tool-planning context | Equal-declaration planning-representation control; shared requirement preparation remains |
| A2 | No automatic predecessor expansion | Compensation control: completion may restore every omitted prerequisite |
| A3 | No support insertion or bridging | Contribution of automatic support insertion |
| A4 | No directed bridging | Included only when the frozen offline audit accepts real declared bridge paths |
| A5 | First eligible tool in fixed ID order | Selection policy, keeping requirements and instance scopes |
| A6 | No execution recovery | Failure repair/replanning, with initial planning and binding retained |
| BIPARTITE | Incidence representation | Same declaration information expressed through strategy/capability nodes and edges |
| A2_A3 | No expansion or completion | Predecessor effect when completion cannot compensate |
| A3_A6 | No completion or recovery | Whether recovery compensates for missing automatic support |
| A5_A6 | Fixed selection and no recovery | Whether recovery compensates for tool-selection errors |

The three interactions complete the corresponding 2x2 designs using shared
Full and single-ablation cells. For interventions A and B report
`Y_Full - Y_A - Y_B + Y_AB`, and the conditional A effect `Y_B - Y_AB`.
Do not add A3+A4: A3 already disables bridging.

`prepare.py` records the production library's explicit I/O declarations, bridge
bindings and accepted task-level bridge paths. It promotes A4 from conditional
to primary only when at least one selected task admits a bridge; otherwise A4
is deferred. Synthetic bridge unit tests prove implementation behavior, not
benchmark effectiveness. Even with an offline opportunity, report actual
successful bridge-bound calls in Full before attributing a contribution to A4.
Changing declarations to activate bridging requires a **new** frozen study and
a fresh Full; results from different library snapshots cannot be paired.

## Deterministic selection of 60 tasks

`prepare.py` audits all 603 tasks without model requests or scientific execution.
It preserves the existing random subset and sampler; the study has its own
frozen `inputs/study60.jsonl`.

1. Allocate six tasks to each of the ten domains.
2. In a mixed domain keep one single-step control, prefer up to three chains
   of at least four gold steps, and fill with two-to-three-step chains.
   Pure single-step domains retain six single-step controls; domains with no
   single-step task retain six multistep tasks.
3. Within each domain/chain stratum greedily cover new mechanism opportunities,
   capability IDs, reference tool identities and answer types. Novel mechanism,
   capability, tool and answer-type features have weights 8, 3, 2 and 1.
   Ties prefer more active mechanisms, then the smaller question ID.
4. Freeze every selected ID, selection reason, source hash and population audit.
   Optional `--ids-file manual.json` accepts an explicit list of 60 unique IDs,
   still requiring six tasks per domain. No correctness scores enter selection.

Reference chain length/tool identities are **selection and analysis metadata**.
The existing public-input builder keeps reference answers and chains out of
model requests. The runtime receives only the question, public resources and
public answer-format contract.

The current population allows a panel with **23 single-step, 24 two-to-three-step,
and 13 longer tasks**: 37 multistep tasks, versus 20 in the existing random panel.
Domains with only single-step tasks are not silently dropped. No claim of
population representativeness or post-hoc power is made.

## Locked controls and budget

`design.json` is the protocol source. All cells use `skill_guided`, the same
101 registered tool candidates (`all_registered`), model/effort/provider,
answer contracts, public resources, routing hints and binding assistance.
Rule routing and diagnostic bridge executions are disabled. Equal candidate
access isolates module logic from differences in candidate retrieval; results
must not be mixed with the deployed `retrieved` policy's baseline scores.

Each task has three planning attempts, three execution attempts per tool,
64 model requests and a total token-accounting cap of 262144. The latter follows
the runtime's reported-usage/conservative-reservation policy, not an exact
tokenizer estimate. Task/model/tool timeouts are 1800/600/120 seconds.
Concurrency is two tasks and at most four model/tool calls. Context window,
output-token limit, effort and transport/protocol retry settings come from the
frozen current configuration and are recorded, not silently replaced.

Outer task retries are **zero**, including for A6. Otherwise a whole-task retry
would reintroduce recovery outside the disabled module. HTTP and malformed-call
repair remain the shared protocol behavior. A terminal failed task stays in the
denominator; its saved answer is scored by the normal scorer. Missing tasks are
incomplete runs, never zeros or silently dropped pairs.

| Phase | Cells | Questions | Repeats | Task episodes |
| --- | ---: | ---: | ---: | ---: |
| Pilot | 8 primary cells when A4 is eligible | 10 (one per domain) | 1 | 80 |
| Main | Full, A1–A6, BIPARTITE | 60 | 3 | 1440 |
| Interactions | A2_A3, A3_A6, A5_A6 | 60 | 3 | 540 |
| Total with A4 eligible | | | | **2060** |

If A4 has no offline opportunity, omit its 10 pilot and 180 main episodes:
the corresponding totals are 70 pilot, 1260 main and **1870 overall**.

Pilot outputs never enter main estimates. Use the pilot to verify API/context
compatibility, resources, answer formatting and protocol completion; do not
tune prompts or select questions based on pilot FAS. Any method/configuration
change requires a fresh freeze and rerunning all compared cells.

The three repetitions are independent fresh trajectories, not retries or a
best-of-three selection. Average all repetitions. Sampling/scheduling/bootstrap
seeds do not imply provider-side decoding reproducibility; no unsupported model
seed is sent. Cell order is shuffled once and rotated between repetitions;
question order is shared across cells within a repetition. Run one cell at a
time to avoid different treatment cells competing for the same service quota.

## Commands

Run through a currently discovered CPU worker; the wrapper never allocates a pod
or falls back to the development machine. It applies the lab proxy environment.
If that proxy fails, follow the project's documented personal-proxy fallback;
do not silently change endpoints during a study.

```sh
# Offline audit, deterministic selection and immutable freeze. No model calls.
bash scripts/ablation/on_cpu_worker.sh prepare.py

# Preview only; --execute is necessary for real model requests.
bash scripts/ablation/on_cpu_worker.sh run.py --suite pilot

# Execute the pilot, then inspect complete results before main runs.
bash scripts/ablation/on_cpu_worker.sh run.py --suite pilot --execute
bash scripts/ablation/on_cpu_worker.sh summarize.py --suite pilot --require-complete

bash scripts/ablation/on_cpu_worker.sh run.py --suite primary --execute
bash scripts/ablation/on_cpu_worker.sh summarize.py --suite primary --require-complete
bash scripts/ablation/on_cpu_worker.sh run.py --suite interaction --execute
bash scripts/ablation/on_cpu_worker.sh summarize.py --suite all --require-complete

# Resume a specific cell/repetition, preserving completed and failed outcomes.
bash scripts/ablation/on_cpu_worker.sh run.py --suite primary --cell A5 --repeat 2 --execute
```

Default directory: `outputs/ablations/module-mechanism-study`. Use `--study` on
all commands to choose another directory. Preparation refuses to overwrite a
nonempty study. Code, declarations, project/model configuration and lockfile
hashes are checked before execution and after each cell. Prepared-data size/mtime
metadata is also checked; it is explicitly not a content checksum for every
payload, and excludes raw downloads and symlink trees. Keep prepared data and
the working tree frozen during execution. No credentials are copied into the
study manifest; only public model settings and file/endpoint hashes are saved.
Previewing a frozen schedule remains possible after source changes and reports
`source_matches_frozen_study=false`; actual execution is refused. To keep the
same panel in a newly frozen study, pass the previous
`inputs/selected_ids.json` to `prepare.py --ids-file` and use a new `--study`.
The offline opportunity audit uses deterministic terminal requirements without
model selection. It screens for possible activation, not a prediction of the
actual model's selected requirements or execution success.

## Analysis and reporting

The primary endpoint is **domain-macro FAS on the fixed 60-task diagnostic panel**.
The official AgentGS scorer supplies per-task FAS/TSF1/TOS; its composite
`final_score` is not substituted for this study's primary endpoint.

Report all three repeat macro scores, their standard deviation, pooled
question-mean macro FAS, overall micro FAS, multistep micro FAS, per-domain FAS,
mean TSF1/TOS, operational run failure rate, execution/reported completion rates,
model/physical-tool request counts and known token
usage. Unknown usage remains unknown, with its coverage reported.

For each main contrast use `100 * (Full - variant)` in percentage points.
Average repeated trajectories per question, then bootstrap **paired questions
within domain**, retaining repeats as a cluster (5000 draws). Do not treat 180
episodes as 180 independent benchmark questions. Interactions use the same
paired resampling. Confidence intervals are descriptive uncertainty within this
purposeful panel, not unbiased 603-question population intervals or
multiple-testing-adjusted significance claims.

Always inspect mechanism evidence beside FAS:

- A1/BIPARTITE: representation event and equal declaration digest; they are
  representation controls with shared preparation, not removal of every graph
  computation.
- A2: predecessor additions removed, final requirement changes and compensating
  support additions. Zero final changes means full compensation, not evidence
  that prerequisites are unimportant. Use A3 vs A2_A3 for the conditional effect.
- A3: support insertions stop while diagnosis remains visible.
- A4: distinguish offline accepted paths from successful bridge-bound runtime
  calls; no bridge contribution claim if Full never executes one.
- A5: record candidate scopes and whether fixed selection changes the proposed
  implementation, not merely whether the flag was set.
- A6: initial planning remains comparable and failed executions have no repair
  or execution-driven replanning. Failure-conditioned slices are descriptive,
  because failures are themselves affected by treatment.

`summarize.py` emits JSON and CSV, validates result hashes and requires complete
paired cells for contrasts. It can list pending cells before any run. It never
fills missing cells with fabricated scores. Do not select only successful
questions, the best repeat, or module-activated cases after observing scores.
Mechanism-enriched slices come from the frozen offline audit; natural-failure
slices must be labeled exploratory.
