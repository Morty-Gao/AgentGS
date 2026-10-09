# Agent message protocol

AgentGS sends exactly one leading `system` message and uses only `system`,
`user`, `assistant`, and `tool` roles. A user task advances through assistant
function calls and their tool results. An accepted textual answer ends the turn.

## Prompt ownership

`agent/prompts` constructs the framework system prompt. It defines the selected
planning mode, evidence requirements, tool protocol and final-answer rules.
`TaskInput.dataset_instructions`, when present, is escaped and appended inside
`<dataset-instructions>` at the end of that system prompt. Dataset formatting
requirements apply to the final answer, not intermediate function calls.

Dataset adapters provide instructions, public resources and public output
contracts explicitly. Expected answers and reference tool chains remain in the
evaluator. Ordinary runtime tasks do not automatically load a GeoToolEval prompt
or look up an answer contract by question text. The CLI accepts
`--dataset-instructions TEXT` or `--dataset-instructions @file`; task JSON can
also supply the field.

The initial user message contains the original question and a `<task-context>`
block with entry-stage context, resources and the output contract. The original
question remains in `TaskInput.question` and message metadata. The system prompt
is fixed for the run. Imported system messages must match it; changing dataset
or mode instructions requires a separate conversation.

## Stage feedback

Before the next model request, the controller attaches its instruction to the
last unsent tool result:

```text
<tool-result>
{"status":"ok","mean":12.5}
</tool-result>

<system-reminder>
Stage: final_answer_generation
Use the observed measurements and the public output format. No tools remain.
</system-reminder>
```

All model-visible tool results are escaped inside `tool-result`. Only the final
result of a batch carries a reminder. The reminder is a framework convention
described in the system prompt, not a new API role. Raw output is retained in
`raw_tool_output` metadata and remains the evidence used by execution analysis.
Sent feedback is never edited to introduce another stage. Every real tool call
must receive exactly one result before another model request is allowed.

`submit_tool_plan` is a control call; its response reports syntactic validation
and pending execution validation. The next reminder supplies the grounded action.
Control calls do not count as scientific tool invocations.

Bash exploration uses the same protocol with `tool_choice="auto"`. Its textual
answer is validated and returned directly. Exhausting the Bash budget returns
the last tool result with final-answer instructions, followed by one request
without tools. There is no additional answer-generation request after an
accepted answer, including when its format validation fails.

## Errors and evidence

Invalid arguments and rejected native calls receive real tool feedback. A text
response in a stage requiring a function call cannot receive a fabricated tool
result. It is recorded as `model.protocol_failed`, with the complete raw response
in `model.response`, and retried using the identical legal request within the
argument-attempt budget. Rejected attempts retain their usage. Exhaustion ends
the run; it does not rerun successful scientific operations.

`RuntimeEvent` is a separate audit type interleaved in the durable log, never a
model message. `model.started` stores an immutable request-message snapshot;
`model.response` stores the provider request payload and original response without
HTTP authentication headers. Request reconstruction uses the stored snapshot.
Transport-level request and response events additionally retain every HTTP attempt
before response parsing, including malformed responses and provider errors. The
recorder is scoped to the asynchronous task, so concurrent runs do not share it.
Current-turn reasoning is preserved; only a real subsequent user turn changes
the reasoning export boundary.

Tool execution results are saved before feedback preparation. Checkpoints expose
whether a result has been sent to the model. Importing unfinished tool turns or
unresolved calls as a fresh user task is rejected to avoid accidental replay.
Benchmark retries remain explicit new attempts; the runtime does not promise
in-place continuation of an arbitrary interrupted controller stack.

## Archives and validation

Conversation schema `2` and result `conversation_schema=2` identify this protocol.
Benchmark manifests include the protocol and public-input identity. Schema-1
records are not silently rewritten or resumed. `messages.archive.read_archive`
reads historical JSON/JSONL as raw records for inspection, preserving their roles.

Validation covers request contents, raw evidence, replay snapshots, malformed
calls, Bash termination, new user turns and dataset isolation. Model-template
rendering verifies that the provider receives the instructions; only live
inference can establish a particular checkpoint's actual tool-use reliability.
