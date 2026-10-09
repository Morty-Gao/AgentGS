# Method evaluation

These commands score a finished `predictions.jsonl`. They do not call a model
or run a scientific tool. Run them from the AgentGS project root.

```sh
uv run scripts/evaluation/agentgs/score.py outputs/evaluations/agentgs/predictions.jsonl --allow-partial --output outputs/evaluations/agentgs/scores.json
uv run scripts/evaluation/earthagent/score.py outputs/evaluations/earthagent/predictions.jsonl --allow-partial --output outputs/evaluations/earthagent/scores.json
uv run scripts/evaluation/geoagent/score.py outputs/evaluations/geoagent/predictions.jsonl --allow-partial --output outputs/evaluations/geoagent/scores.json
```

Earth-Agent scores the final attempt's ReAct observations. Plain text on the
final assistant turn is an answer. Tool output from earlier turns is not.

GeoAgent scores the selected execution. A failed search scores the last
attempted execution. Pruned branches do not contribute tool calls. An unmarked
answer must be the suffix of that execution's stdout. Model usage from the
other branches still counts.

AgentGS scores the answer string produced by its answering stage and the saved
conversation. It does not look for answer markers. When the run manifest says
`planning_mode` is `data_only`, the comparison is FAS only and tool match is
not applicable. Other modes use the shared composite score.
