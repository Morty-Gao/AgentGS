import { useState } from "react";
import type { FormEvent } from "react";
import type { ConsoleConfig, SkillSpec, TaskRecord, TaskSubmission, ToolSpec } from "../contracts";
import { submitTask } from "../api";
import { Json } from "./Common";

function objectJson(text: string, name: string): Record<string, unknown> {
  const value: unknown = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name} 必须是 JSON 对象`);
  return value as Record<string, unknown>;
}

export function Composer({ tools, skills, config, onSubmitted }: {
  tools: ToolSpec[]; skills: SkillSpec[]; config?: ConsoleConfig; onSubmitted: (task: TaskRecord) => void;
}) {
  const [mode, setMode] = useState("question");
  const [question, setQuestion] = useState("");
  const [planning, setPlanning] = useState("");
  const [model, setModel] = useState("");
  const [effort, setEffort] = useState("");
  const [budget, setBudget] = useState("");
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [calls, setCalls] = useState("[]");
  const [parallel, setParallel] = useState(false);
  const [toolId, setToolId] = useState("");
  const [extra, setExtra] = useState("{}");
  const [raw, setRaw] = useState('{\n  "question": "",\n  "planning_mode": "tool_only"\n}');
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const tool = tools.find(t => t.id === toolId);
  const effectiveMode = planning || config?.algorithm.planning_mode;
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(""); setBusy(true);
    try {
      let input: TaskSubmission;
      if (mode === "json") input = objectJson(raw, "TaskInput") as TaskSubmission;
      else {
        input = objectJson(extra, "补充参数") as TaskSubmission;
        if (mode === "question") {
          input = { ...input, question, ...(planning ? { planning_mode: planning as TaskSubmission["planning_mode"] } : {}),
            ...(model.trim() ? { model: model.trim() } : {}), ...(effort.trim() ? { effort: effort.trim() } : {}),
            ...(budget ? { max_model_calls: Number(budget) } : {}),
            ...(effectiveMode === "skill_guided" ? { skill_ids: selectedSkills } : {}),
          };
        } else {
          const value: unknown = JSON.parse(calls);
          if (!Array.isArray(value) || value.length === 0) throw new Error("请添加至少一个工具调用");
          input = { ...input, calls: value, parallel_tools: parallel };
        }
      }
      const task = await submitTask(input);
      onSubmitted(task);
    } catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  };
  const appendTool = () => {
    if (!tool) return;
    try {
      const current: unknown = JSON.parse(calls);
      if (!Array.isArray(current)) throw new Error("调用列表必须是数组");
      const properties = tool.input_schema.properties as Record<string, { default?: unknown }> | undefined;
      const args = Object.fromEntries(Object.entries(properties ?? {})
        .filter(([, value]) => value.default !== undefined).map(([key, value]) => [key, value.default]));
      let nextId = current.length + 1;
      while (current.some(call => call?.id === `call_${nextId}`)) nextId += 1;
      setCalls(JSON.stringify([...current, { id: `call_${nextId}`,
        tool: tool.id, arguments: args, depends_on: [] }], null, 2));
      setError("");
    } catch (e) { setError(String(e)); }
  };
  return <section className="panel composer">
    <div className="section-heading"><div><p className="eyebrow">NEW RUN</p><h2>新建地学任务</h2></div></div>
    <p className="muted">配置问题或显式工具链，运行后查看完整执行证据。</p>
    <div className="segments" aria-label="输入方式">{[["question", "问题"], ["tools", "工具链"], ["json", "完整 JSON"]].map(([key, label]) =>
      <button key={key} aria-pressed={mode === key} onClick={() => setMode(key)} disabled={busy}>{label}</button>)}</div>
    <form onSubmit={submit}>
      <fieldset disabled={busy}>
        {mode === "question" && <>
          <label>问题<textarea aria-label="问题" rows={5} required value={question} onChange={e => setQuestion(e.target.value)} placeholder="描述要解决的地学问题、数据范围与期望输出…" /></label>
          <div className="form-grid"><label>规划模式<select aria-label="规划模式" value={planning} onChange={e => setPlanning(e.target.value)}>
            <option value="">使用项目配置（{config?.algorithm.planning_mode ?? "加载中"}）</option>
            <option value="tool_only">tool_only · 工具规划</option><option value="skill_guided">skill_guided · Skill 引导</option>
            <option value="data_only">data_only · 数据与 Shell</option></select></label>
            <label>模型调用上限<input type="number" min={1} max={10000} value={budget} onChange={e => setBudget(e.target.value)} placeholder="使用默认预算" /></label>
            <label>模型<input value={model} onChange={e => setModel(e.target.value)} placeholder="使用 llm.toml 默认模型" /></label>
            <label>推理强度<input value={effort} onChange={e => setEffort(e.target.value)} placeholder="使用模型默认配置" /></label></div>
          {effectiveMode === "skill_guided" && <fieldset className="skill-options"><legend>限定 Skill（留空自动选择）</legend>
            {skills.map(skill => <label key={skill.id} className="check"><input type="checkbox" checked={selectedSkills.includes(skill.id)}
              onChange={e => setSelectedSkills(ids => e.target.checked ? [...ids, skill.id] : ids.filter(id => id !== skill.id))} />
              <span>{skill.id}<small>{skill.description}</small></span></label>)}</fieldset>}
        </>}
        {mode === "tools" && <>
          <label>工具目录<select aria-label="工具目录" value={toolId} onChange={e => setToolId(e.target.value)}><option value="">选择一个工具…</option>
            {tools.map(t => <option key={t.id} value={t.id}>{t.category ? `${t.category} / ` : ""}{t.id}</option>)}</select></label>
          {tool && <div className="tool-description"><p>{tool.description}</p><Json value={tool.input_schema} label="参数 Schema" />
            {tool.input_description && <p className="muted">{tool.input_description}</p>}
            <button type="button" onClick={appendTool}>添加到调用列表</button></div>}
          <label>调用列表<textarea aria-label="调用列表" className="code" rows={12} required value={calls} onChange={e => setCalls(e.target.value)} spellCheck={false} /></label>
          <p className="muted">填写 arguments；depends_on 引用上游调用 ID。工具参数以 Schema 为准。</p>
          <label className="check"><input type="checkbox" checked={parallel} onChange={e => setParallel(e.target.checked)} />并行执行独立调用</label>
        </>}
        {mode === "json" ? <><label>TaskInput<textarea aria-label="TaskInput" className="code" rows={16} value={raw} onChange={e => setRaw(e.target.value)} spellCheck={false} /></label>
          <p className="muted">支持 messages、input_resource、file_path_list、output_contract 及全部任务字段。</p></>
          : <details><summary>补充任务参数</summary><p className="muted">可设置数据资源、输出契约、重试预算和规则规划等参数。上方已填写字段优先。</p>
            <label>参数 / JSON<textarea aria-label="参数 / JSON" className="code" rows={7} value={extra} onChange={e => setExtra(e.target.value)} spellCheck={false} /></label></details>}
      </fieldset>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="form-footer"><span className="muted">{config ? `超时 ${config.runtime.default_timeout_seconds}s · 内存 ${config.runtime.default_rss_limit_mb} MiB` : "正在读取运行配置…"}</span>
        <button className="primary" disabled={busy}>{busy ? "正在提交…" : "提交任务 →"}</button></div>
    </form>
  </section>;
}
