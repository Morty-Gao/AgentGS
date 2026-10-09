import { useState } from "react";
import type { AgentResult, TaskRecord, ToolResult } from "../contracts";
import { artifactUrl, request, taskPath } from "../api";
import { bytes, date, terminal } from "../format";
import { useEvents, useTask } from "../hooks";
import { Copy, Empty, Json, Pager, Status } from "./Common";
import { ConversationView } from "./ConversationView";
import { EventsView } from "./EventsView";
import { ArtifactPreview } from "./ArtifactPreview";

type Tab = "overview" | "tools" | "conversation" | "models" | "events" | "artifacts";

function ToolResults({ results, taskId }: { results: ToolResult[]; taskId: string }) {
  const [page, setPage] = useState(0);
  if (!results.length) return <Empty>暂无工具结果。已发起的调用可在会话和事件中查看。</Empty>;
  return <><div className="tool-results">{results.slice(page * 30, (page + 1) * 30).map((result, index) =>
    <article className="result-card" key={`${result.call_id}-${page * 30 + index}`}>
      <div className="section-heading"><h3>{result.tool}</h3><span className={`badge ${result.succeeded ? "succeeded" : "failed"}`}>{result.succeeded ? "执行成功" : "执行失败"}</span></div>
      <div className="metadata"><code>{result.call_id}</code><span>{result.source_type}</span></div>
      {result.error && <p className="error">{result.error.code} · {result.error.message}</p>}
      <Json value={result.data} label="结构化结果" />
      {result.raw_output !== null && <Json value={result.raw_output} label="原始输出" />}
      {result.artifacts.map(artifact => <a className="artifact-inline" key={artifact.id} href={artifactUrl(taskId, artifact.id)}>{artifact.path} ↓</a>)}
    </article>)}</div><Pager page={page} total={results.length} size={30} onChange={setPage} /></>;
}

function Artifacts({ result, id }: { result: AgentResult | null; id: string }) {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const artifacts = [...new Map((result?.tool_results ?? []).flatMap(r => r.artifacts).map(a => [a.id, a])).values()]
    .filter(a => `${a.path} ${a.source}`.toLocaleLowerCase().includes(search.toLocaleLowerCase()));
  return <><label>搜索产物<input type="search" value={search} onChange={e => { setSearch(e.target.value); setPage(0); }} placeholder="文件名或来源" /></label>
    {artifacts.length === 0 && <Empty>暂无匹配的产物。</Empty>}
    <div className="artifact-grid">{artifacts.slice(page * 30, (page + 1) * 30).map(a => <article className="result-card" key={a.id}>
      <span className="file-type">{a.media_type}</span><h3>{a.path}</h3><p className="muted">{bytes(a.bytes)} · {a.source}</p>
      <details><summary>校验信息</summary><code className="digest">SHA256 {a.sha256}</code></details>
      <a className="button" href={artifactUrl(id, a.id)}>下载文件 ↓</a>
      <ArtifactPreview taskId={id} artifact={a} />
    </article>)}</div><Pager page={page} total={artifacts.length} size={30} onChange={setPage} /></>;
}

function Models({ result }: { result: AgentResult | null }) {
  const [page, setPage] = useState(0);
  const usage = result?.usage ?? [];
  const requests = (result?.messages ?? []).filter(m => m.role === "event" && m.metadata.type === "model.started");
  return <><p className="muted">{usage.length} 次模型调用。用量来自 provider 回执；未返回的字段保持为空。</p>
    {usage.length === 0 && <Empty>暂无模型调用记录。显式工具任务无需调用模型。</Empty>}
    {usage.slice(page * 30, (page + 1) * 30).map((row, index) => {
      const requestEvent = requests.find(m => m.metadata.index === row.call);
      return <article className="result-card" key={page * 30 + index}>
        <div className="section-heading"><h3>#{String(row.call ?? page * 30 + index + 1)} · {String(row.stage ?? "model")}</h3><span className="badge">{String(row.status ?? "未知")}</span></div>
        <p className="muted">{String(row.response_model ?? "模型未记录")}</p>
        <Json value={row} label="调用状态与用量" />
        {requestEvent && <Json value={requestEvent.metadata} label="模型请求（消息、工具与参数）" />}
      </article>;
    })}<Pager page={page} total={usage.length} size={30} onChange={setPage} /></>;
}

function Overview({ task }: { task: TaskRecord }) {
  const result = task.result;
  return <><div className="metrics">
    <div><span>工具结果</span><strong>{result?.tool_results.length ?? 0}</strong></div>
    <div><span>模型调用</span><strong>{result?.usage.length ?? 0}</strong></div>
    <div><span>内存峰值</span><strong>{bytes(task.peak_rss_bytes)}</strong></div>
    <div><span>运行限制</span><strong>{task.spec.timeout_seconds}s</strong><small>{task.spec.rss_limit_mb} MiB</small></div>
  </div>
    {result && <div className="outcomes"><span>完成情况：<strong>{result.completion_status}</strong></span>
      <span>工具链：<strong>{result.tool_chain_completed ? "完成" : "未完成"}</strong></span>
      <span>回答生成：<strong>{result.answer_generation_status}</strong></span></div>}
    {task.error && <p className="error" role="alert"><strong>{task.error.code}</strong><br />{task.error.message}</p>}
    <section className="answer"><div className="section-heading"><h3>最终回答</h3>{result?.answer && <Copy text={result.answer} />}</div>
      {result?.answer ? <pre>{result.answer}</pre> : <Empty>{terminal(task.status) ? "本次任务没有生成最终回答，可查看执行记录和错误。" : "任务进行中，执行记录随运行进度更新。"}</Empty>}</section>
    <Json value={task.spec.input} label="任务输入与配置" />
    {result && <><p className="muted">Algorithm · {result.algorithm_id || "未记录"} / Conversation schema · {result.conversation_schema}</p>
      <Json value={result.provenance} label="来源与执行依据" /></>}
  </>;
}

export function TaskDetail({ id, revision, onChange }: { id: string; revision: number; onChange: () => void }) {
  const { data: task, error } = useTask(id, revision);
  const stream = useEvents(id);
  const [tab, setTab] = useState<Tab>("overview");
  const [cancelling, setCancelling] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);
  const [actionError, setActionError] = useState("");
  const cancel = async () => {
    setCancelling(true); setActionError("");
    try {
      await request(taskPath(id) + "/cancel", { method: "POST" });
      setCancelRequested(true); onChange();
    } catch (e) { setActionError(String(e)); }
    finally { setCancelling(false); }
  };
  if (!task) return <section className="panel"><Empty>{error || "正在加载任务…"}</Empty>{error && <button onClick={onChange}>重试</button>}</section>;
  const result = task.result;
  const tabs: [Tab, string, number | undefined][] = [
    ["overview", "概览", undefined], ["tools", "工具执行", result?.tool_results.length],
    ["conversation", "会话", result?.messages.length], ["models", "模型调用", result?.usage.length],
    ["events", "运行事件", stream.events.length], ["artifacts", "产物", undefined],
  ];
  return <section className="panel detail">
    <div className="section-heading"><div><p className="eyebrow">TASK INSPECTOR</p><h2 className="run-title">{task.spec.input.question || task.spec.input.calls.map(c => c.tool).join(" → ") || "任务详情"}</h2></div><Status value={task.status} /></div>
    <div className="metadata"><code>{id}</code><time>{date(task.created_at)}</time><span>{task.spec.input.planning_mode ?? "项目默认模式"}</span></div>
    <div className="detail-actions"><span role="status" className="muted">{stream.connection}{!terminal(task.status) && " · 显示最近已保存的进度"}</span>
      <a className="button" href={`/api${taskPath(id)}/export`}>导出任务 JSON</a>
      {!terminal(task.status) && <button className="danger" disabled={cancelling || cancelRequested} onClick={() => void cancel()}>{cancelRequested ? "已请求取消" : cancelling ? "取消中…" : "取消任务"}</button>}</div>
    {(error || actionError) && <p role="alert" className="error">{error || actionError}</p>}
    <div className="tabs" aria-label="任务详情视图">{tabs.map(([key, label, count]) =>
      <button key={key} aria-pressed={tab === key} onClick={() => setTab(key)}>{label}{count !== undefined && <span>{count}</span>}</button>)}</div>
    <div className="tab-content">
      {tab === "overview" && <Overview task={task} />}
      {tab === "tools" && <ToolResults results={result?.tool_results ?? []} taskId={id} />}
      {tab === "conversation" && <ConversationView messages={result?.messages ?? task.spec.input.messages} />}
      {tab === "models" && <Models result={result} />}
      {tab === "events" && <EventsView {...stream} />}
      {tab === "artifacts" && <Artifacts result={result} id={id} />}
    </div>
  </section>;
}
