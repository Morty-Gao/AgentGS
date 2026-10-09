import { useEffect, useState } from "react";
import type { TaskPage, TaskStatus } from "../contracts";
import { usePolling } from "../hooks";
import { date, statuses, taskTitle } from "../format";
import { Empty, Pager, Status } from "./Common";

export function TaskList({ selected, onSelect, onNew, revision }: {
  selected: string; onSelect: (id: string) => void; onNew: () => void; revision: number;
}) {
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(0);
  useEffect(() => {
    const timer = setTimeout(() => { setQuery(search); setPage(0); }, 250);
    return () => clearTimeout(timer);
  }, [search]);
  const params = new URLSearchParams({ search: query, limit: "30", offset: String(page * 30) });
  if (status) params.set("status", status);
  const { data, error } = usePolling<TaskPage>(`/tasks?${params}`, 4000, revision);
  return <aside className="sidebar">
    <div className="brand"><span className="brand-symbol" aria-hidden="true">A</span><div>AgentGS<small>地学任务工作台</small></div></div>
    <button className="primary" onClick={onNew}>＋ 新建任务</button>
    <div className="sidebar-heading"><h2>任务记录</h2><span>{data ? Object.values(data.counts).reduce((a, b) => a + b, 0) : "—"}</span></div>
    <label className="sr-only" htmlFor="task-search">搜索任务</label>
    <input id="task-search" type="search" placeholder="搜索问题、工具或 ID" value={search} onChange={e => setSearch(e.target.value)} />
    <label className="sr-only" htmlFor="task-status">任务状态</label>
    <select id="task-status" value={status} onChange={e => { setStatus(e.target.value); setPage(0); }}>
      <option value="">全部状态</option>{Object.entries(statuses).map(([key, label]) =>
        <option key={key} value={key}>{label} · {data?.counts[key as TaskStatus] ?? 0}</option>)}
    </select>
    {error && <p className="error" role="alert">{error}</p>}
    {!data && !error && <p role="status">正在加载任务…</p>}
    <nav aria-label="任务列表" className="task-list">
      {data?.items.map(task => <button key={task.id} className={`task-row ${selected === task.id ? "selected" : ""}`}
        aria-current={selected === task.id ? "page" : undefined} onClick={() => onSelect(task.id)}>
        <span className="task-title">{taskTitle(task)}</span><Status value={task.status} />
        <small>{date(task.created_at)}<br /><code>{task.id.slice(0, 12)}</code></small>
      </button>)}
      {data?.items.length === 0 && <Empty>{search || status ? "没有匹配的任务。" : "暂无任务，点击上方按钮开始。"}</Empty>}
    </nav>
    <Pager page={page} total={data?.total ?? 0} size={30} onChange={setPage} />
    <footer>任务 · 会话 · 执行证据<a href="/docs" target="_blank" rel="noreferrer">API 文档 ↗</a></footer>
  </aside>;
}
