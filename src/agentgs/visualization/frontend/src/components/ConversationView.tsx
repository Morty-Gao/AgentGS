import { useMemo, useState } from "react";
import type { Message, RuntimeEvent } from "../contracts";
import { date, pretty } from "../format";
import { Empty, Json, Pager } from "./Common";

type Entry = Message | RuntimeEvent;
export function ConversationView({ messages }: { messages: Entry[] }) {
  const [role, setRole] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const calls = useMemo(() => new Map(messages.flatMap(m => m.role === "event" ? [] :
    m.tool_calls.map(c => [c.id, c] as const))), [messages]);
  const rows = useMemo(() => messages.filter(m => (!role || m.role === role) &&
    (!search || JSON.stringify(m).toLocaleLowerCase().includes(search.toLocaleLowerCase()))), [messages, role, search]);
  const pageIndex = Math.min(page, Math.max(0, Math.ceil(rows.length / 30) - 1));
  return <>
    <div className="toolbar"><label>角色<select aria-label="角色" value={role} onChange={e => { setRole(e.target.value); setPage(0); }}>
      <option value="">全部记录</option>{["system", "user", "assistant", "tool", "event"].map(value => <option key={value}>{value}</option>)}
    </select></label><label>搜索内容<input type="search" value={search} onChange={e => { setSearch(e.target.value); setPage(0); }} placeholder="消息、call ID 或 request ID" /></label></div>
    {rows.length === 0 && <Empty>暂无匹配的会话记录。运行中的记录随 checkpoint 更新。</Empty>}
    <div className="conversation">{rows.slice(pageIndex * 30, (pageIndex + 1) * 30).map(entry => <article key={entry.id} className={`message role-${entry.role}`}>
      <div className="message-heading"><strong>{entry.role === "event" ? String(entry.metadata.type ?? "event") : entry.role}</strong><time>{date(entry.created_at)}</time></div>
      {entry.role === "event" ? <Json value={entry.metadata} label="审计记录 / 模型请求" /> : <>
        {entry.tool_call_id && <div className="call-link">{calls.get(entry.tool_call_id)?.function.name ?? entry.name ?? "工具返回"} · <code>{entry.tool_call_id}</code></div>}
        {entry.content !== null && <pre className="message-content">{pretty(entry.content)}</pre>}
        {entry.reasoning != null && <Json value={entry.reasoning} label="推理内容" />}
        {entry.tool_calls.map(call => <div className="call" key={call.id}><strong>{call.function.name}</strong><code>{call.id}</code>
          <Json value={call.function.arguments} label="原始调用参数" /></div>)}
        {Object.keys(entry.metadata).length > 0 && <Json value={entry.metadata} label="执行元数据 / 绑定参数" />}
        {Object.keys(entry.provider_fields).length > 0 && <Json value={entry.provider_fields} label="Provider 原始字段" />}
      </>}
    </article>)}</div>
    <Pager page={pageIndex} total={rows.length} size={30} onChange={setPage} />
  </>;
}
