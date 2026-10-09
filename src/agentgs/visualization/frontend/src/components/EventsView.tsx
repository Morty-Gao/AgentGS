import { useMemo, useState } from "react";
import type { Event } from "../contracts";
import { date } from "../format";
import { Empty, Json, Pager } from "./Common";

export function EventsView({ events, connection, dropped }: { events: Event[]; connection: string; dropped: number }) {
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  const [descending, setDescending] = useState(true);
  const rows = useMemo(() => {
    const selected = events.filter(e => !filter || e.type.includes(filter));
    return descending ? selected.reverse() : selected;
  }, [events, filter, descending]);
  const pageIndex = Math.min(page, Math.max(0, Math.ceil(rows.length / 50) - 1));
  return <><div className="toolbar"><label>事件类型<input type="search" value={filter} placeholder="例如 model. / tool. / task."
    onChange={e => { setFilter(e.target.value); setPage(0); }} /></label>
    <button aria-pressed={descending} onClick={() => { setDescending(v => !v); setPage(0); }}>{descending ? "最新在前" : "最早在前"}</button>
    <span className="muted" role="status">{connection}</span></div>
    {dropped > 0 && <p className="notice">当前保留最近 1,000 条事件，较早的 {dropped} 条仍保存在服务端。</p>}
    {rows.length === 0 && <Empty>暂无匹配事件。</Empty>}
    <ol className="events">{rows.slice(pageIndex * 50, (pageIndex + 1) * 50).map(event => <li key={event.sequence}>
      <span className="sequence">{event.sequence}</span><div><div className="message-heading"><strong>{event.type}</strong><time>{date(event.time)}</time></div>
        <Json value={event.payload} label="事件详情" /></div>
    </li>)}</ol><Pager page={pageIndex} total={rows.length} size={50} onChange={setPage} /></>;
}
