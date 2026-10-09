import { useState } from "react";
import type { ReactNode } from "react";
import type { TaskStatus } from "../contracts";
import { pretty, statuses } from "../format";

export function Status({ value }: { value: TaskStatus }) {
  return <span className={`badge ${value}`}><span aria-hidden="true" className="status-dot" />{statuses[value]}</span>;
}
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function Json({ value, label = "查看数据" }: { value: unknown; label?: string }) {
  return <details className="json"><summary>{label}</summary><pre>{pretty(value)}</pre></details>;
}
export function Pager({ page, total, size, onChange }: {
  page: number; total: number; size: number; onChange: (page: number) => void;
}) {
  return <div className="pager"><button disabled={page === 0} onClick={() => onChange(page - 1)}>上一页</button>
    <span>{total ? page * size + 1 : 0}–{Math.min((page + 1) * size, total)} / {total}</span>
    <button disabled={(page + 1) * size >= total} onClick={() => onChange(page + 1)}>下一页</button></div>;
}
export function Copy({ text }: { text: string }) {
  const [label, setLabel] = useState("复制");
  return <button onClick={() => {
    if (!navigator.clipboard) { setLabel("请手动选择复制"); return; }
    void navigator.clipboard.writeText(text).then(() => setLabel("已复制"), () => setLabel("复制失败"));
  }}>{label}</button>;
}
