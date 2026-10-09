import { useEffect, useState } from "react";
import { request, taskPath } from "./api";
import { terminal } from "./format";
import type { Event, TaskRecord } from "./contracts";

/** Serial polling with cancellation prevents old selections from overwriting new data. */
export function usePolling<T>(path: string | null, interval = 4000, revision = 0,
  done?: (value: T) => boolean) {
  const [state, setState] = useState<{ path: string | null; data?: T; error: string }>({ path, error: "" });
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let stopped = false;
    setState(previous => ({ path, data: previous.path === path ? previous.data : undefined, error: "" }));
    async function poll() {
      let finished = false;
      try {
        const data = await request<T>(path!, { signal: controller.signal });
        if (!stopped) setState({ path, data, error: "" });
        finished = !!done?.(data);
      } catch (error) {
        if (!stopped) setState(previous => ({ ...previous, error: String(error) }));
      }
      if (!stopped && !finished && interval > 0) timer = setTimeout(poll, interval);
    }
    void poll();
    return () => { stopped = true; controller.abort(); clearTimeout(timer); };
  }, [path, interval, revision, done]);
  return state.path === path ? state : { path, data: undefined, error: "" };
}

const taskDone = (task: TaskRecord) => terminal(task.status);
export function useTask(id: string, revision: number) {
  return usePolling<TaskRecord>(id ? taskPath(id) : null, 2000, revision, taskDone);
}

export function useEvents(id: string) {
  const [state, setState] = useState<{ id: string; events: Event[]; connection: string; dropped: number }>({
    id, events: [], connection: "连接中", dropped: 0,
  });
  useEffect(() => {
    if (!id) return;
    let disposed = false;
    const source = new EventSource(`/api${taskPath(id)}/events`);
    setState({ id, events: [], connection: "连接中", dropped: 0 });
    source.onopen = () => { if (!disposed) setState(s => ({ ...s, connection: "实时连接" })); };
    source.onmessage = message => {
      if (disposed) return;
      try {
        const event: Event = JSON.parse(message.data);
        setState(s => {
          if (s.events.some(row => row.sequence === event.sequence)) return s;
          const rows = [...s.events, event];
          return { ...s, events: rows.slice(-1000), dropped: s.dropped + Math.max(0, rows.length - 1000) };
        });
      } catch { setState(s => ({ ...s, connection: "事件解析失败" })); }
    };
    source.addEventListener("end", () => {
      source.close();
      if (!disposed) setState(s => ({ ...s, connection: "已结束" }));
    });
    source.onerror = () => {
      if (!disposed) setState(s => ({ ...s, connection: "连接中断，正在重连" }));
    };
    return () => { disposed = true; source.close(); };
  }, [id]);
  return state.id === id ? state : { id, events: [], connection: "连接中", dropped: 0 };
}

export function useSelection() {
  const read = () => new URLSearchParams(window.location.hash.slice(1)).get("task") ?? "";
  const [id, setId] = useState(read);
  useEffect(() => {
    const update = () => setId(read());
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return [id, (next: string) => {
    window.location.hash = next ? new URLSearchParams({ task: next }).toString() : "";
    setId(next);
  }] as const;
}
