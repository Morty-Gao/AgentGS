import type { TaskRecord, Event as RuntimeEvent, ToolSpec } from "./contracts";
export type Task = TaskRecord;
export type Event = RuntimeEvent;
export type Tool = ToolSpec;
export async function request<T>(path:string,init?:RequestInit):Promise<T>{
  const response=await fetch('/api'+path,init);
  if(!response.ok){throw new Error(`请求失败 ${response.status}: ${await response.text()}`);}
  return response.json();
}
