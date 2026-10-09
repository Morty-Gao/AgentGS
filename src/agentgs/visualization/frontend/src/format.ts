import type { TaskStatus, TaskSummary } from "./contracts";

export const statuses: Record<TaskStatus, string> = {
  queued: "排队中", preparing: "准备中", running: "运行中",
  succeeded: "已成功", failed: "已失败", cancelled: "已取消",
};
export const terminal = (status: TaskStatus) => ["succeeded", "failed", "cancelled"].includes(status);
export const taskTitle = (task: TaskSummary) => task.question || task.tools.join(" → ") || task.id;
export const pretty = (value: unknown) => typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "—";
export function bytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1048576) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / 1048576).toFixed(1)} MiB`;
}
export const date = (value: string) => new Date(value).toLocaleString("zh-CN", { hour12: false });
