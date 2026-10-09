import type { TaskRecord, TaskSubmission } from "./contracts";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail
        : Array.isArray(body.detail) ? body.detail.map((e: { loc: string[]; msg: string }) =>
          `${e.loc.join(".")}: ${e.msg}`).join("\n") : message;
    } catch { /* A proxy may return a non-JSON error page. */ }
    throw new ApiError(response.status, message);
  }
  return response.json();
}

export function submitTask(input: TaskSubmission) {
  return request<TaskRecord>("/tasks", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input),
  });
}

export const taskPath = (id: string) => `/tasks/${encodeURIComponent(id)}`;
export const artifactUrl = (id: string, artifact: string) =>
  `/api${taskPath(id)}/artifacts/${encodeURIComponent(artifact)}`;
