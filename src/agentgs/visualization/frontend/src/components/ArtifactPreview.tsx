import { useEffect, useState } from "react";
import type { ArtifactRef } from "../contracts";
import { artifactUrl } from "../api";

export function ArtifactPreview({ taskId, artifact }: { taskId: string; artifact: ArtifactRef }) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState<{ image?: string; text?: string; error?: string }>({});
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    let objectUrl: string | undefined;
    let disposed = false;
    setValue({});
    async function load() {
      try {
        const response = await fetch(artifactUrl(taskId, artifact.id) + "/preview", { signal: controller.signal });
        if (!response.ok) {
          const message = response.status === 413 ? "文件超过预览大小限制，请下载查看。"
            : response.status === 415 ? "此格式暂不支持预览，请下载查看。"
              : "预览加载失败，请重试或下载文件。";
          throw new Error(message);
        }
        if (response.headers.get("content-type")?.startsWith("image/")) {
          const blob = await response.blob();
          if (disposed) return;
          objectUrl = URL.createObjectURL(blob);
          setValue({ image: objectUrl });
        } else {
          const text = await response.text();
          if (!disposed) setValue({ text });
        }
      } catch (error) { if (!disposed) setValue({ error: String(error) }); }
    }
    void load();
    return () => { disposed = true; controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [open, taskId, artifact.id]);
  return <div className="artifact-preview"><button aria-expanded={open} onClick={() => setOpen(v => !v)}>{open ? "收起预览" : "预览文件"}</button>
    {open && <div>{value.error ? <p className="error" role="alert">{value.error}</p>
      : value.image ? <img src={value.image} alt={artifact.path} />
        : value.text !== undefined ? <pre>{value.text}</pre> : <p className="muted" role="status">正在加载预览…</p>}</div>}
  </div>;
}
