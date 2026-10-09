import { useState } from "react";
import type { ConsoleConfig, SkillSpec, ToolSpec } from "./contracts";
import { usePolling, useSelection } from "./hooks";
import { Composer } from "./components/Composer";
import { TaskDetail } from "./components/TaskDetail";
import { TaskList } from "./components/TaskList";

export function App() {
  const [selected, select] = useSelection();
  const [revision, setRevision] = useState(0);
  const refresh = () => setRevision(value => value + 1);
  const tools = usePolling<ToolSpec[]>("/tools", 0, revision);
  const skills = usePolling<SkillSpec[]>("/skills", 0, revision);
  const config = usePolling<ConsoleConfig>("/config", 0, revision);
  const error = tools.error || skills.error || config.error;
  return <div className="console">
    <a className="skip-link" href="#main-content">跳转到主要内容</a>
    <TaskList selected={selected} onSelect={select} onNew={() => select("")} revision={revision} />
    <main id="main-content"><header className="page-header"><div><p className="eyebrow">GEOSCIENCE / EXECUTION WORKSPACE</p>
      <h1>{selected ? "执行与证据" : "从问题到结果"}</h1><p>浏览任务进度，追踪模型决策与工具输出。</p></div>
      <button onClick={refresh}>刷新数据</button></header>
      {error && <p className="error" role="alert">配置或工具目录加载失败：{error}<button onClick={refresh}>重试</button></p>}
      <div hidden={!!selected}><Composer tools={tools.data ?? []} skills={skills.data ?? []} config={config.data}
        onSubmitted={task => { select(task.spec.id); refresh(); }} /></div>
      {selected && <TaskDetail key={selected} id={selected} revision={revision} onChange={refresh} />}
    </main>
  </div>;
}
