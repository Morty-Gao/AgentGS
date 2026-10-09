# AgentGS 可视化工作台

工作台直接读取当前 TaskInput、TaskRecord、Conversation 和 ToolResult 契约。FastAPI 管理任务服务生命周期，React 展示任务与执行证据。

## 启动与构建

在项目根目录准备 Python visualization 依赖，再生成前端类型和构建静态资源：

```sh
uv sync --locked --extra visualization
uv run python scripts/build/build_api_types.py
cd src/agentgs/visualization/frontend
npm ci
npm run build
```

回到项目根目录启动：

```sh
uv run agentgs ui serve --project . --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000`。开发时在 frontend 目录运行 `npm run dev`，Vite 将 `/api` 代理到 `127.0.0.1:8000`。前端未构建时，根路径返回带构建说明的 503，API 和 `/docs` 仍可使用。修改 API 路由后须重启服务。

PJLab 环境的构建、执行验证及任务运行应在当前有效 worker 上进行；联网安装前按环境要求设置代理。服务会执行任务中的工具或命令，应沿用 CLI 的可信本地访问方式。

## 功能

- 任务列表：分页、状态筛选、按问题／工具／ID 搜索，URL hash 保存选中任务，支持浏览器前进后退。
- 提交任务：问题、显式工具链、完整 TaskInput JSON。问题模式可指定 `tool_only`、`skill_guided` 或 `data_only`，模型、推理强度、调用预算及 Skill。补充 JSON 可设置资源、输出契约和恢复预算；可视表单中已填写的同名字段优先。
- 工具链：从工具目录查看真实输入 Schema，添加带默认参数的调用，编辑 `arguments`、`id`、`depends_on`，可开启 `parallel_tools`。
- 详情：分别查看最终回答、完成状态、工具结果、完整会话、模型请求与用量、运行事件、产物。任务成功状态不代表评测得分，也不推断未记录的 token 用量。
- 会话：保留 `Message | RuntimeEvent`，展示原始调用参数、reasoning、绑定参数及 provider 字段；以 `tool_call_id` 对应调用名称。
- 产物：按注册 ID 下载，展示来源、大小、SHA256。PNG/JPEG/WebP/GIF 预览限制为 16 MiB；JSON/GeoJSON/CSV/纯文本/Markdown 限制为 256 KiB。文本以纯文本展示，HTML/SVG 等格式仅下载。
- 导出：下载完整 TaskRecord JSON，包含输入、结果、会话与产物引用。该导出不打包产物文件或独立的运行事件流。

运行中的详情每 2 秒读取最近的原子 `result.json` checkpoint，尚未保存的内容不会显示。列表每 4 秒刷新摘要，详情进入终态后停止轮询。SSE 自动断线重连，按序号去重；收到 `end` 后关闭。浏览器最多保留最近 1,000 条运行事件，较早记录仍在服务端。会话、工具结果、模型请求和产物分页展示。

## 模块边界

- `backend/app.py`：应用生命周期与静态资源。
- `backend/routes.py`：HTTP、SSE、产物访问。
- `backend/queries.py`：SQLite 只读分页投影与 checkpoint 读取，不修改 runtime 持久化格式。
- `backend/models.py`：列表摘要和控制台配置响应。
- `frontend/src/contracts.ts`：由 `scripts/build/build_api_types.py` 生成；请求使用 validation schema，响应使用 serialization schema，不手改。
- `frontend/src/api.ts`、`hooks.ts`：HTTP 错误、串行轮询、请求取消、事件订阅、URL 选择状态。
- `frontend/src/components/`：任务列表、提交表单、详情及各证据视图。

## API

| 路径 | 返回／作用 |
| --- | --- |
| `GET /api/config` | 项目规划策略与当前 RuntimeSettings，不读取模型凭据 |
| `GET /api/tasks?limit=30&offset=0&status=failed&search=...` | `TaskPage`，包含 `items`、筛选后 `total`、分页参数及全局状态 `counts`；limit 为 1–100 |
| `POST /api/tasks` | 提交 `TaskInput`，返回 202 + `TaskRecord`；参数错误 422，队列满 429 |
| `GET /api/tasks/{id}` | `TaskRecord`，运行中附带最近 checkpoint |
| `POST /api/tasks/{id}/cancel` | 交给 TaskService 取消，保留已有执行记录 |
| `GET /api/tasks/{id}/messages` | 完整 `LogEntry[]`，包含 `role=event` |
| `GET /api/tasks/{id}/conversation` | `Conversation`（schema_version + messages） |
| `GET /api/tasks/{id}/export` | TaskRecord JSON 下载 |
| `GET /api/tasks/{id}/events?after=0` | SSE，支持 Last-Event-ID；命名事件 `end` 表示任务已终止 |
| `GET /api/tasks/{id}/artifacts/{artifact_id}` | 已注册且位于当前任务产物目录内的文件下载 |
| `GET /api/tasks/{id}/artifacts/{artifact_id}/preview` | 有大小限制的栅格图片或纯文本预览；过大 413，不支持的格式 415 |
| `GET /api/tools`、`GET /api/skills` | 当前工具／Skill 注册表 |

`GET /api/tasks` 返回分页摘要对象，不再返回完整 TaskRecord 数组；详情消费者使用单任务接口。消息消费者必须处理 `role=event`，需要发送给模型时使用核心 Conversation 的转换方法。

## 验证

```sh
uv run pytest src/tests/visualization src/tests/messages src/tests/contracts src/tests/runtime/test_service.py -q
uv run ruff check src/agentgs/visualization scripts/build/build_api_types.py src/tests/visualization
cd src/agentgs/visualization/frontend
npm run build
```

测试覆盖分页筛选、完整会话与运行中 checkpoint、任务参数、SSE 游标、文件访问和有界预览；已有 API 测试通过真实 TaskService/worker 执行离线地震目录工具，不调用外部模型。
