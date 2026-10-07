# 初步实现状态

2026-10-07。

已实现config/client、共享契约、Skill接口、agent循环、task runtime、CLI、JSONL评测、FastAPI/SSE和React控制台。实际8个旧工具迁入、新增ExecutePython；完整110工具/28Skill清单见migration_inventory.json。地理计算Skill是新增聚合能力，地震目录Skill为原能力的初步实现，不代表全部28个旧Skill都已迁入。

已实际安装项目local Node、bwrap与PRoot；bwrap执行与秘密/路径隔离测试通过；PRoot明确兼容模式手动执行通过。任务RSS限制是watchdog，不是kernel硬上限。Windows仅提供WSL桥接路径，未完成原生沙箱、池化、崩溃恢复和重型科学后端。

下载脚本与现有原始数据保留；新的prepare_usgs、registry索引导出、API类型生成、frontend构建及评测入口可执行。旧派生数据pipeline、向量/KG检索、完整评测评分、streaming client、embedding client与其余领域工具留待后续迁移。

验证覆盖真实task worker、排队/取消/超时、RSS、代码隔离、typed工具参数、产物、HTTP/SSE、配置保密、mock模型工具调用和gold隔离。没有付费LLM调用、旧全量评测或浏览器视觉验收。

## 本轮验证记录

- uv sync --locked --extra visualization：通过。
- 核心测试：19 passed；有1条第三方TestClient弃用提示。
- Ruff：通过。
- TypeScript检查与Vite生产构建：通过。
- wheel/sdist构建：通过；wheel有prompt与静态资源，没有node_modules或测试包。
- 离线suite：2/2 task成功（不是旧benchmark分数）。
- 实际冻结USGS目录查询：2010-2020、M>=8返回11条。
- bwrap：功能探测及隔离执行通过；PRoot：明确兼容模式手动执行通过。
- 已构建UI根页面与OpenAPI：HTTP 200；未进行浏览器视觉验收。

完整能力状态为7项migrated、1项partial、102项pending；USGS Query部分旧聚合接口仍待迁移。旧Skill清单中地震目录标为partial，新增geoscience Skill不冒充旧28项全部完成。
