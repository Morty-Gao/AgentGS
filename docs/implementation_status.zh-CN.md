# 初步实现状态

更新：2026-10-08。

## 算法意图对齐阶段（后续更新）

按用户“核心算法完全不变”的要求，以旧主项目实际工作树冻结48个关键文件，算法源码仍使用这48个文件。用户已指定迁移范围收窄为GeoEval catalog中的101工具、关联24 Skill与31 Capability，形成20项范围内的全链路契约。交付副本及vector/hybrid分支仍独立记录，未混入源码基线。

已实现并用原源码决策验收：24 Skill声明、加权词法检索、选择阈值/工具提示优先、完整Skill计划、原约束诊断、31节点能力图、101项工具声明映射、能力扩展、候选打分、grounding重复实例、候选上下文完整性及工具结果分类。组件验收与runtime可执行能力分开；现有runtime默认领域工具为两项USGS工具与GeoQuestions快照查询、两个初步Skill；ExecutePython仅用于显式内部沙箱请求，生产AgentEngine仍为function-calling原型。

src/tests/fixtures/alignment/baseline.json由旧数据/纯算法源码直接生成，包含源树身份、完整声明与独立期望，测试不需要旧Windows目录。报告的325项组件决策比对不包含16项只冻结、未迁移的答案校验用例，也不代表正式benchmark成绩。审计脚本显式给出component_parity与runtime_equivalent两个结论，严格runtime检查在尚未全链路对齐时返回非零。

桥接、KG检索/embedding、领域规则与绑定、科学后端、执行修复/成功前缀恢复、分阶段模型prompt、最终回答/非修改性验证及冻结FAS评分继续以pending/partial记录。禁止通过非空模型回答、descriptor文件或基础测试通过替代算法等价验收。

下面的基础框架记录已按最新范围更新，不能用“已实现agent循环”推断旧算法已经完整迁移。

已实现config/client、共享契约、Skill接口、agent循环、task runtime、CLI、JSONL评测、FastAPI/SSE和React控制台。领域工具已接入USGS GetEvent、部分USGS Query及GeoQuestions快照协议；完整101工具/24 Skill清单见migration_inventory.json。新增地理计算Skill已移除，地震目录Skill为原能力的初步实现，不代表24个Skill均可执行。

已实际安装项目local Node、bwrap与PRoot；bwrap执行与秘密/路径隔离测试通过；PRoot明确兼容模式手动执行通过。任务RSS限制是watchdog，不是kernel硬上限。Windows仅提供WSL桥接路径，未完成原生沙箱、池化、崩溃恢复和重型科学后端。

下载脚本与现有原始数据保留；新的prepare_usgs、registry索引导出、API类型生成、frontend构建及评测入口可执行。旧派生数据pipeline、向量/KG检索、完整评测评分、streaming client、embedding client与其余领域工具留待后续迁移。

验证覆盖真实task worker、排队/取消/超时、RSS、代码隔离、typed工具参数、产物、HTTP/SSE、配置保密、mock模型工具调用和gold隔离。没有付费LLM调用、旧全量评测或浏览器视觉验收。

## GeoQuestions恢复快照协议（2026-10-08）

按用户新指令，GeoBigDataExecuteGeoSPARQL恢复历史source_id/查询结构匹配协议，并接入新AgentGS默认工具注册表、执行器、Skill与前端。无需空间RDF服务。数据覆盖原40项选题，由本地官方文件可重复生成，SHA核验后结果逐条与旧40项子集一致。

保留旧匹配器评分、阈值、唯一性门槛和同分顺序；source_id优先，ID未命中再查询。typed入口仅接受source_id/query（别名sparql）/dataset，模糊或无匹配失败。结果明确mode=pinned_yago2geo_snapshot、source_type=derived_lookup、answer_lookup=true，保留来源SHA与表格精度。此项为用户授权协议例外，其余工具算法仍沿原基线。

- 125项回放oracle由hash核验的旧源码函数直接执行生成，覆盖40项ID、40项完整查询、40项变量改名查询及5项优先级/别名/失败输入。
- 本轮工具与Agent相关测试136 passed，包含数据缺失/篡改、参数校验、真实worker产物与mock模型工具循环。
- 40项ID加40项查询，共80项真实worker任务成功，工具链匹配；新CLI source_id=591实际成功。
- 规划组件325项差分仍通过；Skill描述按协议更新后用原planner重新冻结期望，旧验收快照保存在outputs/alignment。
- TypeScript/Vite与wheel/sdist构建通过，wheel已核验包含回放代码、数据profile及Skill prompt；Ruff通过。
- 全套测试在本轮受执行沙箱限制：bwrap功能探测不可用，既有取消/关闭及API流程检查出现挂起并中止；不将这些项报告为通过。
- 没有付费模型实验，80项只是快照协议运行验证，不代表实时空间计算或正式benchmark成绩。整体runtime_equivalent仍为false。

## 范围裁剪验证（2026-10-08）

- 默认领域注册表、模型工具schema、CLI与前端菜单已去除额外9项工具；原地理计算Skill已移除。
- 自动化测试143 passed（含9项排除工具的实际拒绝验证）；Ruff通过。
- 101工具、24 Skill、31 Capability声明与迁移清单一致；迁移状态不因裁剪而冒称完成。
- 以同一hash核验的旧源码和用户指定声明子集重新生成oracle，325项组件决策比对通过。原110范围oracle保存在outputs/alignment作本机历史记录。
- 严格runtime审计仍返回1，runtime_equivalent=false；全链路尚未对齐。
- React TypeScript检查、Vite构建、wheel/sdist构建通过；wheel确认包含新范围JSON且不含已移除Skill。
- 离线USGS suite两项任务均成功，领域工具链匹配；没有付费LLM或正式benchmark实验。

## 上一轮基础框架验证记录（2026-10-07）

- uv sync --locked --extra visualization：通过。
- 核心测试：19 passed；有1条第三方TestClient弃用提示。
- Ruff：通过。
- TypeScript检查与Vite生产构建：通过。
- wheel/sdist构建：通过；wheel有prompt与静态资源，没有node_modules或测试包。
- 离线suite：2/2 task成功（不是旧benchmark分数）。
- 实际冻结USGS目录查询：2010-2020、M>=8返回11条。
- bwrap：功能探测及隔离执行通过；PRoot：明确兼容模式手动执行通过。
- 已构建UI根页面与OpenAPI：HTTP 200；未进行浏览器视觉验收。

目标工具状态为2项migrated、1项partial、98项pending；USGS Query部分旧聚合接口仍待迁移。24项目标Skill中地震目录和GeoQuestions为partial，其余22项pending。额外9项工具、4项Skill与7项独有Capability不计入待迁移数。
