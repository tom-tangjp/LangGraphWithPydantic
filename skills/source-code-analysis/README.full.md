# Source Code Analysis (Full Guide)

本文件为详版流程说明，适合人类阅读与存档，不建议在运行时注入到 LLM prompt。

## 1. 覆盖率策略（推荐）
- Core-path must-read：入口/路由/配置/核心接口/关键数据结构
- Coverage target：核心业务目录尽量全覆盖
- Exclusions：vendor/third_party/generated/build/dist、大型 fixtures
- Sampling：对低价值目录抽样（按文件名模式/修改时间/引用热度）

## 2. 文件读取策略
- 小文件：全读
- 大文件：先读开头/目录/关键符号，再按 grep 定位补读
- 批量读取：对同目录关键文件用 batch_read 提高效率

## 3. 结构化记录（建议）
对关键文件记录：
- file, role, key symbols (classes/functions), dependencies, notes/risks

目录汇总：
- responsibilities, key files, internal interactions, external deps

## 4. 架构与执行流
- 入口识别：main/cli/server/boot
- 路由识别：router/handler/controller
- 数据流：storage/cache/network boundaries
- Mermaid：只在确认模块关系后绘制，避免幻觉命名

## 5. 问题与优化
- Bug：空指针、资源泄漏、竞态、错误处理缺失、边界条件
- 性能：热路径、N+1、锁竞争、I/O 阻塞、缓存策略
- 安全：输入校验、路径穿越、注入、敏感信息日志
- 风格：可维护性、模块耦合、重复代码

## 6. 报告模板（可选）
- Overview / Structure / Key Files / Architecture / Flow / Issues / Stats
- 输出 `project_analysis.md`，并记录已读/未读范围与原因
