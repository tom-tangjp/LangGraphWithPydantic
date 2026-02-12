---
name: source_code_analysis
description: |
  用“可落地的覆盖率策略”分析项目源码：先识别入口与核心路径，再按目录汇总模块职责、架构与执行流，最后给出具体 bug/风险/优化建议，并明确说明已读/未读范围，避免形式主义全量灌入。
Triggers: 分析源码, 项目架构, 执行流程, 模块划分, 代码走读, 查bug, 性能优化, 安全风险, C++, Python, Java, Go
allowed-tools: list_source_files list_dir read_source_file grep_text walk_dir read_file extract_cpp_functions extract_functions write_text_file
compatibility: 工具名可能不同；优先用最少工具调用，extractor 不可用时要可降级。
---

# 源码分析（精炼注入版）

## 目标
- 找到入口文件与核心业务链路（core flow）。
- 按目录汇总模块职责与关键文件。
- 输出架构与执行流（信息足够时再画 Mermaid）。
- 给出可执行的 bug/风险/优化建议（带文件路径/符号锚点）。

## 硬规则（省 token/省调用）
- **核心路径必读**：入口（main/cli/server）、路由/handler、配置启动、核心接口与关键数据结构必须读。
- **覆盖率目标**：业务核心目录尽量全覆盖；`vendor/third_party/generated/build/dist`、大体量测试数据可排除或抽样。
- **高效读取**：优先批量读取；单文件读取要设置 `max_chars` 上限，必要时二次读取补关键段。
- **诚实输出**：必须说明已读/未读范围与原因，禁止假装读完。

## 最小流程
1) 盘点：`list_source_files(root)` + `walk_dir(root)` 获取规模与结构。
2) 定位入口：`grep_text` 搜 `main/cmd/server/router/init/Run/Start` 等关键词。
3) 阅读与抽取：
   - 入口与核心模块优先读全（大文件截取再补读）。
   - 有 `extract_*` 用它；没有就用代码+grep 推断。
4) 目录汇总：逐目录写“职责 + 关键文件 + 交互关系”。
5) 架构/执行流：信息足够再画图，避免臆测。
6) 问题清单：bug/风险/优化点要具体可执行，标注文件/符号。
7) 输出：能写文件则保存 `project_analysis.md`；否则在对话输出。

## 输出要求（短约定）
- 概览（目标、技术栈、入口点）
- 模块/目录地图（目录→职责）
- 关键文件表（文件→作用→关键符号）
- 架构与执行流（可选 Mermaid）
- 问题与修复建议（可操作、可定位）
- 统计（总文件数、已分析数、排除规则、日期）
