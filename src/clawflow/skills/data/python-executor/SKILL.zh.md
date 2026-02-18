---
name: python-executor-mcp
description: |
  通过 MCP 的 python_execute 运行可复现 Python：用于计算/数据处理/生成 CSV/PNG/PDF 等产物，要求非交互、可控超时。
Triggers: 运行python, 执行脚本, 数据分析, pandas, numpy, matplotlib, 报告, csv, json, 画图, 生成文件, 自动化
allowed-tools: python_execute
compatibility: |
  需要可用的 MCP Python 执行工具；如工具名/参数不同请同步修改。
---

# Python 执行器（精炼版）

## 核心规则
- 仅非交互（禁用 input()）。
- stdout 只打印关键摘要，避免全量数据。
- 产物统一写到 `outputs/`，并打印路径。
- 用 `exit_code`/`stderr` 判定成功；失败只做最小修改后重跑。
- 禁止读取 secret/凭证；禁止破坏性操作。

## 约定工具契约（假设）
入参：`code`, `timeout_s`, `working_dir`, `capture_output`  
出参：`stdout`, `stderr`, `exit_code`, `artifacts?`, `error?`

## 示例（基础校验）
```text
python_execute({
  "code": "import sys; print(sys.version)\nprint('ok')\n",
  "timeout_s": 30,
  "capture_output": true,
  "working_dir": null
})
