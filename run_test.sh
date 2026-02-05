#!/usr/bin/env bash
set -euo pipefail

# 测试脚本：验证系统在遇到未来日期请求时，是否使用历史代理模式

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置未来日期模拟（2026年1月29日）
export RUNTIME_UTC_OVERRIDE="2026-01-29T00:00:00Z"

echo "=== 开始测试：未来日期数据请求处理 ==="
echo "模拟系统时间: $RUNTIME_UTC_OVERRIDE"
echo ""

# 用户请求：2026年市场数据
QUESTION="请提供2026年1月29日标普500指数、纳斯达克综合指数和道琼斯工业平均指数的开盘价、收盘价、最高价、最低价以及交易量数据。"

echo "用户问题: $QUESTION"
echo ""

# 运行主程序，捕获输出
LOG_FILE="test_future_date.log"
echo "运行主程序，日志输出到: $LOG_FILE"
echo ""

python3 main.py --question "$QUESTION" 2>&1 | tee "$LOG_FILE"

echo ""
echo "=== 测试完成 ==="
echo "请检查日志文件 $LOG_FILE，确认以下行为："
echo "1. 系统是否识别当前为2026年（通过日志中的 'Current UTC Time' 确认）"
echo "2. 研究员节点是否报告 '未来日期数据不可用' 或类似信息"
echo "3. 研究员节点是否切换到 '历史代理模式' 或 '模拟模式'，例如使用2025年数据作为基准"
echo "4. 反射节点是否接受该处理方式，而不是要求更多真实数据"
echo "5. 最终答案是否明确标注使用了代理数据或模拟数据"
echo ""
echo "如果上述行为符合预期，则时间悖论修复成功。"