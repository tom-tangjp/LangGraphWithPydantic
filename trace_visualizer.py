import json
from typing import List, Dict, Any

def generate_mermaid_sequence(traces: List[Dict[str, Any]]) -> str:
    """
    将执行轨迹转换为 Mermaid 时序图
    """
    if not traces:
        return "sequenceDiagram\n    Note over User: No traces found"

    lines = ["sequenceDiagram", "    autonumber", "    actor User"]
    
    # 定义节点显示的别名和顺序（美观起见）
    # 格式: 内部节点名: 显示名
    node_map = {
        "intent_node": "Intent",
        "plan_node": "Planner",
        "initial_plan": "Init_Planner", # 兼容旧名称
        "dynamic_plan": "Dynamic_Planner", # 兼容旧名称
        "plan_reviewer_node": "Reviewer",
        "route_node": "Route",
        "researcher_node": "Researcher",
        "solver_node": "Solver",
        "writer_node": "Writer",
        "code_researcher_node": "CodeResearcher",
        "code_reviewer_node": "CodeReviewer",
        "reflect_node": "Reflect",
        "response_node": "Respond",
        "respond": "Respond"
    }

    # 动态发现所有参与的节点，确保它们在图中声明（可选，为了控制顺序可硬编码）
    seen_nodes = set()
    for t in traces:
        n = t.get("node")
        if n and n not in seen_nodes:
            seen_nodes.add(n)
            display_name = node_map.get(n, n)
            if display_name != "User":
                lines.append(f"    participant {display_name}")

    last_participant = "User"
    
    for trace in traces:
        raw_node_name = trace.get("node")
        current_participant = node_map.get(raw_node_name, raw_node_name)

        # 尝试提取一些有意义的摘要信息显示在箭头上
        inputs = trace.get("input", {})
        outputs = trace.get("output", {})
        result = outputs.get("result")
        
        note = ""
        action = ""
        
        # 根据节点类型定制显示信息
        if "intent" in raw_node_name:
            if isinstance(result, dict):
                action = f"Task: {result.get('task_type')}"
        elif "plan_reviewer" in raw_node_name:
            if isinstance(result, dict):
                decision = result.get("decision")
                action = f"Review: {decision}"
                if decision == "reject":
                    note = f"Note right of {current_participant}: Feedback: {str(result.get('feedback'))[:30]}..."
        elif "reflect" in raw_node_name:
            if isinstance(inputs, dict):
                user_req = inputs.get("user_request", {}) # 这里 input 结构可能不一致，需防御
                if isinstance(user_req, dict):
                    content = user_req.get("content", "")
                    if content:
                        note = f"Note right of {current_participant}: Input: {content[:30]}..."
            
            # reflect 的 result 通常是 "next to X" 字符串，或者 dict
            action = str(result)
            if "next to" in action:
                action = action.replace("next to ", "Goto ")

        elif "node" in raw_node_name and ("researcher" in raw_node_name or "solver" in raw_node_name):
             # Agent 节点
             if isinstance(result, dict):
                 # result 可能是 AIMessage 对象或者 dict
                 content = result.get("content", "") or ""
                 if not content and hasattr(result, "content"):
                     content = result.content
                 
                 action = "Execute"
                 if "tool_calls" in str(result):
                     action = "Call Tools"
        else:
            action = "Process"

        # 绘制箭头
        lines.append(f"    {last_participant}->>{current_participant}: {action}")
        
        # 绘制 Note（如果有）
        if note:
            lines.append(f"    {note}")
            
        last_participant = current_participant

    return "\n".join(lines)

# --- 演示用法 ---
if __name__ == "__main__":
    # 模拟一些 Trace 数据 (与真实运行产生的数据结构一致)
    mock_traces = [
        {"node": "intent_node", "input": {"user_request": "查股价"}, "output": {"result": {"task_type": "research"}}},
        {"node": "plan_node", "input": {}, "output": {"result": {"steps": [{"id": "s1"}]}}},
        {"node": "route_node", "input": {}, "output": {"result": 1}},
        {"node": "researcher_node", "input": {}, "output": {"result": "Tool Call..."}},
        {"node": "reflect_node", "input": {}, "output": {"result": "next to reflect"}},
        # 假设重试
        {"node": "reflect_node", "input": {}, "output": {"result": "next to dynamic_plan (revise)"}},
        {"node": "plan_node", "input": {}, "output": {"result": "New Plan"}},
        {"node": "plan_reviewer_node", "input": {}, "output": {"result": {"decision": "approve"}}},
        {"node": "route_node", "input": {}, "output": {"result": 2}},
        {"node": "response_node", "input": {}, "output": {"result": "Final Answer"}}
    ]

    mermaid_code = generate_mermaid_sequence(mock_traces)
    
    print("=== Generated Mermaid Sequence Diagram ===")
    print(mermaid_code)
    print("==========================================")
    print("可以将上述代码粘贴到 https://mermaid.live/ 进行预览")
