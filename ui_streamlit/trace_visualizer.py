from typing import List, Dict, Any

def generate_mermaid_sequence(traces: List[Dict[str, Any]]) -> str:
    """
    Convert execution traces into a Mermaid sequence diagram.
    """
    if not traces:
        return "sequenceDiagram\n    Note over User: No traces found"

    lines = ["sequenceDiagram", "    autonumber", "    actor User"]
    
    # Define display aliases and ordering (for aesthetics)
    # Format: internal node name -> display name
    node_map = {
        "intent_node": "Intent",
        "plan_node": "Planner",
        "initial_plan": "Init_Planner", # Backward compatible name
        "dynamic_plan": "Dynamic_Planner", # Backward compatible name
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

    # Dynamically discover participants and declare them in the diagram
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

        # Try to extract a meaningful summary for the arrow label
        inputs = trace.get("input", {})
        outputs = trace.get("output", {})
        result = outputs.get("result")
        
        note = ""
        action = ""
        
        # Customize display by node type
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
                user_req = inputs.get("user_request", {}) # Input structure may vary; be defensive
                if isinstance(user_req, dict):
                    content = user_req.get("content", "")
                    if content:
                        note = f"Note right of {current_participant}: Input: {content[:30]}..."
            
            # reflect result is usually a "next to X" string or a dict
            action = str(result)
            if "next to" in action:
                action = action.replace("next to ", "Goto ")

        elif "node" in raw_node_name and ("researcher" in raw_node_name or "solver" in raw_node_name):
             # Agent node
             if isinstance(result, dict):
                 # result may be an AIMessage object or a dict
                 content = result.get("content", "") or ""
                 if not content and hasattr(result, "content"):
                     content = result.content
                 
                 action = "Execute"
                 if "tool_calls" in str(result):
                     action = "Call Tools"
        else:
            action = "Process"

        # Draw arrows
        lines.append(f"    {last_participant}->>{current_participant}: {action}")
        
        # Draw Note (if any)
        if note:
            lines.append(f"    {note}")
            
        last_participant = current_participant

    return "\n".join(lines)

# --- Demo usage ---
if __name__ == "__main__":
    # Mock some trace data (matches real runtime structure)
    mock_traces = [
        {"node": "intent_node", "input": {"user_request": "查股价"}, "output": {"result": {"task_type": "research"}}},
        {"node": "plan_node", "input": {}, "output": {"result": {"steps": [{"id": "s1"}]}}},
        {"node": "route_node", "input": {}, "output": {"result": 1}},
        {"node": "researcher_node", "input": {}, "output": {"result": "Tool Call..."}},
        {"node": "reflect_node", "input": {}, "output": {"result": "next to reflect"}},
        # Assume retry
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
