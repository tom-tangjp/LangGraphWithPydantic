import asyncio
import logging
import os
import uuid

from agent import AgentState
from utils import truncate
from llm import build_llm, build_reflection_multi_agent_graph
from skills_registry import WORKSPACE_ROOT, SKILLS_DIR


def setup_logging(level=logging.INFO):
    # 【关键】在这里统一配置
    logging.basicConfig(
        level=level,
        format='time="%(asctime)s" level=%(levelname)s event=%(message)s',
        handlers=[
            logging.FileHandler("app.log"),  # 输出到文件
            logging.StreamHandler(),  # 输出到控制台
        ],
    )


async def run(graph, request: str):
    thread_id = f"run-{uuid.uuid4()}"
    logger.info(f"🔥 [New Session] Thread ID: {thread_id}")

    init_state: AgentState = {
        "user_request": request,
        "messages": [],
        "artifacts": {},
        "reflections": [],
        "seen_step_hashes": [],
        "step_idx": 0,
        "iter_count": 0,
        "max_iters": 20,
        "final_answer": "",
        "done": False,
        "step_failures": {},
        "last_feedback": {},
        "last_output_hash": {},
        "no_progress": {},
    }

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}

    logger.info(f"start request={truncate(request, 320)}")

    final_answer = ""

    async for upd in graph.astream(init_state, config=config, stream_mode="updates"):
        node, patch = next(iter(upd.items()))

        logger.info(f"graph.update node={node} keys={list((patch or {}).keys())[:30]}")

        if node == "respond":
            final_answer = (patch or {}).get("final_answer", "") or ""
            break

    logger.info(f"done final_answer_len={len(final_answer)}")
    logger.info("\nFINAL ANSWER:\n%s", final_answer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reflection Multi-Agent (LangGraph) - executable demo"
    )
    parser.add_argument(
        "--question", "-p", type=str, default=None, help="User request / question"
    )
    parser.add_argument(
        "--workspace", type=str, default=None, help="Override AGENT_WORKSPACE_ROOT"
    )
    parser.add_argument(
        "--skills-dir", type=str, default=None, help="Override AGENT_SKILLS_DIR"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider: qwen|doubao|yuanbao|openai_compat|openai|ollama",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Default model name (role-specific env overrides still apply)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base_url (for qwen/doubao/yuanbao/openai_compat)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (will be set to AGENT_LLM_API_KEY if provided)",
    )
    parser.add_argument(
        "--search-backend",
        type=str,
        default=None,
        help="Search backend: searxng|tavily|serpapi|bing|none",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Run benchmark autorun: tidb|cockroachdb|vitess|yba|all",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
    )
    args = parser.parse_args()

    if args.workspace:
        os.environ["AGENT_WORKSPACE_ROOT"] = args.workspace
    if args.skills_dir:
        os.environ["AGENT_SKILLS_DIR"] = args.skills_dir
    if args.provider:
        os.environ["AGENT_LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["AGENT_LLM_MODEL"] = args.model
    if args.base_url:
        os.environ["AGENT_LLM_BASE_URL"] = args.base_url
        os.environ.setdefault("OPENAI_BASE_URL", args.base_url)
    if args.api_key:
        os.environ["AGENT_LLM_API_KEY"] = args.api_key
        os.environ.setdefault("OPENAI_API_KEY", args.api_key)
    if args.search_backend:
        os.environ["AGENT_SEARCH_BACKEND"] = args.search_backend

    log_level = args.log_level.upper()
    setup_logging(getattr(logging, log_level, logging.INFO))

    logger = logging.getLogger(__name__)

    logger.info(f"[boot] WORKSPACE_ROOT={WORKSPACE_ROOT} SKILLS_DIR={SKILLS_DIR}")

    question = args.question

    if not question:
        try:
            question = input("Enter question: ").strip()
        except EOFError:
            question = ""
    if not question:
        raise SystemExit("No question provided. Use --question or stdin.")

    async def main_entry():
        # Initialize tools via MCP within the same event loop
        from llm import init_mcp_tools
        logger.info("Initializing MCP tools...")
        tool_map = await init_mcp_tools()

        intent_llm, planner_llm, agent_llm, reflector_llm, responder_llm = build_llm()
        graph = build_reflection_multi_agent_graph(
            intent_llm, planner_llm, agent_llm, reflector_llm, responder_llm=responder_llm,
            tool_map=tool_map
        )
        await run(graph, question)

    asyncio.run(main_entry())
