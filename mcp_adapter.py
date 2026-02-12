import asyncio
import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Type
from contextlib import asynccontextmanager, AsyncExitStack

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, create_model, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool

logger = logging.getLogger(__name__)

# Cache for active sessions: {script_name: (exit_stack, session)}
_ACTIVE_SESSIONS = {}

class MCPClientManager:
    """
    Manages connections to local MCP servers (subprocesses).
    """
    
    def __init__(self):
        self._servers: Dict[str, ClientSession] = {}
        self._stacks: Dict[str, AsyncExitStack] = {}
        self._loops: Dict[str, asyncio.AbstractEventLoop] = {} # Track loop for each session

    async def get_session(self, script_name: str) -> ClientSession:
        """
        Get or create a session for the given MCP server script.
        Ensures the session belongs to the current running event loop.
        """
        current_loop = asyncio.get_running_loop()
        
        if script_name in self._servers:
            # Check if session belongs to current loop
            created_loop = self._loops.get(script_name)
            if created_loop is current_loop and not created_loop.is_closed():
                return self._servers[script_name]
            else:
                logger.info(f"Session for {script_name} belongs to a different/closed loop. Recreating.")
                # We cannot safely close the old stack if its loop is closed, just drop references
                self._servers.pop(script_name, None)
                self._stacks.pop(script_name, None)
                self._loops.pop(script_name, None)

        logger.info(f"Starting MCP server: {script_name}")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[script_name],
            env=os.environ.copy() # type: ignore[arg-type]
        )

        # Use AsyncExitStack to manage lifecycles safely
        stack = AsyncExitStack()
        # We store the stack so we can close it later
        
        # Enter transport context
        # stdio_client returns an async context manager, we enter it via stack
        read, write = await stack.enter_async_context(stdio_client(server_params))
        
        # Enter session context
        session = await stack.enter_async_context(ClientSession(read, write))
        
        await session.initialize()
        
        self._servers[script_name] = session
        self._stacks[script_name] = stack
        self._loops[script_name] = current_loop
        
        return session

    async def close_all(self):
        """Close all active sessions for the current loop."""
        current_loop = asyncio.get_running_loop()
        to_remove = []
        
        for name, stack in self._stacks.items():
            if self._loops.get(name) is current_loop:
                logger.info(f"Closing MCP server: {name}")
                try:
                    await stack.aclose()
                except Exception as e:
                    logger.error(f"Error closing stack {name}: {e}")
                to_remove.append(name)
        
        for name in to_remove:
            self._servers.pop(name, None)
            self._stacks.pop(name, None)
            self._loops.pop(name, None)

    async def get_tools(self, script_name: str, temporary: bool = True) -> List[StructuredTool]:
        """
        Connect to server, list tools, and convert them to LangChain tools.
        
        :param temporary: If True, uses a temporary session that closes immediately after listing tools.
                         This is recommended for initialization to prevent zombie processes.
                         The returned tools will still be able to connect via get_session() when invoked.
        """
        if temporary:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[script_name],
                env=None 
            )
            # Use manual context management for clean temporary session
            logger.info(f"Temporarily connecting to MCP server: {script_name}")
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_list = await session.list_tools()
                    
                    lc_tools = []
                    for tool_info in mcp_list.tools:
                        lc_tool = self._convert_tool(script_name, tool_info)
                        lc_tools.append(lc_tool)
                    return lc_tools
        else:
            # Use persistent session
            session = await self.get_session(script_name)
            mcp_list = await session.list_tools()
            
            lc_tools = []
            for tool_info in mcp_list.tools:
                lc_tool = self._convert_tool(script_name, tool_info)
                lc_tools.append(lc_tool)
                
            return lc_tools

    def _convert_tool(self, script_name: str, tool_info: McpTool) -> StructuredTool:
        """
        Convert an MCP Tool definition to a LangChain StructuredTool.
        """
        name = tool_info.name
        description = tool_info.description or ""
        input_schema = tool_info.inputSchema or {}

        # Dynamically create Pydantic model for args_schema
        # JSON Schema mapping to Pydantic types is non-trivial for complex types,
        # but FastMCP typically produces simple flat schemas.
        # We'll use a simplified converter here.
        args_model = self._create_pydantic_model(name, input_schema)

        async def _tool_coroutine(**kwargs) -> Any:
            try:
                # Dynamically get session for current loop
                session = await CLIENT_MANAGER.get_session(script_name)
                
                # Call MCP tool
                result = await session.call_tool(name, arguments=kwargs)
                
                # Check for error
                if result.isError:
                    raise ToolException(f"MCP Tool {name} failed: {result}")
                
                # FastMCP usually returns a list of contents. We want the text result.
                # Assuming simple text output for now.
                output_texts = []
                for content in result.content:
                    if content.type == "text":
                        output_texts.append(content.text)
                    elif content.type == "image":
                        output_texts.append(f"[Image: {content.mimeType}]")
                    elif content.type == "resource":
                        output_texts.append(f"[Resource: {content.uri}]")
                
                # Try to extract structured result if available (FastMCP custom convention)
                # But standardized MCP returns content list.
                # Let's join text content.
                final_output = "\n".join(output_texts)
                
                # Some FastMCP tools might return JSON string in text, 
                # or we might want to return the raw list if it's complex.
                # For compatibility with LLM agent which expects string:
                return final_output

            except Exception as e:
                raise ToolException(f"Error calling {name}: {str(e)}")

        return StructuredTool.from_function(
            func=None, # We only provide async implementation
            coroutine=_tool_coroutine,
            name=name,
            description=description,
            args_schema=args_model
        )

    def _create_pydantic_model(self, name: str, json_schema: Dict[str, Any]) -> Type[BaseModel]:
        """
        Create a Pydantic model from a JSON Schema (simplified).
        """
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])
        
        fields = {}
        for field_name, field_def in properties.items():
            field_type = str
            t = field_def.get("type")
            if t == "integer":
                field_type = int
            elif t == "boolean":
                field_type = bool
            elif t == "number":
                field_type = float
            elif t == "array":
                field_type = list
            elif t == "object":
                field_type = dict
            
            # Default value logic
            if field_name in required:
                default = ... # Ellipsis means required
            else:
                default = field_def.get("default", None)
            
            fields[field_name] = (field_type, Field(default=default, description=field_def.get("description", "")))
        
        return create_model(f"{name}Input", **fields)

# Global singleton
CLIENT_MANAGER = MCPClientManager()

def update_global_registry(tools: List[StructuredTool]):
    """
    Update the global TOOL_REGISTRY in tools.py with the given tools.
    """
    from tools import TOOL_REGISTRY
    
    for t in tools:
        TOOL_REGISTRY[t.name] = t
        # Also add sanitized names if needed, similar to build_tool_registry
        TOOL_REGISTRY[t.name.strip()] = t
        TOOL_REGISTRY[t.name.strip().lower()] = t

