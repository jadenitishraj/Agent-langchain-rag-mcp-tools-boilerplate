import asyncio
import sys
from langchain_core.tools import tool
from .MCPClient import connect

# ------------------------------------------------------------
# MCP connection (1-liner)
# ------------------------------------------------------------

CONNECTION_CMD = f"{sys.executable} -u -m mcp_servers.sqlite_server"

async def _call(tool_name: str, args: dict) -> str:
    """Async helper to call MCP tool with fresh client per request."""
    client = connect(CONNECTION_CMD)
    try:
        # 30 second timeout for safety
        return await asyncio.wait_for(client.call(tool_name, args), timeout=30.0)
    except asyncio.TimeoutError:
        return "❌ MCP Timeout: Server did not respond in 30 seconds."
    except Exception as e:
        return f"❌ MCP Error: {str(e)}"
    finally:
        await client.close()

# ------------------------------------------------------------
# TOOLS (Async)
# ------------------------------------------------------------

@tool
async def mcp_query_memories(sql: str) -> str:
    """
    Query memories database using SQL (SELECT only).
    """
    res = await _call("query_memories", {"sql": sql})
    if hasattr(res, 'content'): return res.content
    return str(res)


@tool
async def mcp_memory_stats() -> str:
    """
    Get overall memory statistics.
    """
    res = await _call("get_memory_stats", {})
    if hasattr(res, 'content'): return res.content
    return str(res)


@tool
async def mcp_search_memories(keyword: str, limit: int = 10) -> str:
    """
    Search memories by keyword.
    """
    res = await _call("search_memories", {
        "keyword": keyword,
        "limit": limit
    })
    if hasattr(res, 'content'): return res.content
    return str(res)


@tool
async def mcp_list_tools() -> str:
    """
    List all tools exposed by the MCP server (debug / verification).
    """
    client = connect(CONNECTION_CMD)
    try:
        tools = await asyncio.wait_for(client.list_tools(), timeout=30.0)
        return "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )
    except Exception as e:
        return f"❌ Error listing tools: {str(e)}"
    finally:
        await client.close()


# ------------------------------------------------------------
# Tool registry (LangChain / agent use)
# ------------------------------------------------------------

SQLITE_MCP_TOOLS = [
    mcp_query_memories,
    mcp_memory_stats,
    mcp_search_memories,
    mcp_list_tools,
]


def get_sqlite_mcp_tools():
    return SQLITE_MCP_TOOLS
