# Tools Module
# Custom tools for the AgentForge codebase assistant

from .contact_tool import get_contact_info, contact_tool, ALL_TOOLS as CONTACT_TOOLS
from .memory_tool import save_memory, recall_memories, delete_memory, get_all_memories, MEMORY_TOOLS

# Import DuckDuckGo search tools (FREE)
try:
    from mcp_servers.mcp_client import web_search, web_news, MCP_TOOLS
    print("✅ Web search tools loaded (DuckDuckGo - FREE)")
except ImportError as e:
    print(f"⚠️  Web search tools not available: {e}")
    MCP_TOOLS = []
    web_search = None
    web_news = None

# Import SQLite MCP tools (Real MCP!)
try:
    from mcp_servers.sqlite_client import (
        mcp_query_memories, 
        mcp_memory_stats, 
        mcp_search_memories,
        SQLITE_MCP_TOOLS
    )
    print("✅ SQLite MCP tools loaded (Real MCP Server!)")
except ImportError as e:
    print(f"⚠️  SQLite MCP tools not available: {e}")
    SQLITE_MCP_TOOLS = []
    mcp_query_memories = None
    mcp_memory_stats = None
    mcp_search_memories = None

# All tools combined
ALL_TOOLS = CONTACT_TOOLS + MEMORY_TOOLS + MCP_TOOLS + SQLITE_MCP_TOOLS
