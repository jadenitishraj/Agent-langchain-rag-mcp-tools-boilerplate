# MCP (Model Context Protocol) Module
# 
# This module provides:
# 1. DuckDuckGo web search (FREE, no API key)
# 2. SQLite MCP Server (True MCP with subprocess!)
# 3. MCP Client Manager (for connecting multiple MCPs)

# DuckDuckGo search tools
from .mcp_client import web_search, web_news, MCP_TOOLS, get_mcp_tools

# SQLite MCP tools (True MCP with subprocess!)
from .sqlite_client import (
    mcp_query_memories,
    mcp_memory_stats, 
    mcp_search_memories,
    SQLITE_MCP_TOOLS,
    get_sqlite_mcp_tools,
    TrueMCPClient,
    MCPClientManager,
    mcp_manager,
    get_mcp_manager
)

# All MCP tools
ALL_MCP_TOOLS = MCP_TOOLS + SQLITE_MCP_TOOLS

def register_mcp_server(name: str, module: str) -> TrueMCPClient:
    """
    Register a new MCP server.
    
    Example:
        # Add a new MCP server
        client = register_mcp_server("my-server", "mcp_servers.my_server")
    """
    return mcp_manager.register_server(name, module)

def stop_all_mcp_servers():
    """Stop all MCP servers (call on shutdown)."""
    mcp_manager.stop_all()
