# MCP (Model Context Protocol) Module
# 
# This module provides:
# 1. DuckDuckGo web search (FREE, no API key)
# 2. SQLite MCP Server (True MCP with subprocess!)
# 3. Reusable MCPClient for connecting to ANY MCP server

# DuckDuckGo search tools
from .mcp_client import web_search, web_news, MCP_TOOLS, get_mcp_tools

# SQLite MCP tools (True MCP with subprocess!)
from .sqlite_client import (
    mcp_query_memories,
    mcp_memory_stats, 
    mcp_search_memories,
    SQLITE_MCP_TOOLS,
    get_sqlite_mcp_tools,
    MCPClientManager,
    mcp_manager,
    get_mcp_manager
)

# Reusable MCP Client
from .MCPClient import MCPClient, connect

# All MCP tools
ALL_MCP_TOOLS = MCP_TOOLS + SQLITE_MCP_TOOLS

def register_mcp_server(name: str, command: str) -> MCPClient:
    """
    Register a new MCP server using the reusable MCPClient.
    
    Args:
        name: Human-readable name for the server
        command: Command to start server (e.g., "python -m mcp_servers.my_server")
    
    Example:
        # Add a new MCP server
        client = register_mcp_server("weather", "python -m mcp_servers.weather_server")
    """
    return mcp_manager.register_server(name, command)

async def stop_all_mcp_servers():
    """Stop all MCP servers (call on shutdown)."""
    await mcp_manager.stop_all()
