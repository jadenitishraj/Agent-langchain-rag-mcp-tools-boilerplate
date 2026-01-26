# mcp.py
# MCP (Model Context Protocol) Module
#
# This module provides:
# 1. DuckDuckGo web search (FREE, no API key)
# 2. SQLite MCP Server (via generic MCP client)
#
# No managers. No lifecycle. No server registry.
# Just connections + tools.

# ------------------------------------------------
# DuckDuckGo search tools (pure local tools)
# ------------------------------------------------
from .mcp_client import web_search, web_news, MCP_TOOLS, get_mcp_tools


# ------------------------------------------------
# SQLite MCP tools (via MCP server)
# ------------------------------------------------
from .sqlite_client import (
    mcp_query_memories,
    mcp_memory_stats,
    mcp_search_memories,
    SQLITE_MCP_TOOLS,
    get_sqlite_mcp_tools,
)


# ------------------------------------------------
# Generic MCP client
# ------------------------------------------------
from .MCPClient import connect, MCPClient


# ------------------------------------------------
# MCP Connections (1-liners)
# ------------------------------------------------
sqlite = connect("python -m mcp_servers.sqlite_server")


# ------------------------------------------------
# All MCP tools for agent
# ------------------------------------------------
ALL_MCP_TOOLS = MCP_TOOLS + SQLITE_MCP_TOOLS
