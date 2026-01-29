"""
SQLite MCP Client - Uses the reusable MCPClient

This wraps the generic MCPClient to provide SQLite-specific functionality.
"""
import os
import sys
import asyncio
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

# Import the reusable MCPClient
from .MCPClient import MCPClient, MCPResponse, connect

# ============================================================================
# MCP Client Manager - Uses the reusable MCPClient
# ============================================================================

class MCPClientManager:
    """
    Manages multiple MCP client connections using the reusable MCPClient.
    
    Use this to add more MCP servers in the future!
    """
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
    
    def register_server(self, name: str, command: str) -> MCPClient:
        """
        Register and start an MCP server.
        
        Args:
            name: Human-readable name for the server
            command: Command to start the server (e.g., "python -m mcp_servers.sqlite_server")
        
        Returns:
            MCPClient instance
        """
        if name not in self.clients:
            client = connect(command)
            self.clients[name] = client
            print(f"✅ Registered MCP server: {name}", file=sys.stderr)
        return self.clients[name]
    
    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get an MCP client by name."""
        return self.clients.get(name)
    
    async def stop_all(self):
        """Stop all MCP servers."""
        for name, client in self.clients.items():
            print(f"🛑 Stopping MCP server: {name}", file=sys.stderr)
            await client.close()
        self.clients.clear()

# Global manager instance
mcp_manager = MCPClientManager()

# Register SQLite MCP server using the reusable client!
sqlite_client = mcp_manager.register_server(
    name="sqlite-memories",
    command="python -m mcp_servers.sqlite_server"
)

# ============================================================================
# LangChain Tools - Use the reusable MCPClient
# ============================================================================

def _call_mcp_tool_sync(tool_name: str, arguments: dict) -> MCPResponse:
    """Helper to call MCP tools synchronously from LangChain tools."""
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # If we're here, we're in an async context - use executor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_async_in_new_loop, tool_name, arguments)
            return future.result()
    except RuntimeError:
        # No loop running, we can safely create one
        return _run_async_in_new_loop(tool_name, arguments)

def _run_async_in_new_loop(tool_name: str, arguments: dict) -> MCPResponse:
    """Run async MCP call in a new event loop with a fresh client."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Create a fresh client for this call to avoid session reuse issues
        client = connect("python -m mcp_servers.sqlite_server")
        result = loop.run_until_complete(client.call(tool_name, arguments))
        # Close the client
        loop.run_until_complete(client.close())
        return result
    except Exception as e:
        return MCPResponse(success=False, content="", error=str(e))
    finally:
        loop.close()
        asyncio.set_event_loop(None)

@tool
def mcp_query_memories(sql: str) -> str:
    """
    Query the memories database using SQL (via MCP Server subprocess).
    
    This tool connects to a REAL MCP server running as a separate process!
    Only SELECT queries are allowed for safety.
    
    Use for:
    - Analytics: "How many memories about meditation?" 
    - Filtering: "Show memories from last week"
    - Aggregation: "What are my top categories?"
    
    Args:
        sql: A SELECT SQL query
    
    Returns:
        Query results from the MCP server
    """
    response = _call_mcp_tool_sync("query_memories", {"sql": sql})
    
    if response.success:
        return response.content
    return f"❌ MCP Error: {response.error}"

@tool
def mcp_memory_stats() -> str:
    """
    Get memory statistics via MCP Server (separate process).
    
    Returns comprehensive stats about saved memories:
    - Total count
    - Breakdown by category
    - Date range
    - Recent memories preview
    """
    response = _call_mcp_tool_sync("get_memory_stats", {})
    
    if response.success:
        return response.content
    return f"❌ MCP Error: {response.error}"

@tool
def mcp_search_memories(keyword: str, limit: int = 10) -> str:
    """
    Search memories by keyword via MCP Server (separate process).
    
    Args:
        keyword: Word or phrase to search for
        limit: Maximum results (default 10)
    """
    response = _call_mcp_tool_sync("search_memories", {"keyword": keyword, "limit": limit})
    
    if response.success:
        return response.content
    return f"❌ MCP Error: {response.error}"

# All SQLite MCP tools
SQLITE_MCP_TOOLS = [mcp_query_memories, mcp_memory_stats, mcp_search_memories]

def get_sqlite_mcp_tools() -> List:
    """Get all SQLite MCP tools."""
    return SQLITE_MCP_TOOLS

def get_mcp_manager() -> MCPClientManager:
    """Get the global MCP manager."""
    return mcp_manager
