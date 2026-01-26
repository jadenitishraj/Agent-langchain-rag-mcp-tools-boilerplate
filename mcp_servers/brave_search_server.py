"""
Brave Search MCP Server
Provides web search capabilities via MCP protocol

To run as standalone server:
    python -m mcp_servers.brave_search_server

Environment:
    BRAVE_API_KEY: Your Brave Search API key
"""
import os
import asyncio
import json
import httpx
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Brave Search API
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Create the MCP server
server = Server("brave-search")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="brave_web_search",
            description="""Search the web using Brave Search.
            
Use this tool when:
- The user asks about current events or news
- The RAG context doesn't have relevant information
- The user explicitly asks to search online
- Questions about topics not in the codebase

Args:
    query: The search query string
    count: Number of results to return (default: 5, max: 10)
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results (1-10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a tool call."""
    
    if name == "brave_web_search":
        query = arguments.get("query", "")
        count = min(arguments.get("count", 5), 10)
        
        if not BRAVE_API_KEY:
            return [TextContent(
                type="text",
                text="❌ BRAVE_API_KEY not configured. Please set the environment variable."
            )]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    BRAVE_SEARCH_URL,
                    headers={
                        "X-Subscription-Token": BRAVE_API_KEY,
                        "Accept": "application/json"
                    },
                    params={
                        "q": query,
                        "count": count
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
            
            # Extract results
            web_results = data.get("web", {}).get("results", [])
            
            if not web_results:
                return [TextContent(
                    type="text",
                    text=f"🔍 No results found for: {query}"
                )]
            
            # Format results
            formatted_results = []
            for i, result in enumerate(web_results[:count], 1):
                title = result.get("title", "No title")
                description = result.get("description", "No description")
                url = result.get("url", "")
                formatted_results.append(
                    f"{i}. **{title}**\n   {description}\n   🔗 {url}"
                )
            
            result_text = f"🔍 Search results for: '{query}'\n\n" + "\n\n".join(formatted_results)
            
            return [TextContent(type="text", text=result_text)]
            
        except httpx.HTTPError as e:
            return [TextContent(
                type="text",
                text=f"❌ Search failed: {str(e)}"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Error: {str(e)}"
            )]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

def run_brave_search_server():
    """Entry point to run the server."""
    asyncio.run(main())

if __name__ == "__main__":
    run_brave_search_server()
