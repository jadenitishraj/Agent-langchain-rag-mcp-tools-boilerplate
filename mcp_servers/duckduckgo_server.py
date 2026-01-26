"""
DuckDuckGo Search MCP Server
FREE web search via MCP protocol - NO API KEY REQUIRED!

Uses DuckDuckGo which is:
- Free forever
- No API key required
- No rate limits (within reason)
- Privacy-focused
"""
import asyncio
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent
from duckduckgo_search import DDGS

# Create the MCP server
server = Server("duckduckgo-search")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="web_search",
            description="""Search the web using DuckDuckGo (FREE, no API key).
            
Use this tool when:
- The user asks about current events, news, or recent information
- The RAG context doesn't have relevant information
- The user explicitly asks to search online
- Questions about topics not in the codebase
- User says "search", "look up", "find online"

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
        ),
        Tool(
            name="web_news",
            description="""Search for news articles using DuckDuckGo News (FREE).
            
Use this tool when:
- User asks about "latest news", "recent news", "current events"
- User wants to know what's happening in the world

Args:
    query: The news search query
    count: Number of results (default: 5)
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The news search query"
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
    
    query = arguments.get("query", "")
    count = min(arguments.get("count", 5), 10)
    
    if not query:
        return [TextContent(type="text", text="❌ No search query provided")]
    
    try:
        if name == "web_search":
            # Web search
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=count))
            
            if not results:
                return [TextContent(type="text", text=f"🔍 No results found for: {query}")]
            
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                body = r.get("body", "No description")
                url = r.get("href", "")
                formatted.append(f"{i}. **{title}**\n   {body[:200]}...\n   🔗 {url}")
            
            return [TextContent(
                type="text",
                text=f"🔍 Search results for: '{query}'\n\n" + "\n\n".join(formatted)
            )]
        
        elif name == "web_news":
            # News search
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=count))
            
            if not results:
                return [TextContent(type="text", text=f"📰 No news found for: {query}")]
            
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                body = r.get("body", "")[:150]
                date = r.get("date", "")
                source = r.get("source", "Unknown")
                url = r.get("url", "")
                formatted.append(f"{i}. **{title}**\n   📅 {date} | 📰 {source}\n   {body}...\n   🔗 {url}")
            
            return [TextContent(
                type="text",
                text=f"📰 News results for: '{query}'\n\n" + "\n\n".join(formatted)
            )]
        
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Search error: {str(e)}")]

async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

def run_search_server():
    """Entry point to run the server."""
    asyncio.run(main())

if __name__ == "__main__":
    run_search_server()
