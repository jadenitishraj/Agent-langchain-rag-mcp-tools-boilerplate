"""
MCP Client - FREE Web Search using DuckDuckGo

No API key required! Uses the ddgs package.
"""
import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.tools import tool

# Use the newer ddgs package
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

@dataclass
class MCPToolResult:
    """Result from an MCP tool call."""
    success: bool
    content: str
    error: Optional[str] = None

# ============================================================================
# LangChain Tool Wrappers (using DuckDuckGo - FREE!)
# ============================================================================

@tool
def web_search(query: str, count: int = 5) -> str:
    """
    Search the web using DuckDuckGo (FREE, no API key needed).
    
    Use this tool when:
    - The user asks about current events, news, or recent information
    - The RAG context doesn't have relevant information  
    - The user explicitly asks to "search online" or "look up"
    - Questions about topics not covered in the codebase
    - User asks about weather, sports, current events, technology
    
    DO NOT use this for:
    - Questions about this codebase (use RAG instead)
    - Personal memory questions (use memory tools)
    - Contact information (use contact tool)
    
    Args:
        query: The search query string
        count: Number of results to return (1-10)
    
    Returns:
        Search results with titles, descriptions, and URLs
    """
    try:
        count = min(max(count, 1), 10)
        
        ddgs = DDGS()
        results = ddgs.text(query, max_results=count)
        
        if not results:
            return f"🔍 No results found for: {query}"
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No description")[:200]
            url = r.get("href", "")
            formatted.append(f"{i}. **{title}**\n   {body}...\n   🔗 {url}")
        
        return f"🔍 Search results for: '{query}'\n\n" + "\n\n".join(formatted)
        
    except Exception as e:
        return f"❌ Search failed: {str(e)}"

@tool
def web_news(query: str, count: int = 5) -> str:
    """
    Search for latest news using DuckDuckGo News (FREE).
    
    Use this tool when:
    - User asks about "latest news", "recent news", "what's happening"
    - User wants current events information
    - Questions about recent developments in any topic
    
    Args:
        query: The news search query
        count: Number of results (1-10)
    
    Returns:
        News articles with titles, dates, sources, and URLs
    """
    try:
        count = min(max(count, 1), 10)
        
        ddgs = DDGS()
        results = ddgs.news(query, max_results=count)
        
        if not results:
            return f"📰 No news found for: {query}"
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "")[:150]
            date = r.get("date", "Unknown")
            source = r.get("source", "Unknown")
            url = r.get("url", "")
            formatted.append(f"{i}. **{title}**\n   📅 {date} | 📰 {source}\n   {body}...\n   🔗 {url}")
        
        return f"📰 News for: '{query}'\n\n" + "\n\n".join(formatted)
        
    except Exception as e:
        return f"❌ News search failed: {str(e)}"

# All MCP-based tools
MCP_TOOLS = [web_search, web_news]

def get_mcp_tools() -> List:
    """Get all MCP-based tools."""
    return MCP_TOOLS
