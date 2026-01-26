"""
SQLite MCP Server - Real MCP Implementation

This server:
1. Runs as a SEPARATE process
2. Communicates via stdio (MCP protocol)
3. Exposes SQLite query capabilities
4. Follows the MCP specification

Run with: python -m mcp_servers.sqlite_server
"""
import os
import sys
import json
import sqlite3
import asyncio
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "memories.db")

# Create MCP Server
server = Server("sqlite-memories")

def get_db_connection():
    """Get SQLite connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_table_exists():
    """Ensure memories table exists."""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            memory TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Ensure table exists on startup
ensure_table_exists()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available tools - this is called by the MCP client
    to discover what this server can do.
    """
    return [
        Tool(
            name="query_memories",
            description="""Execute a READ-ONLY SQL query on the memories database.
            
Use this for:
- Counting memories: SELECT COUNT(*) FROM memories
- Filtering: SELECT * FROM memories WHERE memory LIKE '%keyword%'
- Analytics: SELECT category, COUNT(*) FROM memories GROUP BY category
- Time queries: SELECT * FROM memories WHERE created_at > date('now', '-7 days')

Table schema:
- id: INTEGER (primary key)
- user_id: TEXT (default 'default')
- memory: TEXT (the saved memory content)
- category: TEXT (personal, preference, fact, general)
- created_at: TIMESTAMP
- updated_at: TIMESTAMP

ONLY SELECT queries allowed for safety.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT query to execute"
                    }
                },
                "required": ["sql"]
            }
        ),
        Tool(
            name="get_memory_stats",
            description="""Get statistics about saved memories.
            
Returns:
- Total count
- Categories breakdown
- Oldest and newest memory dates
- Recent memories preview""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="search_memories",
            description="""Search memories by keyword.
            
Args:
    keyword: Word or phrase to search for
    limit: Max results (default 10)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10
                    }
                },
                "required": ["keyword"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    Execute a tool call - this is where the actual work happens.
    """
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if name == "query_memories":
            sql = arguments.get("sql", "")
            
            # Safety: Only allow SELECT queries
            if not sql.strip().upper().startswith("SELECT"):
                return [TextContent(
                    type="text",
                    text="❌ Only SELECT queries are allowed for safety. Use other tools for modifications."
                )]
            
            try:
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                if not rows:
                    return [TextContent(type="text", text="📭 No results found.")]
                
                # Format results
                columns = [description[0] for description in cursor.description]
                results = []
                for row in rows[:50]:  # Limit to 50 rows
                    row_dict = {columns[i]: row[i] for i in range(len(columns))}
                    results.append(row_dict)
                
                return [TextContent(
                    type="text",
                    text=f"📊 Query Results ({len(results)} rows):\n\n{json.dumps(results, indent=2, default=str)}"
                )]
                
            except sqlite3.Error as e:
                return [TextContent(type="text", text=f"❌ SQL Error: {str(e)}")]
        
        elif name == "get_memory_stats":
            # Total count
            cursor.execute("SELECT COUNT(*) as total FROM memories")
            total = cursor.fetchone()[0]
            
            # Categories
            cursor.execute("SELECT category, COUNT(*) as count FROM memories GROUP BY category")
            categories = cursor.fetchall()
            
            # Date range
            cursor.execute("SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM memories")
            dates = cursor.fetchone()
            
            # Recent memories
            cursor.execute("SELECT memory, category FROM memories ORDER BY created_at DESC LIMIT 5")
            recent = cursor.fetchall()
            
            stats = f"""📊 Memory Statistics:
━━━━━━━━━━━━━━━━━━━━━━
📝 Total memories: {total}

📁 By category:
{chr(10).join([f"   • {cat[0]}: {cat[1]}" for cat in categories]) if categories else "   No categories yet"}

📅 Date range:
   • Oldest: {dates[0] or 'N/A'}
   • Newest: {dates[1] or 'N/A'}

🕐 Recent memories:
{chr(10).join([f"   • [{m[1]}] {m[0][:50]}..." for m in recent]) if recent else "   No memories yet"}
"""
            return [TextContent(type="text", text=stats)]
        
        elif name == "search_memories":
            keyword = arguments.get("keyword", "")
            limit = arguments.get("limit", 10)
            
            cursor.execute(
                "SELECT id, memory, category, created_at FROM memories WHERE memory LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{keyword}%", limit)
            )
            rows = cursor.fetchall()
            
            if not rows:
                return [TextContent(type="text", text=f"🔍 No memories found containing '{keyword}'")]
            
            results = [f"🔍 Found {len(rows)} memories containing '{keyword}':\n"]
            for row in rows:
                results.append(f"• [{row[2]}] {row[1]} (ID: {row[0]}, {row[3][:10]})")
            
            return [TextContent(type="text", text="\n".join(results))]
        
        else:
            return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error: {str(e)}")]
    
    finally:
        conn.close()

async def main():
    # Force line buffering for stdout to ensure MCP messages are sent immediately
    sys.stdout.reconfigure(line_buffering=True)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
