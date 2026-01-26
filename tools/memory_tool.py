"""
Memory Tool - Saves and retrieves user memories from SQLite database

This tool is called when users say:
- "Remember that..."
- "Please save..."
- "Save this to memory..."
- "Don't forget..."
- "Store this..."

It can also retrieve memories when users ask:
- "What do you remember about me?"
- "What did I tell you to remember?"
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional, List
from langchain_core.tools import tool

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "memories.db")

def get_db_connection():
    """Get SQLite database connection, creating table if needed."""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Create memories table if not exists
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
    
    return conn

@tool
def save_memory(memory: str, category: str = "general", user_id: str = "default") -> str:
    """
    Save something to the user's long-term memory.
    
    Use this tool when the user asks you to:
    - Remember something about them (name, preferences, etc.)
    - Save information for later
    - Store something in memory
    - Don't forget something
    - Keep track of something
    
    Common triggers: "remember", "save", "store", "don't forget", "keep in mind"
    
    Args:
        memory: The information to remember (e.g., "User's name is Rahul", "User likes meditation")
        category: Category of memory (personal, preference, fact, reminder). Default: general
        user_id: User identifier. Default: default
    
    Returns:
        Confirmation message
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO memories (user_id, memory, category) VALUES (?, ?, ?)",
            (user_id, memory, category)
        )
        conn.commit()
        memory_id = cursor.lastrowid
        conn.close()
        
        return f"✅ Memory saved successfully! (ID: {memory_id})\nI will remember: {memory}"
    except Exception as e:
        return f"❌ Failed to save memory: {str(e)}"

@tool
def recall_memories(user_id: str = "default", category: Optional[str] = None, limit: int = 10) -> str:
    """
    Retrieve saved memories for a user.
    
    Use this tool when the user asks:
    - What do you remember about me?
    - What did I tell you to save?
    - What are my saved preferences?
    - Recall my memories
    
    Args:
        user_id: User identifier. Default: default
        category: Optional category filter (personal, preference, fact, reminder)
        limit: Maximum number of memories to retrieve. Default: 10
    
    Returns:
        List of saved memories
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if category:
            cursor.execute(
                "SELECT memory, category, created_at FROM memories WHERE user_id = ? AND category = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, category, limit)
            )
        else:
            cursor.execute(
                "SELECT memory, category, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "📭 No memories found. I haven't saved anything yet."
        
        memories = []
        for row in rows:
            date = row['created_at'][:10] if row['created_at'] else 'Unknown'
            memories.append(f"• [{row['category']}] {row['memory']} (saved: {date})")
        
        # return f"🧠 Your saved memories ({len(rows)} found):\n\n" + "\n".join(memories)
        return "🧠 Your saved memories"

    except Exception as e:
        return f"❌ Failed to retrieve memories: {str(e)}"

@tool
def delete_memory(memory_keyword: str, user_id: str = "default") -> str:
    """
    Delete a memory containing a specific keyword.
    
    Use this tool when the user asks to:
    - Forget something
    - Delete a memory
    - Remove something from memory
    
    Args:
        memory_keyword: Keyword to search for in memories to delete
        user_id: User identifier. Default: default
    
    Returns:
        Confirmation message
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM memories WHERE user_id = ? AND memory LIKE ?",
            (user_id, f"%{memory_keyword}%")
        )
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            return f"✅ Deleted {deleted_count} memory(ies) containing '{memory_keyword}'"
        else:
            return f"📭 No memories found containing '{memory_keyword}'"
    except Exception as e:
        return f"❌ Failed to delete memory: {str(e)}"

# Function to get all memories for context injection
def get_all_memories(user_id: str = "default") -> List[str]:
    """Get all memories as a list (for use in system prompt)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT memory FROM memories WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [row['memory'] for row in rows]
    except:
        return []

# List of all memory tools
MEMORY_TOOLS = [save_memory, delete_memory]
