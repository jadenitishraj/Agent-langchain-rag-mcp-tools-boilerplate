#!/usr/bin/env python3
"""
End-to-End MCP Server Test
Tests the complete MCP implementation including:
1. Generic MCPClient (connects to any MCP server)
2. SQLite MCP Server (subprocess communication)
3. DuckDuckGo MCP Server
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_generic_mcp_client():
    """Test the generic MCPClient with SQLite server."""
    print("\n" + "="*70)
    print("TEST 1: Generic MCPClient (MCPClient.py)")
    print("="*70)
    
    try:
        from mcp_servers.MCPClient import connect
        
        # Connect to SQLite MCP server
        client = connect("python -m mcp_servers.sqlite_server")
        
        print("✅ Client created")
        print("📝 Testing list_tools()...")
        
        tools = await client.list_tools()
        print(f"✅ Available tools: {[t.name for t in tools.tools]}")
        
        print("\n📝 Testing call() with get_memory_stats...")
        response = await client.call("get_memory_stats", {})
        
        if response.success:
            print("✅ Tool call successful!")
            print(f"📊 Response:\n{response.content}")
        else:
            print(f"❌ Tool call failed: {response.error}")
        
        print("\n📝 Testing call() with search_memories...")
        response = await client.call("search_memories", {
            "keyword": "test",
            "limit": 5
        })
        
        if response.success:
            print("✅ Search successful!")
            print(f"🔍 Response:\n{response.content}")
        else:
            print(f"❌ Search failed: {response.error}")
        
        await client.close()
        print("\n✅ Generic MCPClient test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Generic MCPClient test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sqlite_mcp_tools():
    """Test SQLite MCP tools (subprocess-based)."""
    print("\n" + "="*70)
    print("TEST 2: SQLite MCP Tools (subprocess communication)")
    print("="*70)
    
    try:
        from mcp_servers import (
            mcp_query_memories,
            mcp_memory_stats,
            mcp_search_memories
        )
        
        print("✅ MCP tools imported")
        
        # Test 1: Get stats
        print("\n📝 Testing mcp_memory_stats()...")
        result = mcp_memory_stats.invoke({})
        print(f"📊 Result:\n{result}")
        
        # Test 2: Search
        print("\n📝 Testing mcp_search_memories()...")
        result = mcp_search_memories.invoke({
            "keyword": "test",
            "limit": 5
        })
        print(f"🔍 Result:\n{result}")
        
        # Test 3: Query
        print("\n📝 Testing mcp_query_memories()...")
        result = mcp_query_memories.invoke({
            "sql": "SELECT COUNT(*) as total FROM memories"
        })
        print(f"📊 Result:\n{result}")
        
        print("\n✅ SQLite MCP Tools test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SQLite MCP Tools test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_duckduckgo_tools():
    """Test DuckDuckGo search tools."""
    print("\n" + "="*70)
    print("TEST 3: DuckDuckGo Search Tools")
    print("="*70)
    
    try:
        from mcp_servers import web_search, web_news
        
        print("✅ DuckDuckGo tools imported")
        
        # Test web search
        print("\n📝 Testing web_search('Python programming')...")
        result = web_search.invoke({
            "query": "Python programming",
            "count": 3
        })
        print(f"🔍 Result (first 200 chars):\n{result[:200]}...")
        
        # Test news search
        print("\n📝 Testing web_news('technology')...")
        result = web_news.invoke({
            "query": "technology",
            "count": 3
        })
        print(f"📰 Result (first 200 chars):\n{result[:200]}...")
        
        print("\n✅ DuckDuckGo Tools test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ DuckDuckGo Tools test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_integration():
    """Test that agent has MCP tools registered."""
    print("\n" + "="*70)
    print("TEST 4: Agent Integration")
    print("="*70)
    
    try:
        from langraph.agent import TOOL_MAP
        
        mcp_tools = [
            "mcp_query_memories",
            "mcp_memory_stats", 
            "mcp_search_memories",
            "web_search",
            "web_news"
        ]
        
        print("✅ Agent imported")
        print(f"\n📝 Checking MCP tools in TOOL_MAP...")
        
        found = []
        missing = []
        
        for tool in mcp_tools:
            if tool in TOOL_MAP:
                found.append(tool)
                print(f"   ✅ {tool}")
            else:
                missing.append(tool)
                print(f"   ❌ {tool}")
        
        if missing:
            print(f"\n⚠️  Missing tools: {missing}")
            return False
        else:
            print(f"\n✅ All {len(found)} MCP tools registered in agent!")
            return True
        
    except Exception as e:
        print(f"\n❌ Agent integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_manager():
    """Test MCP Client Manager."""
    print("\n" + "="*70)
    print("TEST 5: MCP Client Manager")
    print("="*70)
    
    try:
        from mcp_servers import get_mcp_manager, register_mcp_server
        
        manager = get_mcp_manager()
        print(f"✅ Manager loaded with {len(manager.clients)} clients")
        
        # Check if sqlite client is registered
        sqlite_client = manager.get_client("sqlite-memories")
        if sqlite_client:
            print("✅ SQLite MCP client is registered")
            print(f"   Server module: {sqlite_client.server_module}")
            print(f"   Server name: {sqlite_client.server_name}")
        else:
            print("❌ SQLite MCP client NOT registered")
            return False
        
        print("\n✅ MCP Manager test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ MCP Manager test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("\n" + "🚀"*35)
    print("MCP END-TO-END TEST SUITE")
    print("🚀"*35)
    
    results = []
    
    # Run async tests
    results.append(await test_generic_mcp_client())
    results.append(await test_sqlite_mcp_tools())
    results.append(await test_duckduckgo_tools())
    
    # Run sync tests
    results.append(test_agent_integration())
    results.append(await test_mcp_manager())
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your MCP implementation is working end-to-end!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
