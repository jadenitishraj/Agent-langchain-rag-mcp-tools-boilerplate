from rag_v2.cache_manager import search_cache

print("--- Testing Cache Persistence (Read) ---")
query = "Persistence Test Query"
expected = "Persistence Test Response"

print(f"🔍 Searching (New Process): '{query}'")
hit = search_cache(query)

if hit == expected:
    print("✅ Persistence Success: Retrieved correct value.")
else:
    print(f"❌ Persistence Failed. Got: {hit}")
