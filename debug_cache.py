from rag_v2.cache_manager import search_cache, save_to_cache
import time

print("--- Debugging Cache Manager ---")

query = "What is the capital of France?"
response = "The capital of France is Paris."

# 1. Save
print(f"💾 Saving: '{query}' -> '{response}'")
save_to_cache(query, response)

# 2. Wait a moment (Qdrant index update)
time.sleep(2)

# 3. Search Exact
print(f"🔍 Searching Exact: '{query}'")
hit = search_cache(query)
if hit:
    print(f"✅ Hit: {hit}")
else:
    print("❌ Miss (Expected Hit)")

# 4. Search Similar - Debug Score
query_sim = "Tell me the capital city of France"
print(f"🔍 Searching Similar: '{query_sim}'")

from rag_v2.vector_store import get_client
from rag_v2.cache_manager import CACHE_COLLECTION
from rag_v2.embedder import embed_text
client = get_client()
vector = embed_text(query_sim)
results = client.query_points(
    collection_name=CACHE_COLLECTION,
    query=vector,
    limit=1,
    with_payload=True
).points

if results:
    hit = results[0]
    print(f"   ℹ️  Top Score: {hit.score:.4f}")
    if hit.score >= 0.70:
         print(f"✅ Hit: {hit.payload.get('response')}")
    else:
         print(f"❌ Miss (Score {hit.score:.4f} < 0.70)")
else:
    print("❌ No results found at all.")
