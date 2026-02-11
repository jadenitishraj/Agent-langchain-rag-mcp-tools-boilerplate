from rag_v2.cache_manager import save_to_cache, search_cache
import time
import shutil
import os
from rag_v2.config import QDRANT_PATH

print("--- Testing Cache Persistence ---")

# 1. Clean start (optional, maybe too aggressive if we want to debug existing state)
# But let's verify if we can save and retrieve in THIS process first.
query = "Persistence Test Query"
response = "Persistence Test Response"

print(f"💾 Saving: '{query}'")
save_to_cache(query, response)

time.sleep(2) # Allow write

print(f"🔍 Searching (Immediate): '{query}'")
hit = search_cache(query)
if hit == response:
    print("✅ Immediate Read Success")
else:
    print(f"❌ Immediate Read Failed. Got: {hit}")

# now we simulate a 'restart' by just getting a new client instance if possible, 
# but the client is a global singleton in vector_store.py.
# To truly test persistence, we should run a separate process to read it.
