import requests
import json
import time

url = "http://localhost:8000/agent/ask"
payload = {
    "question": "What is Python?",
    "history": []
}
headers = {"Content-Type": "application/json"}

def run_test(name):
    print(f"\n--- {name} ---")
    start = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers)
        duration = time.time() - start
        if response.status_code == 200:
            print(f"✅ Response: {response.json()['answer'][:50]}...")
            print(f"⏱️ Time taken: {duration:.4f} seconds")
            return duration
        else:
            print(f"❌ Failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# 1. First Run (No Cache)
print("🔍 Request 1 (Should be slow/normal)")
time1 = run_test("First Run")

# 2. Second Run (Cache Hit)
print("🔍 Request 2 (Should be FAST - Cache Hit)")
time2 = run_test("Second Run")

if time1 and time2 and time2 < time1 * 0.5:
    print("\n✅ Semantic Caching Works! (Speedup: {:.2f}x)".format(time1/time2))
else:
    print("\n⚠️ Caching might not be working as expected (or first run was too fast). check logs.")
