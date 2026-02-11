import requests
import json
import time

url = "http://localhost:8000/agent/ask"
payload = {
    "question": "What is the capital of France?",
    "history": [] # Explicitly empty history
}
headers = {"Content-Type": "application/json"}

print("--- Reproducing Cache Consistency Issue (3x Requests) ---")

answers = []

for i in range(1, 4):
    print(f"\n🚀 Sending Request {i}...")
    start = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers)
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            ans = data.get("answer", "")
            answers.append(ans)
            print(f"   ⏱️ Time: {duration:.4f}s")
            print(f"   📝 Answer snippet: {ans[:30]}...")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    time.sleep(1)

print("\n--- Analysis ---")
if len(answers) == 3:
    if answers[0] == answers[1] == answers[2]:
        print("✅ SUCCESS: All 3 answers are IDENTICAL.")
    else:
        print("❌ FAILURE: Answers are DIFFERENT.")
        print(f"1: {answers[0][:50]}...")
        print(f"2: {answers[1][:50]}...")
        print(f"3: {answers[2][:50]}...")
else:
    print("❌ Not enough successful responses to compare.")
