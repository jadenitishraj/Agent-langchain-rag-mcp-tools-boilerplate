import requests
import json
import time

url = "http://localhost:8000/agent/ask"
payload = {
    "question": "What is the capital of France?",
    "history": []
}
headers = {"Content-Type": "application/json"}

print("--- Testing Cache Content Identity ---")

# 1. First Request
print("\n1. Sending Request 1...")
r1 = requests.post(url, json=payload, headers=headers).json()
ans1 = r1["answer"]
print(f"   Answer 1: {ans1[:50]}...")

# 2. Second Request
print("\n2. Sending Request 2 (Should be Cache Hit)...")
r2 = requests.post(url, json=payload, headers=headers).json()
ans2 = r2["answer"]
print(f"   Answer 2: {ans2[:50]}...")

# 3. Compare
print("\n3. Comparing Answers...")
if ans1 == ans2:
    print("✅ SUCCESS: Answers are IDENTICAL. Cache is working perfectly.")
else:
    print("❌ FAILURE: Answers are DIFFERENT. Cache miss occurred.")
    print(f"   Length 1: {len(ans1)}")
    print(f"   Length 2: {len(ans2)}")
