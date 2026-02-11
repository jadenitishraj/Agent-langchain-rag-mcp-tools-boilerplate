import requests
import json

url = "http://localhost:8000/feedback/"
payload = {
    "user_id": "test_user_123",
    "query": "What is RAG?",
    "response": "Retrieval Augmented Generation...",
    "score": 1,
    "comment": "Great explanation!"
}
headers = {"Content-Type": "application/json"}

print(f"--- Testing Feedback API ({url}) ---")
try:
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print("✅ Success!")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Connection Error: {e}")
