import requests
import json

url = "http://localhost:8000/agent/ask"
payload = {
    "question": "Who created Python?",
    "history": []
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print("✅ API Test Passed!")
        print(f"Response: {response.json()['answer']}")
    else:
        print(f"❌ API Test Failed: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Connection Error: {e}")
