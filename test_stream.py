import requests
import json

url = "http://localhost:8000/agent/ask/stream"
payload = {
    "question": "Tell me a short joke.",
    "history": []
}
headers = {
    "Content-Type": "application/json"
}

print(f"Testing streaming from {url}...")
try:
    with requests.post(url, json=payload, headers=headers, stream=True) as response:
        if response.status_code == 200:
            print("✅ Connection established. Receiving stream...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        try:
                            data = json.loads(json_str)
                            if data['type'] == 'status':
                                print(f"🔹 STATUS: {data['content']}")
                            elif data['type'] == 'token':
                                print(f"📝 TOKEN: {data['content']}")
                            elif data['type'] == 'done':
                                print("✅ DONE Signal received.")
                        except:
                            print(f"RAW: {decoded_line}")
        else:
            print(f"❌ Stream Test Failed: {response.status_code}")
            print(response.text)
except Exception as e:
    print(f"❌ Connection Error: {e}")
