import json
import sys
import urllib.request


port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
payload = {
    "model": "qwen2.5-7b-plugin-smoke",
    "messages": [
        {
            "role": "user",
            "content": "请用两句话简要介绍 Transformer 模型的核心思想。",
        }
    ],
    "temperature": 0.0,
    "max_tokens": 80,
    "seed": 0,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.loads(response.read())

content = result["choices"][0]["message"]["content"]
assert content.strip(), "model returned empty content"
print(json.dumps(result, ensure_ascii=False, indent=2))
