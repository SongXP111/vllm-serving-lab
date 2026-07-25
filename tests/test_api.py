import urllib.request
import urllib.error
import json

BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3-8b-awq"  # fallback; will be overridden by /v1/models if reachable

# ---------------------------------------------------------------------------
# Track pass / fail counts so the script exits with a non-zero code on failure
# ---------------------------------------------------------------------------
_passed = 0
_failed = 0

def _record(ok: bool):
    global _passed, _failed
    if ok:
        _passed += 1
    else:
        _failed += 1

# ---------------------------------------------------------------------------
# 0. Discover the real served model name from /v1/models
# ---------------------------------------------------------------------------
def discover_model_name() -> str:
    """Try to fetch the actual model name from the running vLLM instance.
    Falls back to the hard-coded MODEL_NAME if the endpoint is unreachable."""
    global MODEL_NAME
    req = urllib.request.Request(f"{BASE_URL}/models")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            models = [m['id'] for m in data.get('data', [])]
            if models:
                MODEL_NAME = models[0]
                print(f"ℹ️  Discovered served model name: {MODEL_NAME}")
            else:
                print(f"⚠️  /v1/models returned no models, using fallback: {MODEL_NAME}")
    except Exception as e:
        print(f"⚠️  Could not reach /v1/models ({e}), using fallback: {MODEL_NAME}")
    return MODEL_NAME

# ---------------------------------------------------------------------------
# 1. /v1/models
# ---------------------------------------------------------------------------
def test_models_endpoint():
    print("\n=== Testing /v1/models ===")
    req = urllib.request.Request(f"{BASE_URL}/models")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            models = [m['id'] for m in data.get('data', [])]
            print(f"✅ Success! Available models: {models}")
            if MODEL_NAME not in models:
                print(f"⚠️ Warning: Expected model '{MODEL_NAME}' not found.")
            _record(True)
    except Exception as e:
        print(f"❌ Failed: {e}")
        _record(False)

# ---------------------------------------------------------------------------
# 2. Basic chat completion (Chinese)
# ---------------------------------------------------------------------------
def test_chat_completion():
    print("\n=== Testing Chat Completion (Chinese) ===")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "1+1等于几？请简短回答。"}],
        "temperature": 0,
        "max_tokens": 50
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_data = json.loads(response.read().decode())
            content = resp_data['choices'][0]['message']['content']
            print(f"✅ Success! Response: {content.strip()}")
            _record(True)
    except Exception as e:
        print(f"❌ Failed: {e}")
        _record(False)

# ---------------------------------------------------------------------------
# 3. Chat completion in English
# ---------------------------------------------------------------------------
def test_chat_completion_english():
    print("\n=== Testing Chat Completion (English) ===")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        "temperature": 0,
        "max_tokens": 50
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_data = json.loads(response.read().decode())
            content = resp_data['choices'][0]['message']['content']
            print(f"✅ Success! Response: {content.strip()}")
            _record(True)
    except Exception as e:
        print(f"❌ Failed: {e}")
        _record(False)

# ---------------------------------------------------------------------------
# 4. Multi-turn conversation
# ---------------------------------------------------------------------------
def test_multi_turn():
    print("\n=== Testing Multi-turn Conversation ===")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "我叫小明。"},
            {"role": "assistant", "content": "你好，小明！有什么可以帮你的？"},
            {"role": "user", "content": "我叫什么名字？"}
        ],
        "temperature": 0,
        "max_tokens": 50
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            resp_data = json.loads(response.read().decode())
            content = resp_data['choices'][0]['message']['content']
            print(f"✅ Success! Response: {content.strip()}")
            if "小明" in content:
                print("   ✅ Model correctly recalled the user's name.")
            else:
                print("   ⚠️ Model did not recall the user's name (may still be acceptable).")
            _record(True)
    except Exception as e:
        print(f"❌ Failed: {e}")
        _record(False)

# ---------------------------------------------------------------------------
# 5. Long context (~2K tokens input)
# ---------------------------------------------------------------------------
def test_long_context():
    print("\n=== Testing Long Context Input ===")
    # Build a prompt that is roughly 2K+ tokens to stress-test context handling
    long_passage = (
        "The following is a long passage designed to test the model's ability "
        "to handle longer inputs. " * 150  # ~2K tokens
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": f"{long_passage}\n\nSummarize the above passage in one sentence."}
        ],
        "temperature": 0,
        "max_tokens": 100
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            resp_data = json.loads(response.read().decode())
            content = resp_data['choices'][0]['message']['content']
            usage = resp_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', '?')
            print(f"✅ Success! Prompt tokens: {prompt_tokens}")
            print(f"   Response: {content.strip()[:120]}...")
            _record(True)
    except Exception as e:
        print(f"❌ Failed: {e}")
        _record(False)

# ---------------------------------------------------------------------------
# 6. Streaming (SSE)
# ---------------------------------------------------------------------------
def test_streaming():
    print("\n=== Testing Streaming (SSE) ===")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "解释一下大模型里的 Continuous Batching，请详细说明。"}],
        "stream": True,
        "temperature": 0,
        "max_tokens": 100
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            print("✅ Streaming started: ", end="")
            chunk_count = 0
            for line in response:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        print(delta['content'], end="", flush=True)
                        chunk_count += 1
            print(f"\n✅ Streaming finished. Received {chunk_count} content chunks.")
            _record(True)
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        _record(False)

# ---------------------------------------------------------------------------
# 7. Error handling – invalid model name
# ---------------------------------------------------------------------------
def test_error_handling():
    print("\n=== Testing Error Handling (Invalid Model) ===")
    payload = {
        "model": "invalid-model-name",
        "messages": [{"role": "user", "content": "hello"}],
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            print("❌ Failed: Expected an HTTP error but got 200 OK.")
            _record(False)
    except urllib.error.HTTPError as e:
        error_resp = json.loads(e.read().decode())
        print(f"✅ Success! Caught expected error {e.code}: {error_resp.get('error', {}).get('message')}")
        _record(True)
    except Exception as e:
        print(f"❌ Failed with unexpected error: {e}")
        _record(False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  vLLM API Smoke Test")
    print("=" * 60)

    discover_model_name()

    test_models_endpoint()
    test_chat_completion()
    test_chat_completion_english()
    test_multi_turn()
    test_long_context()
    test_streaming()
    test_error_handling()

    # Summary
    total = _passed + _failed
    print("\n" + "=" * 60)
    print(f"  Results: {_passed}/{total} passed, {_failed}/{total} failed")
    print("=" * 60)

    if _failed > 0:
        raise SystemExit(1)
