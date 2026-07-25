#!/usr/bin/env python3
"""
vLLM Serving Lab - Interactive Chat Client (chat_client.py)

An OpenAI-compatible streaming client for interacting with the vLLM server.
Supports single-shot prompt execution and multi-turn interactive CLI chat sessions.
Reads VLLM_PORT, VLLM_API_KEY, and VLLM_SERVED_MODEL automatically from .env if present.
"""
import sys
import os
import json
import urllib.request
import urllib.error
import argparse

# ── Load environment variables from .env if present ───────
def load_env_file():
    # Find .env in project root (parent directory of clients/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    env_path = os.path.join(project_root, ".env")
    
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    if key not in os.environ:
                        os.environ[key] = val

load_env_file()

PORT = os.environ.get("VLLM_PORT", "8000")
API_KEY = os.environ.get("VLLM_API_KEY", "")
DEFAULT_MODEL = os.environ.get("VLLM_SERVED_MODEL", "qwen3-8b-awq")
BASE_URL = f"http://localhost:{PORT}/v1/chat/completions"

def stream_chat_completion(model_name, messages, system_prompt=None):
    """Send chat messages to vLLM server and stream response tokens to stdout."""
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    payload = {
        "model": model_name,
        "messages": full_messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE_URL, data=data, headers=headers)

    assistant_response = ""
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        assistant_response += content
        sys.stdout.write("\n")
        return assistant_response
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"\n❌ HTTP Error {e.code}: {err_msg}")
        return None
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")
        return None

def run_interactive_session(model_name, system_prompt):
    print("=" * 60)
    print("      💬 vLLM Serving Lab - Interactive Chat Session")
    print("=" * 60)
    print(f"🌐 Server Endpoint : http://localhost:{PORT}/v1/chat/completions")
    print(f"🤖 Model           : {model_name}")
    print(f"🔑 Auth API Key    : {'Configured' if API_KEY else 'None (Open access)'}")
    print("💡 Commands        : Type 'exit', 'quit', or 'clear' to manage session")
    print("=" * 60 + "\n")

    history = []

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Exiting chat session. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("👋 Exiting chat session. Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("🧹 Conversation history cleared.\n")
            continue

        history.append({"role": "user", "content": user_input})
        sys.stdout.write("🤖 Assistant: ")
        sys.stdout.flush()

        response = stream_chat_completion(model_name, history, system_prompt)
        if response:
            history.append({"role": "assistant", "content": response})
        else:
            # Remove failed turn
            history.pop()
        print()

def main():
    parser = argparse.ArgumentParser(description="vLLM Serving Lab Chat Client")
    parser.add_argument("-p", "--prompt", type=str, help="Single-shot user prompt to send")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help=f"Served model name (default: {DEFAULT_MODEL})")
    parser.add_argument("-s", "--system", type=str, default="You are a helpful and precise AI assistant.", help="System prompt")
    args = parser.parse_args()

    if args.prompt:
        # Single-shot mode
        sys.stdout.write(f"🤖 Assistant ({args.model}): ")
        sys.stdout.flush()
        stream_chat_completion(args.model, [{"role": "user", "content": args.prompt}], args.system)
    else:
        # Interactive mode
        run_interactive_session(args.model, args.system)

if __name__ == "__main__":
    main()
