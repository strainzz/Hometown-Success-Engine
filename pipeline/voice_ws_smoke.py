import argparse
import asyncio
import json
import sys
from urllib.parse import urlparse, urlunparse

import websockets


DEFAULT_BASE = "http://127.0.0.1:8080"


def voice_ws_url(base: str) -> str:
    parsed = urlparse(base.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "/voice/ws", "", "", ""))


async def run_case(base: str) -> bool:
    url = voice_ws_url(base)
    saw_ready = False
    saw_tool = False
    saw_transcript = False
    saw_completion = False
    saw_error = False
    output_text: list[str] = []
    tool_calls: list[dict] = []

    async with websockets.connect(url, open_timeout=30) as ws:
        while not saw_ready:
            raw = await asyncio.wait_for(ws.recv(), timeout=45)
            msg = json.loads(raw)
            print("recv:", msg)
            if msg.get("type") == "ready":
                saw_ready = True
            if msg.get("type") == "error":
                saw_error = True
                break

        if saw_error:
            return False

        await ws.send(json.dumps({
            "type": "text",
            "text": "Show Paralympic Hot Spots",
            "audio_enabled": False,
        }))

        deadline = asyncio.get_running_loop().time() + 60
        while asyncio.get_running_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            print("recv:", msg)

            if msg.get("type") == "tool_calls":
                tool_calls.extend(msg.get("tool_calls") or [])
                saw_tool = any(call.get("name") == "filter_to_paralympic" for call in tool_calls)

            if msg.get("type") == "output_transcript" and msg.get("text"):
                saw_transcript = True
                output_text.append(str(msg["text"]))

            if msg.get("type") == "tool_result_text" and msg.get("text"):
                saw_transcript = True
                output_text.append(str(msg["text"]))

            if msg.get("type") == "voice_state" and msg.get("state") == "idle" and saw_tool:
                saw_completion = True

            if msg.get("type") == "error":
                saw_error = True
                break

            if saw_tool and (saw_transcript or saw_completion):
                break

        await ws.send(json.dumps({"type": "close"}))

    text = " ".join(output_text)
    ok = saw_ready and saw_tool and not saw_error and (saw_transcript or saw_completion)
    if "Paralympic" in text:
        ok = ok and True
    print(f"Voice WS smoke: {'OK' if ok else 'FAIL'}")
    print(f"  ready: {saw_ready}")
    print(f"  tools: {tool_calls}")
    print(f"  completed: {saw_completion}")
    print(f"  transcript: {text[:500]}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()
    return 0 if asyncio.run(run_case(args.base)) else 1


if __name__ == "__main__":
    sys.exit(main())
