import argparse
import asyncio
import json
import sys
import uuid
from urllib.parse import urlencode, urlparse, urlunparse

import websockets


DEFAULT_BASE = "http://127.0.0.1:8080"


def voice_ws_url(base: str, session_id: str) -> str:
    parsed = urlparse(base.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"session_id": session_id})
    return urlunparse((scheme, parsed.netloc, "/voice/ws", "", query, ""))


async def run_case(base: str) -> bool:
    cases = [
        ("Show Paralympic Hot Spots", "filter_to_paralympic"),
        ("Tell me about Vail", "select_hub"),
        ("what about its climate?", None),
    ]
    case_results: list[bool] = []
    session_id = f"voice-smoke-{uuid.uuid4().hex}"

    for turn_id, (prompt, expected_tool) in enumerate(cases, start=1):
        saw_ready = False
        saw_error = False
        saw_tool = expected_tool is None
        saw_transcript = False
        saw_completion = False
        output_text: list[str] = []
        tool_calls: list[dict] = []
        url = voice_ws_url(base, session_id)

        async with websockets.connect(url, open_timeout=30) as ws:
            print(f"connected turn {turn_id}: {url}")
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
                "turn_id": turn_id,
                "text": prompt,
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
                msg_turn_id = int(msg.get("turn_id") or turn_id)
                if msg_turn_id != turn_id and msg.get("type") not in {"ready", "connecting"}:
                    continue

                if msg.get("type") == "tool_calls":
                    tool_calls.extend(msg.get("tool_calls") or [])
                    saw_tool = expected_tool is None or any(call.get("name") == expected_tool for call in tool_calls)

                if msg.get("type") == "output_transcript" and msg.get("text"):
                    saw_transcript = True
                    output_text.append(str(msg["text"]))

                if msg.get("type") == "tool_result_text" and msg.get("text"):
                    saw_transcript = True
                    output_text.append(str(msg["text"]))

                if msg.get("type") == "turn_complete":
                    saw_completion = True

                if msg.get("type") == "voice_state" and msg.get("state") == "idle" and saw_tool:
                    saw_completion = True

                if msg.get("type") == "error":
                    saw_error = True
                    break

                if saw_tool and saw_transcript and saw_completion:
                    break

            text = " ".join(output_text)
            case_ok = (expected_tool is None or saw_tool) and not saw_error and (saw_transcript or saw_completion)
            case_results.append(case_ok)
            print(f"Voice WS turn {turn_id}: {'OK' if case_ok else 'FAIL'}")
            print(f"  prompt: {prompt}")
            print(f"  tools: {tool_calls}")
            print(f"  completed: {saw_completion}")
            print(f"  transcript: {text[:500]}")
            await ws.send(json.dumps({"type": "close"}))

    ok = bool(case_results) and all(case_results)
    print(f"Voice WS smoke: {'OK' if ok else 'FAIL'}")
    print(f"  turns: {len(case_results)}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()
    return 0 if asyncio.run(run_case(args.base)) else 1


if __name__ == "__main__":
    sys.exit(main())
