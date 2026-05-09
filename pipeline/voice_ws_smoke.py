import argparse
import asyncio
import json
import sys
import uuid
from urllib.parse import urlencode, urlparse, urlunparse

import websockets


DEFAULT_BASE = "http://127.0.0.1:8080"
FILLER_PHRASES = (
    "understood",
    "ready when you are",
    "adhere to those guidelines",
)


def voice_ws_url(base: str, session_id: str) -> str:
    parsed = urlparse(base.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"session_id": session_id})
    return urlunparse((scheme, parsed.netloc, "/voice/ws", "", query, ""))


async def run_case(base: str) -> bool:
    cases = [
        {
            "prompt": "Show the top Paralympic Hot Spot",
            "expected_tools": ["select_hub"],
            "contains": ["Anchorage", "13.3%"],
        },
        {
            "prompt": "What is the national baseline?",
            "expected_tools": ["explain_engine"],
            "contains": ["4.7%"],
        },
        {
            "prompt": "Tell me about LA",
            "expected_tools": ["select_hub"],
            "contains": ["Los Angeles"],
        },
        {
            "prompt": "Which hubs are strongest for winter sports?",
            "expected_tools": ["query_data"],
            "contains": ["winter"],
        },
        {
            "prompt": "Reset the map and then show Arizona",
            "expected_tools": ["reset_view", "select_state"],
            "contains": ["Arizona"],
        },
        {
            "prompt": "Is geography producing athletes?",
            "expected_tools": ["explain_engine"],
            "contains": ["does not produce athletes"],
        },
        {
            "prompt": "How did you build the hubs?",
            "expected_tools": ["explain_engine"],
            "contains": ["40 hometown hubs"],
        },
        {
            "prompt": "Tell me about Vail",
            "expected_tools": ["select_hub"],
            "contains": ["Vail", "55 mapped athletes"],
        },
    ]
    case_results: list[bool] = []
    session_id = f"voice-smoke-{uuid.uuid4().hex}"

    for turn_id, case in enumerate(cases, start=1):
        prompt = str(case["prompt"])
        expected_tools = list(case.get("expected_tools") or [])
        expected_contains = [str(value).lower() for value in case.get("contains", [])]
        saw_ready = False
        saw_error = False
        saw_tool = not expected_tools
        saw_transcript = False
        saw_completion = False
        output_text: list[str] = []
        readout_texts: list[str] = []
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
                    seen_tools = [call.get("name") for call in tool_calls]
                    cursor = 0
                    saw_tool = True
                    for expected_tool in expected_tools:
                        while cursor < len(seen_tools) and seen_tools[cursor] != expected_tool:
                            cursor += 1
                        if cursor >= len(seen_tools):
                            saw_tool = False
                            break
                        cursor += 1

                if msg.get("type") == "output_transcript" and msg.get("text"):
                    saw_transcript = True
                    output_text.append(str(msg["text"]))

                if msg.get("type") == "tool_result_text" and msg.get("text"):
                    saw_transcript = True
                    readout = str(msg["text"])
                    output_text.append(readout)
                    readout_texts.append(readout)

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
            lower_text = text.lower()
            has_expected_text = all(value in lower_text for value in expected_contains)
            has_no_filler = not any(phrase in lower_text for phrase in FILLER_PHRASES)
            has_compact_readouts = all(len(readout) <= 520 for readout in readout_texts)
            case_ok = (
                (not expected_tools or saw_tool)
                and not saw_error
                and (saw_transcript or saw_completion)
                and has_expected_text
                and has_no_filler
                and has_compact_readouts
            )
            case_results.append(case_ok)
            print(f"Voice WS turn {turn_id}: {'OK' if case_ok else 'FAIL'}")
            print(f"  prompt: {prompt}")
            print(f"  tools: {tool_calls}")
            print(f"  completed: {saw_completion}")
            print(f"  compact readouts: {all(len(readout) <= 520 for readout in readout_texts)}")
            print(f"  no filler: {has_no_filler}")
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
