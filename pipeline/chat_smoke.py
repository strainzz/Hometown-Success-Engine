import argparse
import json
import sys
import urllib.request


DEFAULT_BASE = "http://127.0.0.1:8080"


CASES = [
    {
        "prompt": "Show Paralympic Hot Spots",
        "tool": "filter_to_paralympic",
        "contains": ["10 Paralympic Hot Spots", "7.5%"],
    },
    {
        "prompt": "Tell me about Vail",
        "tool": "select_hub",
        "contains": ["Vail Region, CO", "36.6"],
        "tool_args": {"hub_id": "HUB_CO_VAIL"},
    },
    {
        "prompt": "Tell me about Salt Lake City",
        "tool": "select_hub",
        "contains": ["Salt Lake City Region, UT"],
        "tool_args": {"hub_id": "HUB_UT_SALT_LAKE_CITY"},
    },
    {
        "prompt": "How many athletes and hubs?",
        "tool": "query_data",
        "contains": ["5,119", "40"],
    },
    {
        "prompt": "Reset the view",
        "tool": "reset_view",
        "contains": ["Map reset", "5,119", "40"],
    },
    {
        "prompt": "Rank the top 5 hubs by Paralympic share",
        "tool": "query_data",
        "contains": ["Anchorage Region, AK", "Phoenix Region, AZ", "13.3%"],
    },
    {
        "prompt": "What rank is Vail by Paralympic share?",
        "tool": "query_data",
        "contains": ["Vail Region, CO", "#32", "3.6%"],
    },
    {
        "prompt": "What rank is Utah by total athletes?",
        "tool": "query_data",
        "contains": ["Utah", "#27", "52"],
    },
    {
        "prompt": "Compare California and Colorado",
        "tool": "query_data",
        "contains": ["California", "Colorado", "788", "104", "#16"],
    },
    {
        "prompt": "Which hubs are strongest for skiing?",
        "tool": "query_data",
        "contains": ["ski", "Salt Lake City", "Vail"],
    },
    {
        "prompt": "Show Mountain West hubs",
        "tool": "query_data",
        "contains": ["Mountain West", "Salt Lake City", "Vail"],
    },
]


def post_chat(base: str, prompt: str) -> dict:
    body = json.dumps({"message": prompt, "history": []}).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def check_case(base: str, case: dict) -> bool:
    data = post_chat(base, case["prompt"])
    text = data.get("text", "")
    tool_calls = data.get("tool_calls", [])
    tools = [call.get("name") for call in tool_calls]
    ok = True

    if case["tool"] not in tools:
        ok = False

    expected_args = case.get("tool_args") or {}
    for key, value in expected_args.items():
        if not any(call.get("args", {}).get(key) == value for call in tool_calls):
            ok = False

    for needle in case["contains"]:
        if needle.lower() not in text.lower():
            ok = False

    print(f"{case['prompt']}: {'OK' if ok else 'FAIL'}")
    print(f"  tools: {tool_calls}")
    print(f"  text : {text}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()

    ok = True
    for case in CASES:
        try:
            ok = check_case(args.base, case) and ok
        except Exception as exc:
            ok = False
            print(f"{case['prompt']}: ERROR {exc}")

    if ok:
        print("Chat smoke OK.")
        return 0

    print("Chat smoke failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
