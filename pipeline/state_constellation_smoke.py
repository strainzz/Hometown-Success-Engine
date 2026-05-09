import argparse
import json
import sys
import urllib.request
from collections import defaultdict


DEFAULT_BASE = "http://127.0.0.1:8080"


def fetch_json(base: str, path: str):
    url = f"{base.rstrip('/')}{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run(base: str) -> bool:
    athletes = fetch_json(base, "/athletes")
    aggregates = fetch_json(base, "/states/aggregate")

    grouped = defaultdict(lambda: {"olympic": 0, "paralympic": 0, "both": 0})
    for athlete in athletes:
        state = athlete.get("state")
        status = athlete.get("status")
        if state and state != "XX" and status in grouped[state]:
            grouped[state][status] += 1

    ok = True
    for aggregate in aggregates:
        state = aggregate["state"]
        counts = grouped[state]
        expected_total = counts["olympic"] + counts["paralympic"] + counts["both"]
        expected_para = counts["paralympic"] + counts["both"]
        actual_para = aggregate["paralympic_count"] + aggregate["both_count"]
        if aggregate["total_athletes"] != expected_total or actual_para != expected_para:
            ok = False
            print(
                f"{state}: FAIL aggregate total/para "
                f"{aggregate['total_athletes']}/{actual_para} != dots {expected_total}/{expected_para}"
            )

    aggregate_states = {a["state"] for a in aggregates}
    extra_states = sorted(set(grouped) - aggregate_states - {"XX"})
    if extra_states:
        ok = False
        print(f"States present in dots but missing aggregate: {extra_states}")

    id_counts = grouped["ID"]
    id_para = id_counts["paralympic"] + id_counts["both"]
    print(
        "Idaho constellation:",
        f"{id_counts['olympic'] + id_counts['paralympic'] + id_counts['both']} total,",
        f"{id_para} Paralympic dots",
    )

    print(f"State constellation smoke: {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()
    return 0 if run(args.base) else 1


if __name__ == "__main__":
    sys.exit(main())
