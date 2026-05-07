import json
from pathlib import Path

NARRATIVES_PATH = Path("pipeline/narratives/hubs.json")

REPLACEMENTS = [
    ("Para athletes", "Paralympians"),
    ("para athletes", "Paralympians"),
    ("Para-athletes", "Paralympians"),
    ("para-athletes", "Paralympians"),
    ("Para athlete", "Paralympian"),
    ("para athlete", "Paralympian"),
    ("Para-athlete", "Paralympian"),
    ("para-athlete", "Paralympian"),
    ("athletes with a disability", "Paralympic athletes"),
    ("athlete with a disability", "Paralympic athlete"),
]


def main() -> None:
    if not NARRATIVES_PATH.exists():
        print(f"ERROR: {NARRATIVES_PATH} not found")
        return

    with NARRATIVES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw = json.dumps(data, ensure_ascii=False)

    counts = {}
    for old, new in REPLACEMENTS:
        before = raw.count(old)
        if before > 0:
            raw = raw.replace(old, new)
            counts[old] = (before, new)

    new_data = json.loads(raw)

    with NARRATIVES_PATH.open("w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    if not counts:
        print(f"No replacements needed in {NARRATIVES_PATH}")
        return

    print(f"Replacements made in {NARRATIVES_PATH}:")
    total = 0
    for old, (count, new) in counts.items():
        print(f"  {count:4}x  '{old}'  ->  '{new}'")
        total += count
    print(f"\nTotal: {total} replacements")


if __name__ == "__main__":
    main()