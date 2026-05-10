import hashlib
import json
import urllib.parse
import urllib.request


LOCAL = "http://127.0.0.1:8080"
LIVE = "https://hometown-success-engine-74530725032.us-central1.run.app"
PATHS = ["/health", "/hubs", "/athletes", "/states/aggregate"]


def fetch_json(base: str, path: str):
    url = base + path
    if urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme for smoke test: {url}")
    with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def digest(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare_path(path: str) -> bool:
    local = fetch_json(LOCAL, path)
    live = fetch_json(LIVE, path)
    ok = digest(local) == digest(live)
    print(f"{path}: {'OK' if ok else 'MISMATCH'}")
    if isinstance(local, list):
        print(f"  lengths local/live: {len(local)} / {len(live)}")
    else:
        print(f"  local: {local}")
        print(f"  live : {live}")
    return ok


def main() -> int:
    ok = True
    for path in PATHS:
        ok = compare_path(path) and ok

    hubs = fetch_json(LOCAL, "/hubs")
    for hub in hubs:
        path = f"/hubs/{hub['hub_id']}/narrative"
        local = fetch_json(LOCAL, path)
        live = fetch_json(LIVE, path)
        if digest(local) != digest(live):
            print(f"{path}: MISMATCH")
            ok = False

    if ok:
        print("Local/live parity OK.")
        return 0

    print("Local/live parity failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
