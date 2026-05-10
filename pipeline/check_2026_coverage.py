"""Quick coverage check before extending ingest_wikidata.py.

Asks Wikidata: how many US athletes from Milan-Cortina 2026, Milan-Cortina
2026 Paralympics, and (as a baseline) Paris 2024 currently have hometown
data we could ingest? Decision rule: if Milan 2026 has 100+ entries with
US birthplace, worth extending the pipeline. If under 30, skip and ship.
"""
import urllib.request
import urllib.parse
import json
import time

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "HometownSuccessEngine/1.0 (contact: github.com/strainzz)",
    "Accept": "application/sparql-results+json",
}

QUERIES = {
    "Milan-Cortina 2026 (US, US birthplace)": (
        "wdt:P1344 wd:Q108869916 ; wdt:P19 ?bp . ?bp wdt:P17 wd:Q30 ."
    ),
    "Milan-Cortina 2026 (US, any birthplace)": (
        "wdt:P1344 wd:Q108869916 ."
    ),
    "Milan-Cortina 2026 Paralympics (US)": (
        "wdt:P1344 wd:Q60803304 ."
    ),
    "Paris 2024 (US, US birthplace) baseline": (
        "wdt:P1344 wd:Q14512673 ; wdt:P19 ?bp . ?bp wdt:P17 wd:Q30 ."
    ),
    "Paris 2024 Paralympics (US)": (
        "wdt:P1344 wd:Q105768036 ."
    ),
}

QUERY_TEMPLATE = """
SELECT (COUNT(DISTINCT ?athlete) AS ?count) WHERE {{
  ?athlete wdt:P31 wd:Q5 ;
           wdt:P1532 wd:Q30 ;
           {clause}
}}
"""

for label, clause in QUERIES.items():
    q = QUERY_TEMPLATE.format(clause=clause).strip()
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'query': q})}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        # Fixed HTTPS Wikidata endpoint.
        with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310
            data = json.loads(r.read())
        b = data["results"]["bindings"]
        count = b[0]["count"]["value"] if b else "0"
        print(f"  {label:50s} -> {count:>5}")
    except Exception as e:
        print(f"  {label:50s} -> ERR: {e}")
    time.sleep(1)
