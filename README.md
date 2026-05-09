# Hometown Success Engine

> An interactive map of America's hometown athletic hubs, built for the Google and Team USA Vibe Code for Gold Challenge 2.

**Live:** [hometown-success-engine-11a06.web.app](https://hometown-success-engine-11a06.web.app/)

**Demo Video:** Coming soon

**Built for:** [Vibe Code for Gold with Google and Team USA Hackathon, Challenge 2](https://vibecodeforgoldwithgoogle.devpost.com/)

## What It Is

The Hometown Success Engine maps **5,119 Olympians and Paralympians** from Tokyo 2020 through Milan-Cortina 2026 across **40 hometown hubs** in the United States and territories. The project focuses on where Team USA athletes come from, using hometown-region counts rather than medal counts so the experience is inclusive of all mapped athletes.

Four hubs are flagged as **Paralympic Hot Spots**, meaning their Paralympic share runs more than 2x the current **4.7% national baseline**:

| Hub | Paralympic Share | Athletes |
|---|---:|---:|
| Anchorage Region, AK | 13.3% | 30 |
| Phoenix Region, AZ | 11.7% | 60 |
| Tampa Region, FL | 11.4% | 35 |
| Lincoln Region, NE | 10.4% | 77 |

The map is interactive: users can click a hub, inspect regional narratives, view climate and geographic context, filter Paralympic Hot Spots, explore state summaries, and ask Gemini to drive the map directly.

## How It Works

```text
Vite + TypeScript Web Component
  Google Maps vector map
  deck.gl state, hub, and athlete layers
  Ask Gemini chat panel
        |
        v
FastAPI on Cloud Run
  /hubs
  /athletes
  /states/aggregate
  /chat
        |
        v
Vertex AI Gemini 2.5 Flash
  Function Calling
  grounded tool-result narration
```

**Data pipeline:** Ingest public roster facts, normalize hometown data, geocode places, group hometowns with HDBSCAN, compute hub composition, flag Paralympic Hot Spots, fetch climate context, and generate hub narratives.

**Backend:** Serves hub data, narratives, athlete geo points, state aggregates, and the Gemini interaction layer from FastAPI on Cloud Run.

**Frontend:** A vanilla TypeScript Web Component renders the Google Maps + deck.gl experience and dispatches Gemini tool calls into map actions.

## Gemini Interaction Layer

The chat panel is not a chatbot bolted onto a map. It is an analyst and navigation layer that uses Gemini Function Calling to connect plain-language questions to the live data model.

Gemini receives:

1. A system instruction with the current 5,119-athlete, 40-hub, 4-Hot-Spot dataset summary.
2. The current hub lookup table, Hot Spot list, state codes, ranking rules, and allowed tools.
3. The conversation history and latest user message.

Gemini can call these tools:

| Tool | Purpose |
|---|---|
| `select_hub(hub_id)` | Select a hub and open its narrative card |
| `zoom_to_hub(hub_id)` | Move the map camera to a hub |
| `filter_to_paralympic(macro_region?)` | Highlight Paralympic Hot Spots |
| `select_state(state_code)` | Open the state summary panel |
| `reset_view()` | Clear filters and return to the national map |
| `query_data(...)` | Answer rankings, comparisons, profiles, totals, sport, region, and Hot Spot questions |

The data tool supports:

- Summary questions: "How many athletes and hubs?"
- Hub rankings: "Rank hubs by Paralympic share."
- State rankings: "What rank is Utah by total athletes?"
- Profiles: "Tell me about Vail."
- Comparisons: "Compare California and Colorado."
- Sport questions: "Which hubs are strongest for skiing?"
- Regional questions: "Show Mountain West hubs."

Tool results are generated from runtime data before Gemini narrates them, so answers include grounded counts, ranks, percentages, top sports, hot spot status, climate, and geographic context. The system avoids individual athlete names and avoids any claim that geography guarantees outcomes, using conditional phrasing such as "could help find," "may foster," and "is associated with."

## Tech Stack

**AI and Cloud**

- Vertex AI Gemini 2.5 Flash
- Gemini Function Calling
- Cloud Run
- Firebase Hosting

**Frontend**

- TypeScript and Vite
- Vanilla Web Components
- Google Maps JavaScript API
- deck.gl
- d3-geo

**Backend and Pipeline**

- Python 3.12
- FastAPI and Pydantic
- google-genai SDK
- HDBSCAN-based hometown grouping
- Open-Meteo climate normals

## Run Locally

### Backend

```powershell
cd C:\Users\BigRooster\Documents\Python\Projects\Hometown-Success-Engine
.\pipeline\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
```

### Frontend

```powershell
cd C:\Users\BigRooster\Documents\Python\Projects\Hometown-Success-Engine\frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Verification

```powershell
python pipeline\parity_smoke.py
python pipeline\chat_smoke.py
cd frontend
npm run build
```

Expected public values:

- 5,119 mapped Olympians and Paralympians
- 40 hometown hubs
- 4 Paralympic Hot Spots
- 4.7% national Paralympic share

## Deploy

### Backend

```powershell
cd C:\Users\BigRooster\Documents\Python\Projects\Hometown-Success-Engine
gcloud run deploy hometown-success-engine --source . --region us-central1 --clear-base-image
```

### Frontend

```powershell
cd C:\Users\BigRooster\Documents\Python\Projects\Hometown-Success-Engine\frontend
npm run build
firebase deploy --only hosting
```

## Compliance

- No athlete names, images, or likenesses appear in the public UI.
- Public responses use aggregate hometown-region data.
- Olympic and Paralympic athletes are both represented throughout the product.
- Geography is framed conditionally. The tool identifies places that could help find or may foster Team USA talent; it does not claim that geography produces athletes.
- Raw athlete-level data is excluded from git for NIL safety.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built solo by [Strainz / A.J. Lawrence](https://www.linkedin.com/in/alexanderjonlawrence/).
