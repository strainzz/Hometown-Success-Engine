# Hometown Success Engine

> An interactive map of America's hometown athletic hubs. Discover the 5 regions where Paralympic representation runs more than 2x the national rate.

🌐 **Live:** [hometown-success-engine-11a06.web.app](https://hometown-success-engine-11a06.web.app/)

📺 **Demo Video:** [Coming Soon]

🏆 **Built for:** [Vibe Code for Gold with Google × Team USA Hackathon, Challenge 2](https://vibecodeforgoldwithgoogle.devpost.com/)

---

## What it is

The Hometown Success Engine maps the hometowns of **5,012 Olympians and Paralympians** from the 2020 to 2024 cycle into **37 regional hubs** across the United States. Each hub represents a real geographic cluster of where elite athletes actually grew up, surfaced by HDBSCAN density clustering, not by state lines.

Five of those hubs are flagged as **Paralympic Hot Spots**, regions where the Paralympic share runs at more than twice the **4.6% national baseline**:

| Hub | Paralympic Share | Athletes |
|---|---|---|
| Phoenix Region, AZ | 12.7% | 55 |
| Anchorage Region, AK | 12.5% | 24 |
| Lincoln Region, NE | 10.8% | 74 |
| Stillwater Region, OK | 9.3% | 86 |
| Merced Region, CA | 9.3% | 43 |

The map is fully interactive. Click any hub for its narrative card. Filter by region. You can also ask the Gemini-powered chat assistant for the region or hub you want. The assistant uses Gemini 2.5 Flash with Function Calling to translate your request into a map action..

## How it works

             ┌─────────────────────────┐
             │  Vite + TypeScript SPA  │
             │  (Web Component widget) │
             │                         │
             │  Google Maps Vector     │
             │  + deck.gl overlay      │
             │  + Ask Gemini chat      │
             └────────────┬────────────┘
                          │ HTTPS
                          ▼
             ┌─────────────────────────┐
             │  FastAPI on Cloud Run   │
             │                         │
             │  /hubs                  │
             │  /athletes              │
             │  /states/aggregate      │
             │  /chat (Gemini Tool Use)│
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │  Vertex AI              │
             │  Gemini 2.5 Flash       │
             │  + Function Calling     │
             └─────────────────────────┘

### Three layers, three jobs

**Data pipeline** (`pipeline/`): Geocode publicly-listed Team USA hometowns. Run HDBSCAN on the coordinates. Compute per-hub composition, top sports, and Paralympic Hot Spot flags. Output: `pipeline/clustered/hubs.json`.

**Backend** (`backend/`): FastAPI service deployed on Cloud Run. Serves hub data, narratives, athlete geo points, and state aggregates. The `/chat` endpoint wraps Gemini 2.5 Flash with four tools: `select_hub`, `zoom_to_hub`, `filter_to_paralympic`, `reset_view`. Two-pass Tool Use pattern means Gemini can both call a tool AND narrate the result with real data.

**Frontend** (`frontend/`): A vanilla TypeScript Web Component. Map uses Google Maps Vector tiles with a Map ID, deck.gl GoogleMapsOverlay, and three layers: state choropleth, athlete constellation (5,012 dots), and hub centroids (37 dots). Served from Firebase Hosting.

## Tech stack

**AI / Cloud:**
- Vertex AI Gemini 2.5 Flash
- Vertex AI Function Calling (Tool Use)
- Cloud Run (FastAPI backend)
- Cloud Build (image builds)
- Firebase Hosting (frontend)

**Frontend:**
- TypeScript, Vite 6
- Vanilla Web Components (no React)
- deck.gl (ScatterplotLayer, GeoJsonLayer)
- @googlemaps/js-api-loader
- d3-geo (territory inset projections)

**Backend:**
- Python 3.12
- FastAPI + Pydantic + uvicorn
- google-genai SDK (Vertex AI)
- HDBSCAN (clustering)

## Run locally

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

gcloud auth application-default login
$env:GOOGLE_CLOUD_PROJECT="your-gcp-project"

python -m uvicorn main:app --host 127.0.0.1 --port 8080 --reload
```

### Frontend

```bash
cd frontend
npm install

# Create .env.local with:
#   VITE_API_BASE_URL=http://127.0.0.1:8080
#   VITE_GOOGLE_MAPS_API_KEY=your-key
#   VITE_GOOGLE_MAPS_MAP_ID=your-map-id

npm run dev
# → http://localhost:5173
```

## Deploy

### Backend → Cloud Run

```bash
cd backend
gcloud run deploy hometown-success-engine \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --clear-base-image
```

### Frontend → Firebase Hosting

```bash
cd frontend
npm run build
cd ..
firebase deploy --only hosting
```

## Compliance & content rules

This project follows the contest's content restrictions:

- **No NIL violations.** No athlete names, images, or likenesses appear in the UI or in committed data files. Raw geocoded athlete data is gitignored.
- **Olympic and Paralympic parity.** Both groups are mapped, surfaced, and discussed throughout. The Paralympic Hot Spot framing is the headline insight.
- **Conditional phrasing.** Geography does not guarantee outcomes. The map could help find regions where the ingredients are present.
- **No banned terminology.** No "Paris 2024", no "LA28", no "former Olympian/Paralympian" anywhere.

## Data

Athlete roster data sourced from publicly available Team USA listings, 2020 to 2024 cycle. Hometowns geocoded with the Google Maps Geocoding API. Aggregated to hub level via HDBSCAN. Only hub-level aggregates are committed to the repo (see `pipeline/clustered/hubs.json`); raw athlete-level data is gitignored for NIL safety.

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

Built solo by [Strainz / A.J. Lawrence](https://www.linkedin.com/in/alexanderjonlawrence/). One person, six weeks, soup to nuts.   