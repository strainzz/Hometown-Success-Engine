# Hometown Success Engine

> A Gemini Live voice-powered Team USA hometown intelligence engine for the Google Cloud x Team USA Challenge.

**Live app:** [hometown-success-engine-11a06.web.app](https://hometown-success-engine-11a06.web.app/)

**Cloud Run health:** [hometown-success-engine-74530725032.us-central1.run.app/health](https://hometown-success-engine-74530725032.us-central1.run.app/health)

**Demo video:** [YouTube demo](https://youtu.be/t2DXQcB7jo8)

**Built for:** [Google Cloud x Team USA Challenge, Vibe Code for Gold Challenge 2](https://vibecodeforgoldwithgoogle.devpost.com/)

## Overview

The Hometown Success Engine maps **5,119 Olympians and Paralympians** from Tokyo 2020 through Milan-Cortina 2026 across **40 Team USA hometown hubs** in the continental U.S., Alaska, Hawaii, Washington, D.C., and Puerto Rico.

The defining feature is **Gemini Live voice interaction**. Users can speak naturally to Gemini and have it control the map: select hubs, open states, highlight Paralympic Hot Spots, answer rankings, compare places, explain the dots and layers, focus hometowns, and reset the view. Gemini voice and Gemini chat share the same full engine context, tool schema, ranking rules, sport aliases, hometown lookup, Hot Spot threshold, climate context, and safety rules.

Instead of ranking places by medals, the engine focuses on where mapped athletes are from. That makes the tool more inclusive of all Olympians and Paralympians while giving analysts, judges, and Team USA stakeholders a way to explore how hometown geography, sport mix, climate, and regional context are associated with athlete development.

The product does not claim that geography guarantees outcomes or produces athletes. It uses conditional language throughout: regions **could help find**, **may foster**, **may explain**, or **are associated with** patterns in the mapped hometown data.

## Why It Matters

Challenge 2 asks for a tool that identifies hometown "Hubs" by correlating geography with the sports Team USA is present in, using the number of Olympians and Paralympians from hometown regions instead of medal counts.

This project answers that brief with:

- Gemini Live native voice as the lead interaction, letting spoken questions move the map and return grounded audio responses.
- Gemini chat using the same full tool schema and context for precise typed analysis.
- A national hometown hub map built around aggregate athlete counts, not medals.
- Paralympic Hot Spot detection using a clear, deterministic threshold.
- Sport, state, region, and hometown analysis connected directly to the map.
- Conditional, association-based language that avoids implying geography guarantees athletic success.

## Gemini Live Voice Integration

Gemini Live voice is the signature interaction in the project. It turns the map from a static dashboard into a conversational exploration tool with direct access to the engine's data and controls.

With voice enabled, a user can ask:

- "Show the top Paralympic Hot Spot."
- "Tell me about Vail."
- "What do the dots mean?"
- "How many athletes are from Boise, Idaho?"
- "What state has the highest Paralympic share?"
- "Reset the map and then show Arizona."

The voice pipeline streams microphone audio to Cloud Run, routes the request through Gemini Live, calls deterministic map and data tools, moves the map immediately, and returns a compact grounded spoken response. The chat panel also shows a **Map readout** card so exact tool facts stay visible while Gemini speaks.

This makes Gemini more than narration. It is an interactive control layer for the engine.

## Public Dataset Snapshot

| Metric | Value |
|---|---:|
| Mapped Olympians and Paralympians | 5,119 |
| Hometown hubs | 40 |
| In-scope state regions | 52 |
| Paralympic Hot Spots | 10 |
| National Paralympic baseline | 4.7% |
| Hot Spot threshold | 7.5% |
| Public data range | Tokyo 2020 through Milan-Cortina 2026 |

## Paralympic Hot Spots

A **Paralympic Hot Spot** is any hometown hub where the Paralympic share is **7.5% or higher**. The current national Paralympic baseline is **4.7%**.

| Hub | Paralympic Share | Athletes |
|---|---:|---:|
| Anchorage Region, AK | 13.3% | 30 |
| Phoenix Region, AZ | 11.7% | 60 |
| Tampa Region, FL | 11.4% | 35 |
| Lincoln Region, NE | 10.4% | 77 |
| Stillwater Region, OK | 9.2% | 87 |
| Merced Region, CA | 8.9% | 45 |
| Cleveland Region, OH | 8.1% | 62 |
| Charlotte Region, NC | 7.9% | 101 |
| Portland Region, OR | 7.7% | 65 |
| Allegan Region, MI | 7.7% | 39 |

## Core Experience

- Speak to Gemini Live voice and have it control the map with full engine context.
- Use Gemini chat for the same grounded controls in text form.
- Explore 40 hometown hubs on a Google Maps and deck.gl interface.
- Toggle Paralympic Hot Spots and compare them against the 4.7% national baseline.
- Inspect hub profiles with athlete totals, Olympic and Paralympic split, ranks, top sports, climate, and geographic context.
- Click states to view aggregate counts, ranks, Paralympic share, and top hub context.
- Ask exact hometown questions such as "How many athletes are from Boise, Idaho?"
- Ask ranking questions such as "What state has the highest Paralympic share?" or "Show me the number 24 ranked state."
- Ask sport questions such as "Which hubs are strongest for skiing?" or "Which states are strongest for winter sports?"
- Ask follow-up questions while Gemini keeps conversational context across hubs, states, hometowns, rankings, and map actions.

## Example Prompts

These prompts show Gemini acting as an interactive map and data guide:

- "What do the dots mean?"
- "Why does this matter for Team USA?"
- "Show the top Paralympic Hot Spot."
- "Show places above the national baseline."
- "Tell me about Vail."
- "How many athletes are from Boise, Idaho?"
- "What state has the highest Paralympic share?"
- "Show me the number 24 ranked state."
- "Which hubs are strongest for skiing?"
- "Compare California and Colorado."
- "Reset the map and then show Arizona."

## How It Works

```text
Firebase Hosting
  Vite + TypeScript Web Component
  Google Maps vector map
  deck.gl state, hub, and hometown point layers
  Gemini text, map control, and voice panel
        |
        v
FastAPI on Cloud Run
  /hubs
  /athletes
  /states/aggregate
  /chat
  /voice/ws
        |
        v
Vertex AI Gemini
  Gemini 2.5 Flash Function Calling
  Gemini Live native audio
  grounded tool-result narration
```

**Data pipeline:** Ingest public roster facts, normalize hometown data, geocode places, group hometowns into hubs, compute hub and state aggregates, flag Paralympic Hot Spots, fetch climate context, and generate hub narratives.

**Backend:** FastAPI on Cloud Run serves hub data, public athlete map points, aggregate hometown data, state summaries, profiles, chat, and Gemini Live voice.

**Frontend:** A Vite and TypeScript Web Component renders the Google Maps and deck.gl interface, then dispatches Gemini tool calls into map actions.

## Gemini Interaction Layer

Gemini is the technical center of the product. It is wired as the interaction layer for the map, not a separate chatbot. Gemini Live voice is the signature experience, and typed chat uses the same schema, deterministic routing, result builder, session context, and safety rules.

Gemini receives:

1. Current dataset constants: 5,119 mapped athletes, 40 hubs, 10 Hot Spots, 4.7% baseline, 7.5% threshold, and 52 in-scope state regions.
2. Hub, state, sport, ranking, Hot Spot, and hometown lookup context.
3. Allowed tool definitions for map movement, data queries, explanations, comparisons, and hometown focus.
4. Recent session context so follow-up questions can refer to the last hub, state, hometown, ranking, or map action.

Gemini can call these tools:

| Tool | Purpose |
|---|---|
| `select_hub(hub_id)` | Select a hub and open its profile |
| `zoom_to_hub(hub_id)` | Move the map camera to a hub |
| `filter_to_paralympic(macro_region?)` | Highlight Paralympic Hot Spots |
| `select_state(state_code)` | Open the state summary panel |
| `reset_view()` | Clear filters and return to the national map |
| `explain_map(topic?)` | Explain dots, circles, colors, state shading, Hot Spots, and insets |
| `explain_engine(topic)` | Explain methodology, sources, challenge fit, data scope, baseline, threshold, and conditional language |
| `focus_hometown(hometown, state_code?)` | Focus an exact hometown aggregate without exposing athlete names |
| `highlight_hubs(hub_ids, label, reason)` | Highlight multiple hubs for list-style map questions |
| `query_data(...)` | Answer rankings, comparisons, profiles, totals, sport, region, and Hot Spot questions |

Supported interaction types include:

- Dataset summaries
- Hub and state rankings
- Exact rank lookups
- Middle-of-ranking requests
- Hub, state, and hometown profiles
- Region filters
- Sport group rankings
- Hub and state comparisons
- Map literacy explanations
- Project, methodology, source, scope, and challenge-fit explanations

Tool results are computed from runtime data before Gemini explains them. This keeps responses grounded in exact counts, ranks, percentages, top sports, Hot Spot status, climate, and geographic context while still letting users interact conversationally.

Voice mode uses Gemini Live native audio through `/voice/ws`. The frontend streams microphone audio to Cloud Run, the backend handles Gemini Live function calls, dispatches map tools, and returns compact grounded spoken responses. The UI shows **Map readout** cards so deterministic data remains visually separate from Gemini's spoken response.

## Tech Stack

**AI and Cloud**

- Vertex AI Gemini 2.5 Flash
- Gemini Live native audio
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
- HDBSCAN-style hometown hub grouping
- Open-Meteo climate context

## Prerequisites

- Python 3.12
- Node.js 20 or newer
- Google Cloud CLI
- Firebase CLI
- A Google Cloud project with Vertex AI, Cloud Run, Firebase Hosting, and Maps JavaScript API enabled
- Local Google Cloud credentials through `gcloud auth application-default login` or a service account path in `GOOGLE_APPLICATION_CREDENTIALS`

## Run Locally

Clone the repo and create a local Python environment:

```powershell
git clone https://github.com/strainzz/Hometown-Success-Engine.git
cd Hometown-Success-Engine
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create local environment variables using your own Google Cloud project, Maps API key, and map ID. See `.env.example` for expected names.

Do not commit local `.env` files, service account files, API keys, generated credentials, or raw athlete-level data.

### Backend

```powershell
$env:GOOGLE_CLOUD_PROJECT="your-google-cloud-project-id"
$env:GEMINI_LIVE_LOCATION="us-central1"
$env:GEMINI_LIVE_MODEL="gemini-live-2.5-flash-native-audio"
$env:GEMINI_VOICE_NAME="Kore"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
```

The backend should be available at [http://127.0.0.1:8080](http://127.0.0.1:8080).

### Frontend

```powershell
cd frontend
@'
VITE_API_BASE_URL=http://127.0.0.1:8080
VITE_GOOGLE_MAPS_API_KEY=your-google-maps-api-key
VITE_GOOGLE_MAPS_MAP_ID=your-google-maps-map-id
'@ | Set-Content -Encoding utf8 .env.local
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Verification

Local checks:

```powershell
python -m py_compile backend\main.py pipeline\chat_smoke.py pipeline\voice_ws_smoke.py pipeline\state_constellation_smoke.py
python pipeline\parity_smoke.py
python pipeline\chat_smoke.py --base http://127.0.0.1:8080
python pipeline\voice_ws_smoke.py --base http://127.0.0.1:8080
python pipeline\state_constellation_smoke.py --base http://127.0.0.1:8080
cd frontend
npm run build
```

Live checks:

```powershell
python pipeline\parity_smoke.py
python pipeline\chat_smoke.py --base https://hometown-success-engine-74530725032.us-central1.run.app
python pipeline\voice_ws_smoke.py --base https://hometown-success-engine-74530725032.us-central1.run.app
python pipeline\state_constellation_smoke.py --base https://hometown-success-engine-74530725032.us-central1.run.app
```

Expected public values:

- 5,119 mapped Olympians and Paralympians
- 40 Team USA hometown hubs
- 10 Paralympic Hot Spots
- 4.7% national Paralympic baseline
- 7.5% Paralympic Hot Spot threshold
- 52 in-scope state regions

## Deploy

### Backend

```powershell
gcloud run deploy YOUR_CLOUD_RUN_SERVICE `
  --source . `
  --region YOUR_REGION `
  --project YOUR_GOOGLE_CLOUD_PROJECT_ID `
  --clear-base-image
```

### Frontend

```powershell
cd frontend
npm run build
firebase deploy --only hosting --project YOUR_FIREBASE_PROJECT_ID
```

## Security and Compliance

- No athlete names, images, or likenesses appear in the public UI.
- Gemini responses use aggregate counts only and do not expose athlete names.
- Runtime deployment uses sanitized public data files: anonymous in-scope athlete map points and aggregate hometown counts with no athlete names or birth dates.
- Olympic and Paralympic athletes are represented throughout the product.
- Geography is framed conditionally. The tool identifies places that could help find or may foster Team USA talent; it does not claim that geography produces athletes.
- Public map scope is limited to the continental U.S., Alaska, Hawaii, Washington, D.C., and Puerto Rico.
- Raw athlete-level data is excluded from git, Docker builds, and Cloud Run source uploads.
- Local credential files, API keys, service account files, generated credentials, Firebase local artifacts, and development logs are ignored.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Built solo by [Strainz / A.J. Lawrence](https://www.linkedin.com/in/alexanderjonlawrence/).
