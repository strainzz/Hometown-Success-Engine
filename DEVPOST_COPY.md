# Devpost Copy Draft

## Project Title

Hometown Success Engine

## Tagline

A Gemini-powered Team USA hometown intelligence engine for exploring Olympic and Paralympic athlete geography.

## Short Description

The Hometown Success Engine maps 5,119 Olympians and Paralympians from Tokyo 2020 through Milan-Cortina 2026 across 40 hometown hubs. Gemini chat and Gemini Live voice are the core interaction layer: they understand the engine's map schema, data scope, rankings, hometown lookup, sport groups, Hot Spot rules, and map controls so users can ask natural questions and have Gemini move the map, explain the data, compare places, and answer grounded analysis questions.

## Inspiration

Challenge 2 asks for a tool that identifies hometown hubs by correlating geography with the sports Team USA is present in. I wanted to build something that felt like an analyst tool, not a medal table.

The core idea is simple: where are Olympians and Paralympians from, and what regional patterns can we see when we look at hometowns instead of podiums? That framing keeps the experience inclusive of all mapped athletes and makes Paralympic representation visible in the same interface as Olympic representation.

## What It Does

The Hometown Success Engine is an interactive national map of Team USA hometown hubs.

Users can:

- Explore 5,119 mapped Olympians and Paralympians from Tokyo 2020 through Milan-Cortina 2026.
- Inspect 40 hometown hubs with total athletes, Olympic and Paralympic split, ranks, top sports, climate, and geographic context.
- Identify 10 Paralympic Hot Spots, defined as hubs with Paralympic share at or above 7.5%.
- Compare each hub against the 4.7% national Paralympic baseline.
- Click states to view aggregate counts, rankings, Paralympic share, and top hub context.
- Ask Gemini chat or Gemini Live voice questions that can move the map, select hubs, open states, highlight Hot Spots, explain the data, answer ranking questions, compare places, and focus exact hometowns.

Example prompts:

- "What do the dots mean?"
- "Show the top Paralympic Hot Spot."
- "Tell me about Vail."
- "How many athletes are from Boise, Idaho?"
- "Which states are strongest for skiing?"
- "What state has the highest Paralympic share?"
- "Reset the map and then show Arizona."

## How It Uses Google Cloud And Gemini

The project uses Google Cloud as the runtime layer and Gemini as the interactive intelligence layer. Gemini is the most technically elevated part of the project because it turns natural language into precise map and data actions while staying grounded in deterministic backend results.

- **Cloud Run:** Hosts the FastAPI backend, data endpoints, Gemini tool routing, and Gemini Live WebSocket proxy.
- **Firebase Hosting:** Hosts the Vite and TypeScript frontend.
- **Google Maps JavaScript API:** Powers the base map and map camera.
- **Vertex AI Gemini 2.5 Flash:** Handles function-calling chat for map and data questions.
- **Gemini Live native audio:** Lets users ask spoken questions that move the map and receive grounded spoken responses.

Gemini is connected to deterministic tools such as `select_hub`, `select_state`, `filter_to_paralympic`, `focus_hometown`, `highlight_hubs`, `explain_map`, `explain_engine`, and `query_data`. Both chat and voice share this schema, session context, ranking logic, safety rules, and map controls. Tool results are computed from runtime data first, then Gemini explains them. This keeps answers grounded in exact counts, ranks, percentages, top sports, Hot Spot status, climate, and geographic context.

## How It Was Built

The pipeline ingests public roster facts, normalizes hometown data, geocodes places, groups nearby hometowns into regional hubs, computes hub and state aggregates, flags Paralympic Hot Spots, fetches climate context, and generates hub narratives.

The frontend is a TypeScript Web Component using Google Maps and deck.gl layers for:

- State shading
- Hometown point constellations
- Hub circles
- Paralympic Hot Spot styling
- State and hub detail panels
- Gemini chat and voice controls

The backend is FastAPI on Cloud Run. It serves:

- Hub data
- Public athlete map points
- Aggregate hometown lookup
- State summaries
- Gemini chat
- Gemini Live voice WebSocket

## Data Scope And Safety

The public app uses aggregate data only. It does not show athlete names, images, likenesses, birth dates, or individual profiles.

The public map scope is the continental U.S., Alaska, Hawaii, Washington, D.C., and Puerto Rico. Out-of-scope territories are gracefully declined in Gemini responses instead of being ranked or zoomed.

The language is intentionally conditional. The engine does not claim geography produces athletes or guarantees outcomes. It shows associations in mapped hometown data that could help guide better scouting, outreach, youth-program, and analyst questions.

## Challenges

The hardest part was making the Gemini layer reliable enough for judge testing. A natural language question like "What state has the highest Paralympic share?" cannot be treated like a generic chat response. It has to select the correct metric, apply the correct ranking universe, move the map to the winning state, and explain the result without inventing anything.

The final interaction layer uses deterministic routing, shared typed-chat and voice tools, strict ranking rules, in-scope geography rules, and smoke tests for natural-language evaluation prompts.

Another challenge was keeping the constellation dots, state panels, hub panels, and Gemini answers aligned. The project now uses sanitized public runtime data and smoke checks so local and live data stay in parity.

## Accomplishments

- Built an end-to-end Google Cloud hosted map experience.
- Mapped 5,119 Olympians and Paralympians across 40 hometown hubs.
- Added deterministic Paralympic Hot Spot detection with a 7.5% threshold.
- Built Gemini chat and native voice interactions that can control the map with shared context and deterministic data grounding.
- Added exact ranking, middle ranking, sport ranking, hometown lookup, comparison, and map-literacy support.
- Hardened public deployment data so aggregate map interaction works without exposing athlete names or sensitive local files.
- Added smoke tests for local/live parity, Gemini chat, Gemini voice, and state constellation accuracy.

## What I Learned

The biggest lesson was that AI interaction works best here when Gemini is grounded by deterministic tools rather than asked to reason from prose alone. Gemini is strongest when it can translate user intent into structured map and data actions, while the backend computes the actual facts.

I also learned that voice interaction needs a different presentation style than typed chat. Typed chat can be analytical and detailed. Voice needs to be concise, map-first, and readable in a live demo.

## What's Next

- Add more sport-specific analytics and timeline views.
- Add analyst export workflows for hub and state summaries.
- Add stronger accessibility controls for map layers and color modes.
- Expand the data pipeline as new public roster and hometown facts become available.
- Continue improving Gemini Live conversational follow-ups and direct map narration.

## Built With

- Google Cloud Run
- Firebase Hosting
- Vertex AI Gemini 2.5 Flash
- Gemini Live native audio
- Google Maps JavaScript API
- deck.gl
- TypeScript
- Vite
- Python
- FastAPI
- Pydantic
- google-genai SDK
- Open-Meteo climate context

## Screenshot Shot List

Use 5 to 7 screenshots for Devpost. Recommended order:

1. **Hero map:** Full app showing the national map, header, stat cards, and Gemini panel.
2. **Paralympic Hot Spots:** Hot Spot filter active, red hubs visible, stat card showing 10 Hot Spots.
3. **Hub profile:** Vail or Salt Lake City selected, showing sport mix, ranks, climate, and narrative.
4. **State panel:** Arizona selected for highest Paralympic share, showing rank and Phoenix top hub context.
5. **Gemini map control:** Chat prompt and response for "Show the top Paralympic Hot Spot" with Anchorage selected.
6. **Hometown lookup:** Boise, Idaho focused with the hometown panel visible.
7. **Voice HUD:** Gemini Live voice state visible with a compact Map readout card.

## Thumbnail Concept

Use the live app as the visual base. The best thumbnail is a clean map screenshot with:

- The Hometown Success Engine title visible.
- Red Hot Spot hubs visible.
- Gemini panel open on the right.
- Short overlay text: "Team USA Hometown Hub Map"
- Small subline: "5,119 mapped athletes | 40 hubs | Gemini voice"

Avoid clutter. The thumbnail should communicate map, Team USA, and Gemini interaction within one glance.

## Demo Video Run Of Show

Target length: 90 to 120 seconds.

1. **Open with the problem:** "Instead of ranking medal counts, this maps where Olympians and Paralympians are from."
2. **Show the national map:** Point out 5,119 mapped athletes, 40 hubs, and 10 Paralympic Hot Spots.
3. **Click a Hot Spot:** Show Anchorage or Phoenix and explain the 7.5% threshold and 4.7% baseline.
4. **Use Gemini text:** Ask "What do the dots mean?" or "Show places above the national baseline."
5. **Use Gemini voice:** Ask "Tell me about Vail" or "Show the top Paralympic Hot Spot" and show the map moving.
6. **Show hometown lookup:** Ask "How many athletes are from Boise, Idaho?"
7. **Close with challenge fit:** "This helps explore hometown associations that could guide better Team USA questions without claiming geography guarantees outcomes."
