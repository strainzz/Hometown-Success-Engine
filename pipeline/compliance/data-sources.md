# Data Sources Compliance Memo

**Date:** 2026-05-04
**Author:** AJ Lawrence (Strainz)
**Project:** Hometown Success Engine — Google × Team USA Hackathon

## Purpose

Document the legal and ethical posture for each external data source the
pipeline ingests, so the submission has a clear provenance and compliance
trail for Google judges.

## Sources

### 1. Wikidata SPARQL endpoint
- **URL:** https://query.wikidata.org/sparql
- **License:** CC0 1.0 Universal (public domain)
- **Robots.txt status:** Catch-all `User-agent: *` disallows `/sparql` and `/bigdata` paths from search engine indexing. This is a search-engine indexing directive, not an API access policy. Programmatic SPARQL queries via the documented Wikidata Query Service API are explicitly supported by Wikimedia (per the Wikidata Query Service docs and the existence of the official SPARQLWrapper library). The robots.txt rule prevents Google/Bing from indexing every individual SPARQL query result URL, which would be nonsensical, not from prohibiting use of the API.
- **Rate limit policy:** 60-second per-query hard timeout enforced by the WDQS endpoint. No formal rate limit documented for sequential queries. Wikimedia User-Agent policy requires a descriptive User-Agent header; generic library defaults (e.g., `python-requests/2.x`) get HTTP 403 responses.
- **Compliance posture:** Public domain data, CC0 license, no attribution required (though attribution is good practice). Primary backbone source. Operational compliance via descriptive User-Agent (`HometownSuccessEngine/0.1 (https://github.com/strainzz/Hometown-Success-Engine; strainz@galluslabs.com)`), 60-second query timeout respected, single concurrent query at a time, 429 responses honored.

### 2. Wikidata main domain (wikidata.org)
- **URL:** https://www.wikidata.org
- **License:** CC0 1.0 Universal for data; site terms otherwise
- **Robots.txt status:** Inherits Wikipedia's robots policy. Catch-all `User-agent: *` allows article and entity pages; disallows `/w/`, `/api/`, `/wiki/Special:`, and various deletion/admin discussion pages. Wikidata-specific extension allows `/wiki/Special:EntityData/*.` (the dotted path catches `.json`, `.ttl`, etc. for direct entity API fetches).
- **Compliance posture:** If we fetch entity pages directly outside SPARQL (e.g., via the EntityData JSON API), we do so within allowed paths. Same User-Agent policy as the SPARQL endpoint applies.

### 3. TeamUSA.com (Olympic roster pages)
- **URL:** https://www.teamusa.com
- **License:** All rights reserved (proprietary content)
- **Robots.txt status:** Catch-all `User-agent: *` blocks `/api/`, `/cdn-cgi/`, and image transform URL patterns (`/*w_*`, `/*q_auto*`, `/*f_auto*`, `/*v1/*`). Athlete roster and profile pages are NOT in the disallow list. Site explicitly blocks AI training bots by name (GPTBot, ChatGPT-User, ClaudeBot, anthropic-ai, meta-externalagent, YouBot, PerplexityBot, Amazonbot, Applebot, YandexRenderResourcesBot) and SEO scrapers (SemrushBot, AhrefsBot, MJ12bot, DotBot, BLEXBot, DataForSeoBot, serpstatbot, PetalBot, SeekportBot). Search engines (Googlebot, Bingbot) and link preview bots (Twitterbot, Slackbot, LinkedInBot, facebookexternalhit) are allowed.
- **Compliance posture:** Project is not training a model, so the AI-training-bot directive does not apply directly. However, the spirit of the directive is respected via:
  - Low-volume, single-pass extraction (no continuous crawling)
  - Descriptive project-specific User-Agent that does not impersonate any blocked bot
  - Facts-only storage (athlete name, hometown, sport) — not page content, not copyrighted prose
  - No redistribution of TeamUSA page content in the deployed product
  - 2-second minimum delay between requests (self-imposed, no `Crawl-delay` directive published)
  - Facts (athlete names, hometowns) are not copyrightable per Feist v. Rural Telephone Service
- **Rate limit policy:** No `Crawl-delay` directive present. Self-imposed 2-second minimum delay between requests.

### 4. Paralympic.org (Paralympic roster pages)
- **URL:** https://www.paralympic.org
- **License:** All rights reserved (proprietary content)
- **Robots.txt status:** Standard Drupal-style robots.txt. Blocks admin paths (`/admin/`, `/user/`, `/comment/reply/`, `/search/`, `/node/add/`), media directories (`/media/`, `/es/media/`), live results pages (`/*/info-live-results/`), and video archive. Athlete profile and roster pages are NOT blocked. No AI-training-bot blocks present. Sitemap exposed at `/sitemap.xml`.
- **Compliance posture:** Cleanest of the proprietary sources. Same etiquette as TeamUSA: descriptive User-Agent, 2-second minimum delay, single-pass extraction, facts-only storage, no page content redistribution. Paralympic parity is a strategic judging criterion for this hackathon track.
- **Rate limit policy:** No `Crawl-delay` directive present. Self-imposed 2-second minimum delay.

### 5. Optional: Keith Galli's Olympic dataset (Kaggle CSV)
- **URL:** https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
- **License:** CC0 1.0 Universal (public domain)
- **Approach:** Used only as fallback if Wikidata + TeamUSA gaps persist for specific Olympic Games or sports.
- **Compliance posture:** Public domain, no attribution required.

### 6. US Census + NOAA (enrichment, optional)
- **URL:** https://www.census.gov, https://www.noaa.gov
- **License:** US federal government data (public domain by default)
- **Approach:** Pull city-level demographic and climate data to enrich hometown clusters with context for Gemini explanations.
- **Compliance posture:** Public domain, no concerns.

## NIL & Athlete Privacy Posture

Per USOPC NIL guidelines and the hackathon's Paralympic-parity emphasis:

- The deployed product shows **aggregate cluster views only**. No individual athlete pins on the map.
- Minimum cluster size: **k >= 3 athletes**. Clusters smaller than this are suppressed from the visualization to prevent re-identification.
- No athlete name appears in the deployed UI. The Gemini explanations reference cluster context only ("This cluster includes Olympic gymnasts from the Houston metro area"), never specific people.
- The pipeline **stores** athlete-level data internally for clustering, but the **frontend never receives** it.

## Brand & Trademark Posture

- Project name: "Hometown Success Engine" (hackathon working title)
- This is a hackathon submission, NOT a USOPC product, NOT a Team USA product, NOT a Google product.
- Submission README will include a clear disclaimer: "Independent hackathon build. Not affiliated with or endorsed by USOPC, the IOC, IPC, or Google."
- Olympic rings, Team USA logos, Paralympic logos: NOT used anywhere in the UI or marketing.

## User-Agent String

All HTTP requests from the pipeline will set:

```