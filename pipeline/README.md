Hometown Success Engine Pipeline
Overview
This data pipeline ingests, normalizes, and clusters athlete data to power the Hometown Success Engine frontend map application.

Data Sources
Wikidata SPARQL queries
TeamUSA.com (via Gemini extraction)
Optional Keith Galli CSV fallback
Census/NOAA for environment enrichment
Stages
Ingest: Fetch raw athlete data from the defined sources.
Normalize: Standardize fields and resolve entity structures.
Geocode: Translate hometown strings into latitude and longitude coordinates.
Cluster: Run HDBSCAN spatial clustering natively in BigQuery.
Aggregate: Compute statistics per cluster with a k=3 NIL safety threshold.
Export: Push the safely aggregated cluster views to Firestore.
Compliance Note
NIL Guidelines: This pipeline exports aggregate views only. A minimum threshold of k>=3 athletes per cluster is enforced, and no individual athlete-identifying pins are generated or exposed.
Brand/Trademark: This project is a hackathon build and is not an official USOPC product.
Run Instructions
[Placeholder — actual stage scripts will be added in Phase 1]