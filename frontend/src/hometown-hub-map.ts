// frontend/src/hometown-hub-map.ts
import { Action, Hub, FilterUpdate, WidgetState } from "./widget-contract";
import { Store } from "./store";
import { ApiClient } from "./api-client";
import { Loader } from "@googlemaps/js-api-loader";
import { GoogleMapsOverlay } from "@deck.gl/google-maps";
import { ScatterplotLayer, GeoJsonLayer } from "@deck.gl/layers";
import { geoMercator, geoPath } from "d3-geo";

type StateAggregate = {
  state: string;
  total_athletes: number;
  olympic_count: number;
  paralympic_count: number;
  both_count: number;
  paralympic_share: number;
};

type Narrative = {
  hub_id: string;
  display_name: string;
  headline: string;
  summary: string;
  paralympic_callout: string | null;
  top_sport_phrase: string;
  confidence_qualifier: string;
};

type AthleteGeoPoint = {
  hub_id: string;
  lat: number;
  lon: number;
  status: "olympic" | "paralympic" | "both";
  state: string;
};

const US_STATES_GEOJSON_URL =
  "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json";

const STATE_NAME_TO_CODE: Record<string, string> = {
  "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
  "California": "CA", "Colorado": "CO", "Connecticut": "CT",
  "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
  "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
  "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
  "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
  "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
  "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
  "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
  "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
  "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
  "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
  "Puerto Rico": "PR", "Rhode Island": "RI", "South Carolina": "SC",
  "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
  "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
  "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
  "Wyoming": "WY",
};

function clipAlaskaForInset(feature: any): any {
  if (!feature?.geometry) return feature;
  const geom = feature.geometry;
  if (geom.type !== "MultiPolygon") return feature;
  const filtered = geom.coordinates.filter((poly: any) =>
    poly.some((ring: any[]) =>
      ring.some(([lng, lat]: [number, number]) => lng >= -170 && lng <= -130 && lat >= 50)
    )
  );
  return {
    ...feature,
    geometry: {
      ...geom,
      coordinates: filtered.length ? filtered : geom.coordinates,
    },
  };
}

const DEFAULT_API_URL = import.meta.env.VITE_API_BASE_URL || "https://hometown-success-engine-yumatgk63a-uc.a.run.app";
const GMAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
const GMAPS_MAP_ID = import.meta.env.VITE_GOOGLE_MAPS_MAP_ID;

export class HometownHubMap extends HTMLElement {
  private store: Store;
  private api: ApiClient | null = null;
  private unsubscribe: (() => void) | null = null;
  private map: google.maps.Map | null = null;
  private overlay: GoogleMapsOverlay | null = null;
  private shellRendered: boolean = false;
  private mapInitialized: boolean = false;

  private stateAggregates: StateAggregate[] = [];
  private stateGeoJson: any = null;
  private narrativeCache: Map<string, Narrative> = new Map();
  private athletes: AthleteGeoPoint[] = [];

  constructor() {
    super();
    this.store = new Store();
  }

  static get observedAttributes(): string[] {
    return ["api-url"];
  }

  async connectedCallback(): Promise<void> {
    const apiUrl = this.getAttribute("api-url") || DEFAULT_API_URL;
    this.api = new ApiClient(apiUrl);

    this.unsubscribe = this.store.subscribe(state => this.handleStateUpdate(state));

    if (!this.shellRendered) {
      this.renderShell();
      this.shellRendered = true;
      const resetBtn = this.querySelector("#hubmap-reset-btn");
      if (resetBtn) {
        resetBtn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.resetView();
        });
      }
    }

    await this.initMap();
    await this.loadHubs();
    await this.fetchStateData();
  }

  disconnectedCallback(): void {
    this.unsubscribe?.();
  }

  // ===== PUBLIC API =====

  dispatch(action: Action): WidgetState {
    const newState = this.store.dispatch(action);
    this.dispatchEvent(new CustomEvent("hubmap:state-update", {
      detail: { state: newState, action },
      bubbles: true,
      composed: true
    }));

    if (action.type === "SELECT_HUB") {
      this.dispatchEvent(new CustomEvent("hubmap:hub-selected", {
        detail: { hub_id: action.hub_id },
        bubbles: true,
        composed: true
      }));
    }
    if (action.type === "SET_FILTER" || action.type === "CLEAR_FILTERS") {
      this.dispatchEvent(new CustomEvent("hubmap:filter-changed", {
        detail: { filters: newState.filters },
        bubbles: true,
        composed: true
      }));
    }

    return newState;
  }

  getState(): WidgetState {
    return this.store.getState();
  }

  selectHub(hub_id: string): void {
    this.dispatch({ type: "SELECT_HUB", hub_id });
    if (this.mapInitialized && this.map) {
      const state = this.getState();
      const hub = state.hubs.find(h => h.hub_id === hub_id);
      if (hub) {
        const currentZoom = this.map.getZoom();
        const targetZoom = (currentZoom === undefined || currentZoom < 6) ? 6 : currentZoom;
        this.map.moveCamera({
          center: { lat: hub.centroid_latitude, lng: hub.centroid_longitude },
          zoom: targetZoom,
        });
      }
    }
    void this.ensureNarrative(hub_id);
  }

  filterToParalympic(macro_region?: FilterUpdate["macro_region"]): void {
    this.dispatch({
      type: "SET_FILTER",
      filter: { paralympic_focus: true, macro_region }
    });
  }

  zoomToHub(hub_id: string): void {
    this.dispatch({ type: "SELECT_HUB", hub_id });
    void this.ensureNarrative(hub_id);
    if (!this.map) return;
    const hub = this.store.getState().hubs.find(h => h.hub_id === hub_id);
    if (!hub) return;

    let zoomLevel = 6;
    if (hub_id === "HUB_AK_ANCHORAGE") zoomLevel = 5;
    else if (hub_id === "HUB_HI_HONOLULU") zoomLevel = 7;
    else if (hub_id === "HUB_PR_SAN_JUAN") zoomLevel = 8;

    this.map.moveCamera({
      center: { lat: hub.centroid_latitude, lng: hub.centroid_longitude },
      zoom: zoomLevel,
    });
  }

  resetView(): void {
    this.dispatch({ type: "CLEAR_SELECTION" });
    this.dispatch({ type: "CLEAR_FILTERS" });
    if (this.mapInitialized && this.map) {
      this.map.moveCamera({
        center: { lat: 39.5, lng: -98.0 },
        zoom: 4,
      });
    }
  }

  // ===== INTERNAL =====

  private async loadHubs(): Promise<void> {
    if (!this.api) return;
    try {
      const hubs = await this.api.fetchHubs();
      this.dispatch({ type: "DATA_LOADED", hubs });
      this.dispatchEvent(new CustomEvent("hubmap:data-loaded", {
        detail: { hub_count: hubs.length },
        bubbles: true,
        composed: true
      }));
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      this.dispatch({ type: "DATA_ERROR", error });
      this.dispatchEvent(new CustomEvent("hubmap:data-error", {
        detail: { error },
        bubbles: true,
        composed: true
      }));
    }
  }

  private async fetchStateData(): Promise<void> {
    const baseUrl = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_URL;
    try {
      const [aggregates, geoJson, athletes] = await Promise.all([
        fetch(`${baseUrl}/states/aggregate`).then(r => r.json()),
        fetch(US_STATES_GEOJSON_URL).then(r => r.json()),
        fetch(`${baseUrl}/athletes`).then(r => r.json()),
      ]);
      this.stateAggregates = aggregates;
      this.stateGeoJson = geoJson;
      this.athletes = athletes;
      if (this.overlay) {
        this.updateLayers();
      }
      this.renderInsets();
    } catch (err) {
      console.warn("Failed to fetch state data:", err);
    }
  }

  private async ensureNarrative(hub_id: string): Promise<void> {
    if (this.narrativeCache.has(hub_id)) {
      this.updateNarrativeCard(this.store.getState());
      return;
    }
    const baseUrl = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_URL;
    try {
      const narrative = await fetch(
        `${baseUrl}/hubs/${hub_id}/narrative`
      ).then(r => r.json());
      this.narrativeCache.set(hub_id, narrative);
      this.updateNarrativeCard(this.store.getState());
    } catch (err) {
      console.warn(`Failed to fetch narrative for ${hub_id}:`, err);
    }
  }

  private async initMap(): Promise<void> {
    const loader = new Loader({
      apiKey: GMAPS_API_KEY,
      version: "weekly",
    });
    const { Map } = await loader.importLibrary("maps") as google.maps.MapsLibrary;
    const mapEl = this.querySelector("#hubmap-canvas") as HTMLElement;
    this.map = new Map(mapEl, {
      mapId: GMAPS_MAP_ID,
      center: { lat: 39.5, lng: -98.0 },
      zoom: 4,
      disableDefaultUI: false,
      clickableIcons: false,
      gestureHandling: "greedy",
      backgroundColor: "#ffffff",
    });

    const usBounds = new google.maps.LatLngBounds(
      { lat: 24.0, lng: -125.0 },
      { lat: 49.5, lng: -66.0 }
    );
    this.map.fitBounds(usBounds, 0);
    this.map.setOptions({
      restriction: {
        latLngBounds: { north: 72, south: 17, west: -180, east: -60 },
        strictBounds: false,
      },
      minZoom: 3,
      maxZoom: 12,
    });

    this.mapInitialized = true;
  }

  private buildOverlay(): void {
    this.overlay = new GoogleMapsOverlay({ layers: [] });
    if (this.map) this.overlay.setMap(this.map);
    this.updateLayers();
  }

  private updateLayers(): void {
    const state = this.store.getState();
    const layers: any[] = [];

    if (this.stateGeoJson && this.stateAggregates.length > 0) {
      const stateCounts: Record<string, number> = {};
      let maxCount = 1;
      for (const s of this.stateAggregates) {
        stateCounts[s.state] = s.total_athletes;
        if (s.total_athletes > maxCount) maxCount = s.total_athletes;
      }

      layers.push(new GeoJsonLayer({
        id: "state-choropleth",
        data: this.stateGeoJson,
        getFillColor: (f: any) => {
          const stateName = f.properties?.name || "";
          const code = STATE_NAME_TO_CODE[stateName];
          const count = code ? (stateCounts[code] || 0) : 0;
          if (count === 0) return [239, 234, 230, 100];
          const t = Math.sqrt(count) / Math.sqrt(maxCount);
          const r = Math.round(220 - (220 - 21) * t);
          const g = Math.round(228 - (228 - 41) * t);
          const b = Math.round(240 - (240 - 105) * t);
          return [r, g, b, 200];
        },
        getLineColor: [255, 255, 255, 255],
        getLineWidth: 1,
        lineWidthUnits: "pixels",
        stroked: true,
        filled: true,
        pickable: false,
      }));
    }

    if (this.athletes.length > 0) {
      layers.push(new ScatterplotLayer({
        id: "athlete-constellation",
        data: this.athletes,
        getPosition: (a: AthleteGeoPoint) => [a.lon, a.lat],
        getRadius: 1500,
        radiusUnits: "meters",
        radiusMinPixels: 1.5,
        radiusMaxPixels: 3,
        getFillColor: (a: AthleteGeoPoint) => {
          if (a.status === "paralympic" || a.status === "both") {
            return [211, 17, 24, 180];
          }
          return [21, 41, 105, 100];
        },
        stroked: false,
        filled: true,
        pickable: false,
      }));
    }

    layers.push(new ScatterplotLayer({
      id: "hub-centroids",
      data: state.hubs,
      getPosition: (h: Hub) => [h.centroid_longitude, h.centroid_latitude],
      getRadius: (h: Hub) => {
        const base = Math.sqrt(h.total_athletes) * 8000;
        return h.hub_id === state.selectedHubId ? base * 1.4 : base;
      },
      radiusUnits: "meters",
      radiusMinPixels: 10,
      radiusMaxPixels: 50,
      getFillColor: (h: Hub) => {
        if (h.hub_id === state.selectedHubId) return [211, 17, 24, 255];
        if (h.is_paralympic_hot_spot) return [211, 17, 24, 230];
        return [21, 41, 105, 230];
      },
      getLineColor: (h: Hub) => {
        if (h.hub_id === state.selectedHubId) return [255, 255, 255, 255];
        if (h.is_paralympic_hot_spot) return [211, 17, 24, 255];
        return [255, 255, 255, 255];
      },
      getLineWidth: (h: Hub) => {
        if (h.hub_id === state.selectedHubId) return 6;
        return h.is_paralympic_hot_spot ? 4 : 2;
      },
      lineWidthUnits: "pixels",
      stroked: true,
      filled: true,
      pickable: true,
      onClick: (info: any) => {
        if (info.object) this.selectHub(info.object.hub_id);
      },
      updateTriggers: {
        getRadius: [state.selectedHubId],
        getFillColor: [state.selectedHubId],
        getLineColor: [state.selectedHubId],
        getLineWidth: [state.selectedHubId],
      },
    }));

    if (this.overlay) {
      this.overlay.setProps({
        layers,
        getTooltip: ({ object }: any) => {
          if (!object || !object.hub_id) return null;
          const hub = object as Hub;
          const paraPct = (hub.composition.paralympic_share * 100).toFixed(1);
          const hotSpotTag = hub.is_paralympic_hot_spot
            ? `<div style="color: #d31118; font-weight: 700; font-size: 10px; letter-spacing: 1px; margin-top: 4px;">★ PARALYMPIC HOT SPOT</div>`
            : "";
          return {
            html: `
              <div style="font-family: system-ui, sans-serif;">
                <div style="font-weight: 700; font-size: 14px; color: #152969;">${hub.display_name}</div>
                <div style="font-size: 12px; color: #171fbe; margin-top: 2px;">${hub.region_name}</div>
                <div style="font-size: 11px; color: #484645; margin-top: 4px;">
                  ${hub.total_athletes} athletes · ${paraPct}% Paralympic
                </div>
                ${hotSpotTag}
              </div>
            `,
            style: {
              backgroundColor: "rgba(255, 255, 255, 0.98)",
              border: "1px solid #b9bfd2",
              borderRadius: "4px",
              padding: "8px 10px",
              boxShadow: "0 2px 6px rgba(0, 0, 0, 0.15)",
              fontSize: "12px",
              pointerEvents: "none",
            },
          };
        },
      });
    }
  }

  private handleStateUpdate(state: WidgetState): void {
    if (state.loadStatus === "loaded" && !this.overlay && this.mapInitialized) {
      this.buildOverlay();
    } else if (this.overlay) {
      this.updateLayers();
    }

    if (this.shellRendered) {
      this.updateLegendCards(state);
      this.updateNarrativeCard(state);
      this.renderInsets();
    }
  }

  private renderShell(): void {
    this.innerHTML = `
      <div style="display: flex; flex-direction: column;
            background: #ffffff; font-family: system-ui, -apple-system, sans-serif;">

        <header style="background: #152969; color: #ffffff; padding: 16px 24px;">
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 20px; font-weight: 700; letter-spacing: 0.5px;">
              Hometown Success Engine
            </div>
            <div style="font-size: 13px; color: #b9bfd2;
                  text-transform: uppercase; letter-spacing: 1px;">
              Team USA Athletic Hub Map
            </div>
          </div>
          <div style="font-size: 14px; color: #b9bfd2;
                margin-top: 6px; font-weight: 400; letter-spacing: 0.3px;">
            Mapping 5,012 Olympians and Paralympians across 37 hometown regions where America's next Team USA roster could emerge
          </div>
        </header>

        <div style="position: relative; width: 100%; height: 600px;">
          <div id="hubmap-canvas"
              style="width: 100%; height: 100%; background: #ffffff;"></div>
          <div id="hubmap-insets-wrapper"
              style="position: absolute; bottom: 12px; left: 12px;
                 display: flex; flex-direction: column;
                 gap: 8px; align-items: flex-start;
                 pointer-events: auto;">
            <button id="hubmap-reset-btn"
                    type="button"
                    title="Reset to continental US view"
                    style="display: flex; align-items: center; gap: 6px;
                       background: rgba(255, 255, 255, 0.95);
                       color: #152969;
                       border: 1px solid #b9bfd2; border-radius: 18px;
                       cursor: pointer; padding: 6px 12px;
                       font-size: 12px; font-weight: 600;
                       letter-spacing: 0.5px;
                       font-family: system-ui, -apple-system, sans-serif;
                       box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                       transition: background 0.15s;"
                    onmouseover="this.style.background='#efeae6';"
                    onmouseout="this.style.background='rgba(255, 255, 255, 0.95)';">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                   stroke="#152969" stroke-width="2.5"
                   stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 12a9 9 0 1 0 3-6.7" />
                <polyline points="3 4 3 10 9 10" />
              </svg>
              Reset View
            </button>
            <div id="hubmap-color-legend"
                style="background: rgba(255, 255, 255, 0.95);
                   border: 1px solid #b9bfd2; border-radius: 6px;
                   padding: 10px 12px;
                   box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                   font-family: system-ui, -apple-system, sans-serif;">
              <div style="font-size: 10px; color: #484645;
                    text-transform: uppercase; letter-spacing: 1px;
                    font-weight: 600; margin-bottom: 6px;">
                Athletes per state
              </div>
              <div style="width: 140px; height: 10px;
                    background: linear-gradient(to right,
                      rgb(220, 228, 240),
                      rgb(120, 134, 172),
                      rgb(21, 41, 105));
                    border-radius: 2px;"></div>
              <div style="display: flex; justify-content: space-between;
                    margin-top: 4px;
                    font-size: 10px; color: #484645;">
                <span>1</span>
                <span>~100</span>
                <span>749</span>
              </div>
              <div style="margin-top: 10px; padding-top: 8px;
                    border-top: 1px solid #e4e4e7;
                    display: flex; gap: 12px;
                    font-size: 11px; color: #484645;">
                <div style="display: flex; align-items: center; gap: 4px;">
                  <span style="display: inline-block; width: 10px;
                         height: 10px; border-radius: 50%;
                         background: #d31118;"></span>
                  Hot Spot
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                  <span style="display: inline-block; width: 10px;
                         height: 10px; border-radius: 50%;
                         background: #152969;"></span>
                  Hub
                </div>
              </div>
              <div style="display: flex; gap: 12px;
                    margin-top: 6px;
                    font-size: 11px; color: #484645;">
                <div style="display: flex; align-items: center; gap: 4px;">
                  <span style="display: inline-block; width: 4px;
                         height: 4px; border-radius: 50%;
                         background: #d31118;"></span>
                  Para athlete
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                  <span style="display: inline-block; width: 4px;
                         height: 4px; border-radius: 50%;
                         background: #152969; opacity: 0.4;"></span>
                  Olympian
                </div>
              </div>
            </div>
            <div id="hubmap-insets"
                style="display: flex; gap: 8px;
                   background: rgba(255, 255, 255, 0.95);
                   border: 1px solid #b9bfd2;
                   border-radius: 6px; padding: 8px;
                   box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);">
            </div>
          </div>
        </div>

        <section id="hubmap-legend"
              style="display: grid; grid-template-columns: repeat(4, 1fr);
                 gap: 16px; padding: 24px;
                 background: #efeae6;
                 border-top: 1px solid #b9bfd2;">
        </section>

        <section id="hubmap-narrative"
              style="padding: 24px; background: #ffffff;
                 min-height: 100px;
                 border-top: 1px solid #b9bfd2;">
          <div style="color: #b9bfd2; font-size: 14px;
                font-style: italic;">
            Click a hub on the map to see its narrative.
          </div>
        </section>

      </div>
    `;
  }

  private updateLegendCards(state: WidgetState): void {
    const legendEl = this.querySelector("#hubmap-legend");
    if (!legendEl) return;

    let hotSpots = 0;
    let totalAthletes = 0;
    let totalShare = 0;

    for (const h of state.hubs) {
      if (h.is_paralympic_hot_spot) hotSpots++;
      totalAthletes += h.total_athletes;
      totalShare += h.composition.paralympic_share;
    }

    const avgParaShare = state.hubs.length > 0 ? (totalShare / state.hubs.length) * 100 : 0;
    const hubsDiscovered = state.hubs.length;

    const createCard = (label: string, value: string | number) => `
      <div style="background: #ffffff; padding: 16px; border-radius: 4px;
            border: 1px solid #b9bfd2;">
        <div style="color: #484645; font-size: 11px;
              text-transform: uppercase; letter-spacing: 1.5px;
              font-weight: 600; margin-bottom: 8px;">
          ${label}
        </div>
        <div style="color: #171fbe; font-size: 32px; font-weight: 700;
              line-height: 1;">
          ${value}
        </div>
      </div>
    `;

    legendEl.innerHTML = `
      ${createCard("Paralympic Hot Spots", hotSpots)}
      ${createCard("Total Athletes", totalAthletes)}
      ${createCard("Hubs Discovered", hubsDiscovered)}
      ${createCard("Avg Para Share", avgParaShare.toFixed(1) + "%")}
    `;
  }

  private updateNarrativeCard(state: WidgetState): void {
    const card = this.querySelector("#hubmap-narrative") as HTMLElement;
    if (!card) return;

    if (!state.selectedHubId) {
      card.innerHTML = `
        <div style="color: #b9bfd2; font-size: 14px;
              font-style: italic;">
          Click a hub on the map to see its narrative.
        </div>
       `;
      return;
    }

    const hub = state.hubs.find(h => h.hub_id === state.selectedHubId);
    if (!hub) return;

    const narrative = this.narrativeCache.get(state.selectedHubId);
    const hotSpotBadge = hub.is_paralympic_hot_spot
      ? `<span style="color: #d31118; font-weight: 700;
             margin-left: 8px;
             font-size: 12px; letter-spacing: 1px;">
       ★ HOT SPOT
      </span>`
      : "";

    const callout = narrative?.paralympic_callout
      ? `<div style="margin-top: 12px; padding: 12px;
            background: #d31118; color: #ffffff;
            border-radius: 4px; font-size: 14px;">
       <strong>Paralympic Hot Spot:</strong>
       ${narrative.paralympic_callout}
      </div>`
      : "";

    const summary = narrative
      ? `<p style="margin: 12px 0 0 0; line-height: 1.6;
             color: #484645; font-size: 14px;">
       ${narrative.summary}
      </p>`
      : `<p style="margin: 12px 0 0 0; color: #b9bfd2;
             font-size: 13px; font-style: italic;">
       Loading narrative...
      </p>`;

    const headline = narrative
      ? `<h2 style="margin: 8px 0 0 0; color: #171fbe;
             font-size: 20px; font-weight: 700;">
       ${narrative.headline}
      </h2>`
      : "";

    card.innerHTML = `
      <div>
       <div style="display: flex; align-items: baseline; gap: 8px;">
        <span style="color: #152969; font-size: 24px;
               font-weight: 800;">
         ${hub.display_name}
        </span>
        ${hotSpotBadge}
       </div>
       <div style="color: #171fbe; font-size: 15px;
             margin-top: 4px;">
        ${hub.region_name} · <span style="color: #484645;
                       font-size: 13px;
                       text-transform: uppercase;
                       letter-spacing: 1px;">
         ${hub.macro_region}
        </span>
       </div>
       ${headline}
       ${summary}
       <div style="display: flex; gap: 24px; margin-top: 16px;
             padding-top: 16px;
             border-top: 1px solid #b9bfd2;">
        <div>
         <div style="font-size: 11px; color: #484645;
               text-transform: uppercase;
               letter-spacing: 1px;">
          Athletes
         </div>
         <div style="color: #152969; font-size: 24px;
               font-weight: 700;">
          ${hub.total_athletes}
         </div>
        </div>
        <div>
         <div style="font-size: 11px; color: #484645;
               text-transform: uppercase;
               letter-spacing: 1px;">
          Paralympic Share
         </div>
         <div style="color: #152969; font-size: 24px;
               font-weight: 700;">
          ${(hub.composition.paralympic_share * 100).toFixed(1)}%
         </div>
        </div>
       </div>
       ${callout}
      </div>
     `;
  }

  private renderInsets(): void {
    if (!this.stateGeoJson || this.stateAggregates.length === 0) return;
    const container = this.querySelector("#hubmap-insets") as HTMLElement;
    if (!container) return;

    const stateCounts: Record<string, number> = {};
    let maxCount = 1;
    for (const s of this.stateAggregates) {
      stateCounts[s.state] = s.total_athletes;
      if (s.total_athletes > maxCount) maxCount = s.total_athletes;
    }
    const colorFor = (code: string): string => {
      const count = stateCounts[code] || 0;
      if (count === 0) return "#efeae6";
      const t = Math.sqrt(count) / Math.sqrt(maxCount);
      const r = Math.round(220 - (220 - 21) * t);
      const g = Math.round(228 - (228 - 41) * t);
      const b = Math.round(240 - (240 - 105) * t);
      return `rgb(${r}, ${g}, ${b})`;
    };

    const state = this.store.getState();
    const insets = [
      {
        label: "AK",
        stateName: "Alaska",
        hubId: "HUB_AK_ANCHORAGE",
        width: 110,
        height: 80,
      },
      {
        label: "HI",
        stateName: "Hawaii",
        hubId: "HUB_HI_HONOLULU",
        width: 90,
        height: 70,
      },
      {
        label: "PR",
        stateName: "Puerto Rico",
        hubId: "HUB_PR_SAN_JUAN",
        width: 90,
        height: 60,
      },
    ];

    container.innerHTML = insets.map(inset => {
      const feature = this.stateGeoJson.features.find(
        (f: any) => f.properties?.name === inset.stateName
      );
      if (!feature) return "";

      const hub = state.hubs.find(h => h.hub_id === inset.hubId);

      const displayFeature = inset.label === "AK"
        ? clipAlaskaForInset(feature)
        : feature;

      const projection = geoMercator().fitExtent(
        [[4, 4], [inset.width - 4, inset.height - 4]],
        displayFeature
      );

      const pathGen = geoPath(projection);
      const pathData = pathGen(displayFeature) || "";

      const stateCode = STATE_NAME_TO_CODE[inset.stateName] || "XX";
      const fillColor = colorFor(stateCode);

      let hubDot = "";
      if (hub) {
        const [hx, hy] = projection([
          hub.centroid_longitude,
          hub.centroid_latitude
        ]) || [0, 0];
        const isSelected = hub.hub_id === state.selectedHubId;
        const isHotSpot = hub.is_paralympic_hot_spot;
        const dotFill = isSelected ? "#d31118"
          : isHotSpot ? "#d31118" : "#152969";
        const dotStroke = isHotSpot ? "#d31118" : "#ffffff";
        const dotR = isSelected ? 5 : 4;
        const strokeW = isHotSpot ? 2 : 1;
        hubDot = `<circle
          cx="${hx}" cy="${hy}" r="${dotR}"
          fill="${dotFill}"
          stroke="${dotStroke}"
          stroke-width="${strokeW}"
          style="cursor: pointer;"
          data-hub-id="${hub.hub_id}"
        />`;
      }

      return `
        <div data-inset="${inset.label}"
           data-hub-id="${inset.hubId}"
           style="display: flex; flex-direction: column;
              align-items: center; cursor: pointer;
              padding: 4px; border-radius: 4px;
              transition: background 0.15s;"
           onmouseover="this.style.background='#efeae6';"
           onmouseout="this.style.background='transparent';">
         <svg width="${inset.width}" height="${inset.height}"
            viewBox="0 0 ${inset.width} ${inset.height}"
            style="background: #ffffff; border: 1px solid #e4e4e7;
               border-radius: 3px; pointer-events: none;">
          <path d="${pathData}"
             fill="${fillColor}"
             stroke="#ffffff"
             stroke-width="1"
             stroke-linejoin="round" />
          ${hubDot}
         </svg>
         <div style="font-size: 10px; color: #484645;
               font-weight: 600; letter-spacing: 1px;
               margin-top: 4px;">
          ${inset.label}
         </div>
        </div>
      `;
    }).join("");

    container.querySelectorAll("[data-inset][data-hub-id]").forEach(el => {
      const hubId = el.getAttribute("data-hub-id");
      if (!hubId) return;
      el.addEventListener("click", (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        this.zoomToHub(hubId);
      });
    });
  }
}