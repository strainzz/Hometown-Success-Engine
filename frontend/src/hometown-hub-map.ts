// frontend/src/hometown-hub-map.ts
import { Action, Hub, FilterUpdate, WidgetState } from "./widget-contract";
import { Store } from "./store";
import { ApiClient } from "./api-client";
import { Loader } from "@googlemaps/js-api-loader";
import { GoogleMapsOverlay } from "@deck.gl/google-maps";
import { ScatterplotLayer, GeoJsonLayer } from "@deck.gl/layers";

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

const US_STATES_GEOJSON_URL =
  "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json";

// State NAME (as in PublicaMundi GeoJSON) → 2-letter USPS code
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
        this.map.panTo({ lat: hub.centroid_latitude, lng: hub.centroid_longitude });
        const currentZoom = this.map.getZoom();
        if (currentZoom === undefined || currentZoom < 6) {
          this.map.setZoom(6);
        }
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
    if (this.mapInitialized && this.map) {
      const state = this.getState();
      const hub = state.hubs.find(h => h.hub_id === hub_id);
      if (hub) {
        this.map.panTo({ lat: hub.centroid_latitude, lng: hub.centroid_longitude });
        this.map.setZoom(7);
      }
    }
  }

  resetView(): void {
    this.dispatch({ type: "CLEAR_SELECTION" });
    this.dispatch({ type: "CLEAR_FILTERS" });
    if (this.mapInitialized && this.map) {
      this.map.panTo({ lat: 39.5, lng: -98.0 });
      this.map.setZoom(4);
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
      const [aggregates, geoJson] = await Promise.all([
        fetch(`${baseUrl}/states/aggregate`).then(r => r.json()),
        fetch(US_STATES_GEOJSON_URL).then(r => r.json()),
      ]);
      this.stateAggregates = aggregates;
      this.stateGeoJson = geoJson;
      // Re-trigger overlay update with state layer included
      if (this.overlay) {
        this.updateLayers();
      }
    } catch (err) {
      // Non-fatal: state choropleth missing, but hubs still render
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

    // Constrain default view to continental US
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

    // STATE CHOROPLETH LAYER (background)
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
          if (count === 0) return [239, 234, 230, 100]; // ts-cream faint
          // Sqrt scale for better visual spread
          const t = Math.sqrt(count) / Math.sqrt(maxCount);
          // Interpolate from light blue to ts-navy
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

    // HUB CENTROIDS LAYER (foreground, clickable)
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
      this.overlay.setProps({ layers });
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
    }
  }

  private renderShell(): void {
    this.innerHTML = `
      <div style="display: flex; flex-direction: column;
            background: #ffffff; font-family: system-ui, -apple-system, sans-serif;">

        <header style="background: #152969; color: #ffffff;
              padding: 16px 24px; display: flex;
              align-items: center; justify-content: space-between;">
          <div style="font-size: 20px; font-weight: 700;
                letter-spacing: 0.5px;">
            Hometown Success Engine
          </div>
          <div style="font-size: 13px; color: #b9bfd2;
                text-transform: uppercase;
                letter-spacing: 1px;">
            Team USA Athletic Hub Map
          </div>
        </header>

        <div id="hubmap-canvas"
            style="width: 100%; height: 600px;
                background: #ffffff;"></div>

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
}