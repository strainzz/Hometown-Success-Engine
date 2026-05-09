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

type Climate = {
  annual_avg_temp_f?: number | null;
  annual_precipitation_in?: number | null;
  annual_sunshine_hours?: number | null;
  elevation_ft?: number | null;
};

type Narrative = {
  hub_id: string;
  display_name: string;
  headline: string;
  summary: string;
  paralympic_callout: string | null;
  top_sport_phrase: string;
  confidence_qualifier: string;
  climate?: Climate | null;
  geographic_context?: string | null;
};

type AthleteGeoPoint = {
  hub_id: string;
  lat: number;
  lon: number;
  status: "olympic" | "paralympic" | "both";
  state: string;
};

type ChatToolCall = {
  name: "select_hub" | "filter_to_paralympic" | "zoom_to_hub" | "reset_view" | "select_state" | "query_data" | "explain_map" | "focus_hometown";
  args: Record<string, any>;
};

type HometownFocus = {
  hometown: string;
  state?: string;
  state_code?: string;
  lat?: number;
  lon?: number;
  total_athletes: number;
  olympic_count: number;
  paralympic_count: number;
  both_count: number;
  paralympic_share?: number;
  top_sports?: { sport: string; count: number }[];
  hub_id?: string;
  hub_name?: string;
  distance_to_hub_km?: number | null;
  resolved?: boolean;
  ambiguous?: boolean;
  geocode_query?: string;
  nearest_hub_name?: string;
  nearest_hub_distance_km?: number;
  options?: { hometown: string; state: string; total_athletes: number; hub_name?: string }[];
};

type ChatTurn = {
  role: "user" | "model";
  text: string;
};

type ChatResponse = {
  text: string;
  tool_calls: ChatToolCall[];
  history: ChatTurn[];
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

const SUGGESTED_PROMPTS = [
  "What do the dots mean?",
  "How many athletes are from Boise, Idaho?",
  "Show Paralympic Hot Spots",
  "Tell me about Vail",
];

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

  private chatOpen: boolean = false;
  private chatHistory: ChatTurn[] = [];
  private chatLoading: boolean = false;
  private chatSessionId: string = this.createSessionId();
  private voiceSocket: WebSocket | null = null;
  private voiceListening: boolean = false;
  private voiceInputStream: MediaStream | null = null;
  private voiceInputContext: AudioContext | null = null;
  private voiceInputSource: MediaStreamAudioSourceNode | null = null;
  private voiceInputNode: AudioWorkletNode | ScriptProcessorNode | null = null;
  private voiceInputBuffer: Float32Array[] = [];
  private voiceBufferedSamples: number = 0;
  private voiceInputSampleRate: number = 48000;
  private voiceWorkletUrl: string | null = null;
  private voiceUserDraftIndex: number | null = null;
  private voiceModelDraftIndex: number | null = null;
  private voiceTurnId: number = 0;
  private activeVoiceTurnId: number = 0;
  private voiceClosingAfterTurn: boolean = false;
  private voiceReadyTimer: number | null = null;
  private voiceResponseTimer: number | null = null;
  private voiceLastInputText: string = "";
  private audioEnabled: boolean = true;
  private audioContext: AudioContext | null = null;
  private audioPlayTime: number = 0;
  private audioQueue: Promise<void> = Promise.resolve();
  private audioChunkBuffers: Map<string, { mimeType: string; source: string; total: number; chunks: string[] }> = new Map();
  private audioSources: AudioBufferSourceNode[] = [];
  private audioElements: HTMLAudioElement[] = [];
  private hoveredHubId: string | null = null;
  private selectedStateCode: string | null = null;
  private selectedStateName: string | null = null;
  private selectedHometown: HometownFocus | null = null;

  constructor() {
    super();
    this.store = new Store();
  }

  static get observedAttributes(): string[] {
    return ["api-url"];
  }

  private createSessionId(): string {
    const cryptoObj = window.crypto as Crypto | undefined;
    if (cryptoObj?.randomUUID) return cryptoObj.randomUUID();
    return `hsm-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  async connectedCallback(): Promise<void> {
    const apiUrl = this.getAttribute("api-url") || DEFAULT_API_URL;
    this.api = new ApiClient(apiUrl);

    this.unsubscribe = this.store.subscribe(state => this.handleStateUpdate(state));

    if (!this.shellRendered) {
      this.renderShell();
      this.shellRendered = true;
      this.wireResetButton();
      this.wireChatUI();
    }

    await this.initMap();
    await this.loadHubs();
    await this.fetchStateData();

    this.resetView();
  }

  disconnectedCallback(): void {
    this.unsubscribe?.();
    this.stopVoiceStreaming(false);
    if (this.voiceSocket && this.voiceSocket.readyState === WebSocket.OPEN) {
      this.voiceSocket.send(JSON.stringify({ type: "close" }));
    }
    this.voiceSocket?.close();
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

  private centerOnState(stateCode: string): void {
      if (!this.map || !this.stateGeoJson) return;
      const stateName = Object.keys(STATE_NAME_TO_CODE).find(k => STATE_NAME_TO_CODE[k] === stateCode);
      if (!stateName) return;
      const feature = this.stateGeoJson.features.find((f: any) => f.properties?.name === stateName);
      if (!feature?.geometry) return;

      let minLng = 180, maxLng = -180, minLat = 90, maxLat = -90;
      const walk = (coords: any) => {
        if (typeof coords[0] === "number") {
          const [lng, lat] = coords;
          if (lng < minLng) minLng = lng;
          if (lng > maxLng) maxLng = lng;
          if (lat < minLat) minLat = lat;
          if (lat > maxLat) maxLat = lat;
        } else {
          coords.forEach(walk);
        }
      };
      walk(feature.geometry.coordinates);

      const centerLng = (minLng + maxLng) / 2;
      const centerLat = (minLat + maxLat) / 2;
      const lngSpan = maxLng - minLng;
      const latSpan = maxLat - minLat;
      const span = Math.max(lngSpan, latSpan);

      let zoom = 6;
      if (span > 30) zoom = 4;
      else if (span > 15) zoom = 5;
      else if (span > 8) zoom = 6;
      else if (span > 4) zoom = 7;
      else zoom = 8;

      this.map.moveCamera({
        center: { lat: centerLat, lng: centerLng },
        zoom,
      });
    }
  selectHub(hub_id: string): void {
    this.selectedHometown = null;
    this.renderHometownPanel();
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
    this.selectedHometown = null;
    this.renderHometownPanel();
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
    this.selectedHometown = null;
    this.renderHometownPanel();
    if (this.mapInitialized && this.map) {
      this.map.moveCamera({
        center: { lat: 39.5, lng: -98.0 },
        zoom: 4,
      });
    }
  }

  // ===== CHAT =====

  private async sendChatMessage(
    message: string,
    options: { recordUser?: boolean; speak?: boolean } = {},
  ): Promise<string | null> {
    if (!message.trim() || this.chatLoading) return null;
    const recordUser = options.recordUser !== false;
    this.chatLoading = true;
    if (recordUser) {
      this.chatHistory.push({ role: "user", text: message });
    }
    this.renderChatBody();

    const baseUrl = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_URL;
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 25000);
    try {
      const res = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          message,
          history: this.chatHistory.slice(0, -1),
          session_id: this.chatSessionId,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ChatResponse = await res.json();

      this.chatHistory.push({ role: "model", text: data.text });
      if (options.speak) this.speakText(data.text);

      for (const call of data.tool_calls) {
        this.dispatchToolCall(call);
      }
      return data.text;
    } catch (err) {
      const fallbackText = err instanceof DOMException && err.name === "AbortError"
        ? "Gemini is taking longer than expected. Try a direct map question like 'Show Paralympic Hot Spots' or ask again in a moment."
        : "Sorry, I had trouble connecting. Try again in a moment.";
      this.chatHistory.push({
        role: "model",
        text: fallbackText,
      });
      if (options.speak) this.speakText(fallbackText);
      return null;
    } finally {
      window.clearTimeout(timeoutId);
      this.chatLoading = false;
      this.renderChatBody();
    }
  }

  private dispatchToolCall(call: ChatToolCall): void {
    switch (call.name) {
      case "select_hub":
        this.selectedHometown = null;
        this.renderHometownPanel();
        if (call.args.hub_id) this.selectHub(call.args.hub_id);
        break;
      case "zoom_to_hub":
        this.selectedHometown = null;
        this.renderHometownPanel();
        if (call.args.hub_id) this.zoomToHub(call.args.hub_id);
        break;
      case "filter_to_paralympic":
        this.filterToParalympic(call.args.macro_region);
        break;
      case "select_state":
        this.selectedHometown = null;
        this.renderHometownPanel();
        if (call.args.state_code) {
          const code = (call.args.state_code as string).toUpperCase();
          const stateName = Object.keys(STATE_NAME_TO_CODE).find(k => STATE_NAME_TO_CODE[k] === code);
          if (stateName) {
            this.selectedStateCode = code;
            this.selectedStateName = stateName;
            this.renderStatePanel();
            this.updateLayers();
            this.centerOnState(code);
          }
        }
        break;
      case "reset_view":
        this.resetView();
        break;
      case "query_data":
      case "explain_map":
        break;
      case "focus_hometown":
        void this.focusHometown(call.args as HometownFocus);
        break;
    }
  }

  private async focusHometown(args: HometownFocus): Promise<void> {
    this.selectedStateCode = null;
    this.selectedStateName = null;
    this.dispatch({ type: "CLEAR_SELECTION" });
    const lat = Number(args.lat);
    const lon = Number(args.lon);
    if (args.resolved && Number.isFinite(lat) && Number.isFinite(lon)) {
      this.selectedHometown = { ...args, lat, lon };
      this.renderStatePanel();
      this.renderHometownPanel();
      this.updateLayers();
      this.map?.moveCamera({ center: { lat, lng: lon }, zoom: 9 });
      return;
    }

    if (args.ambiguous) {
      this.selectedHometown = args;
      this.renderStatePanel();
      this.renderHometownPanel();
      this.updateLayers();
      return;
    }

    const geocodeQuery = args.geocode_query || args.hometown;
    if (!geocodeQuery || !this.map || typeof google === "undefined") {
      this.selectedHometown = args;
      this.renderHometownPanel();
      return;
    }

    try {
      const geocoder = new google.maps.Geocoder();
      const response = await geocoder.geocode({ address: geocodeQuery });
      const result = response.results?.[0];
      const loc = result?.geometry?.location;
      if (!loc) throw new Error("No geocode result");
      const resolvedLat = loc.lat();
      const resolvedLon = loc.lng();
      const nearest = this.findNearestHub(resolvedLat, resolvedLon);
      this.selectedHometown = {
        ...args,
        hometown: args.hometown || result.formatted_address || geocodeQuery,
        lat: resolvedLat,
        lon: resolvedLon,
        total_athletes: 0,
        olympic_count: 0,
        paralympic_count: 0,
        both_count: 0,
        nearest_hub_name: nearest?.hub.display_name,
        nearest_hub_distance_km: nearest?.distanceKm,
      };
      this.renderStatePanel();
      this.renderHometownPanel();
      this.updateLayers();
      this.map.moveCamera({ center: { lat: resolvedLat, lng: resolvedLon }, zoom: 9 });
    } catch {
      this.selectedHometown = args;
      this.renderHometownPanel();
    }
  }

  private findNearestHub(lat: number, lon: number): { hub: Hub; distanceKm: number } | null {
    const hubs = this.store.getState().hubs;
    let best: { hub: Hub; distanceKm: number } | null = null;
    for (const hub of hubs) {
      const distanceKm = this.distanceKm(lat, lon, hub.centroid_latitude, hub.centroid_longitude);
      if (!best || distanceKm < best.distanceKm) best = { hub, distanceKm: Math.round(distanceKm) };
    }
    return best;
  }

  private distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const toRad = (value: number) => value * Math.PI / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat / 2) ** 2
      + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
    return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  private getVoiceWsUrl(): string {
    const baseUrl = import.meta.env.VITE_API_BASE_URL || DEFAULT_API_URL;
    const url = new URL(baseUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = "/voice/ws";
    url.search = "";
    url.searchParams.set("session_id", this.chatSessionId);
    url.hash = "";
    return url.toString();
  }

  private setVoiceHud(state: string, label: string, detail: string = ""): void {
    const hud = this.querySelector("#hubmap-voice-hud") as HTMLElement;
    const dot = this.querySelector("#hubmap-voice-dot") as HTMLElement;
    const status = this.querySelector("#hubmap-voice-status") as HTMLElement;
    const sub = this.querySelector("#hubmap-voice-detail") as HTMLElement;
    const wave = this.querySelector("#hubmap-voice-wave") as HTMLElement;
    if (!hud || !dot || !status || !sub || !wave) return;

    const activeRed = state === "listening" || state === "interrupted";
    const activeBlue = state === "replying" || state === "thinking" || state === "tool" || state === "connecting";
    const error = state === "error";
    const color = error ? "#b3261e" : activeRed ? "#d31118" : activeBlue ? "#152969" : "#7886ac";
    hud.setAttribute("data-state", state);
    hud.style.background = activeRed ? "#fff5f5" : activeBlue ? "#f5f7fb" : error ? "#fff4f2" : "#f8f7f5";
    hud.style.borderTopColor = activeRed ? "#f0b3b6" : activeBlue ? "#b9bfd2" : error ? "#f2b8b5" : "#efeae6";
    dot.style.background = color;
    dot.style.boxShadow = activeRed || activeBlue ? `0 0 0 4px ${activeRed ? "rgba(211,17,24,0.12)" : "rgba(21,41,105,0.12)"}` : "none";
    status.textContent = label;
    sub.textContent = detail || (state === "idle" ? "Press the mic to ask Gemini with voice." : "");
    [...wave.children].forEach((child, index) => {
      const bar = child as HTMLElement;
      bar.style.background = color;
      bar.style.opacity = activeRed || activeBlue ? String(0.45 + index * 0.16) : "0.25";
      bar.style.height = activeRed || activeBlue ? `${6 + ((index % 3) * 5)}px` : "5px";
    });
  }

  private setVoiceButton(active: boolean): void {
    const btn = this.querySelector("#hubmap-voice-btn") as HTMLButtonElement;
    if (!btn) return;
    const iconColor = active ? "#ffffff" : "#152969";
    btn.setAttribute("aria-label", active ? "Stop listening" : "Speak to Gemini");
    btn.setAttribute("title", active ? "Stop listening" : "Speak to Gemini");
    btn.style.background = active ? "#d31118" : "#ffffff";
    btn.style.borderColor = active ? "#d31118" : "#152969";
    btn.style.boxShadow = active ? "0 0 0 3px rgba(211, 17, 24, 0.16)" : "none";
    btn.innerHTML = `
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
           stroke="${iconColor}" stroke-width="2.4"
           stroke-linecap="round" stroke-linejoin="round"
           aria-hidden="true">
        <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <path d="M12 19v3" />
      </svg>
    `;
  }

  private setAudioButton(): void {
    const btn = this.querySelector("#hubmap-audio-btn") as HTMLButtonElement;
    if (!btn) return;
    const iconColor = this.audioEnabled ? "#152969" : "#484645";
    btn.setAttribute("aria-label", this.audioEnabled ? "Turn audio off" : "Turn audio on");
    btn.setAttribute("title", this.audioEnabled ? "Audio on" : "Audio off");
    btn.style.background = this.audioEnabled ? "#ffffff" : "#f8f7f5";
    btn.style.borderColor = this.audioEnabled ? "#152969" : "#b9bfd2";
    btn.style.opacity = this.audioEnabled ? "1" : "0.72";
    btn.innerHTML = this.audioEnabled ? `
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
           stroke="${iconColor}" stroke-width="2.3"
           stroke-linecap="round" stroke-linejoin="round"
           aria-hidden="true">
        <path d="M11 5 6 9H3v6h3l5 4V5Z" />
        <path d="M15.5 8.5a5 5 0 0 1 0 7" />
        <path d="M18.5 5.5a9 9 0 0 1 0 13" />
      </svg>
    ` : `
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
           stroke="${iconColor}" stroke-width="2.3"
           stroke-linecap="round" stroke-linejoin="round"
           aria-hidden="true">
        <path d="M11 5 6 9H3v6h3l5 4V5Z" />
        <path d="M17 9 22 14" />
        <path d="M22 9 17 14" />
      </svg>
    `;
  }

  private clearVoiceTimers(): void {
    if (this.voiceReadyTimer !== null) window.clearTimeout(this.voiceReadyTimer);
    if (this.voiceResponseTimer !== null) window.clearTimeout(this.voiceResponseTimer);
    this.voiceReadyTimer = null;
    this.voiceResponseTimer = null;
  }

  private startVoiceResponseWatchdog(): void {
    if (this.voiceResponseTimer !== null) window.clearTimeout(this.voiceResponseTimer);
    this.voiceResponseTimer = window.setTimeout(() => {
      this.voiceResponseTimer = null;
      const fallback = this.voiceLastInputText.trim();
      if (!fallback) {
        this.setVoiceHud("error", "Voice stalled", "Use typed Gemini chat or try the mic again.");
        return;
      }
      this.setVoiceHud("thinking", "Voice fallback", "Using the typed Gemini path for this turn.");
      void this.sendChatMessage(fallback, { recordUser: false, speak: this.audioEnabled });
      this.closeVoiceSocketAfterTurn();
    }, 22000);
  }

  private stopAudioPlayback(): void {
    window.speechSynthesis?.cancel();
    for (const source of this.audioSources) {
      try {
        source.stop();
      } catch (err) {
        // Source may already be stopped.
      }
    }
    this.audioSources = [];
    for (const element of this.audioElements) {
      try {
        element.pause();
        element.src = "";
      } catch (err) {
        // Element may already be detached.
      }
    }
    this.audioElements = [];
    this.audioChunkBuffers.clear();
    this.audioQueue = Promise.resolve();
    this.audioPlayTime = this.audioContext?.currentTime || 0;
  }

  private async toggleAudioOutput(): Promise<void> {
    this.audioEnabled = !this.audioEnabled;
    if (!this.audioEnabled) {
      this.stopAudioPlayback();
      this.setVoiceHud("idle", "Audio off", "Text and map actions still run.");
    } else {
      await this.unlockAudioOutput();
      this.setVoiceHud("idle", "Audio on", "Gemini Live audio responses are enabled.");
    }
    this.setAudioButton();
  }

  private ensureVoiceSocket(): Promise<WebSocket> {
    if (this.voiceSocket && this.voiceSocket.readyState === WebSocket.OPEN) {
      return Promise.resolve(this.voiceSocket);
    }

    return new Promise((resolve, reject) => {
      const socket = new WebSocket(this.getVoiceWsUrl());
      this.voiceSocket = socket;
      this.setVoiceHud("connecting", "Connecting to Gemini Live", "");
      if (this.voiceReadyTimer !== null) window.clearTimeout(this.voiceReadyTimer);
      this.voiceReadyTimer = window.setTimeout(() => {
        this.voiceReadyTimer = null;
        if (this.voiceSocket === socket && socket.readyState === WebSocket.OPEN) {
          this.setVoiceHud("error", "Voice setup is slow", "Typed Gemini chat is ready if voice stalls.");
        }
      }, 15000);

      socket.onopen = () => {
        this.setVoiceHud("idle", "Gemini Live voice ready", "Press the mic to ask a spoken question.");
        resolve(socket);
      };

      socket.onerror = () => {
        this.clearVoiceTimers();
        this.setVoiceHud("error", "Voice connection failed", "Typed Gemini chat is still available.");
        reject(new Error("Voice WebSocket failed"));
      };

      socket.onclose = () => {
        const graceful = this.voiceClosingAfterTurn;
        this.voiceClosingAfterTurn = false;
        if (this.voiceSocket === socket) this.voiceSocket = null;
        if (this.voiceListening) this.stopVoiceStreaming(false);
        this.completeVoiceTurn();
        this.clearVoiceTimers();
        this.setVoiceHud(
          "idle",
          graceful ? "Voice ready" : "Voice session closed",
          graceful ? "Press the mic for another question." : "Press the mic to reconnect.",
        );
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleVoiceMessage(message);
        } catch (err) {
          this.setVoiceHud("error", "Voice message parse error", "");
        }
      };
    });
  }

  private handleVoiceMessage(message: any): void {
    const incomingTurnId = Number(message.turn_id || 0);
    if (incomingTurnId > 0 && this.activeVoiceTurnId > 0 && incomingTurnId < this.activeVoiceTurnId) {
      return;
    }
    if (incomingTurnId > this.activeVoiceTurnId) {
      this.activeVoiceTurnId = incomingTurnId;
    }

    if (message.type === "turn_started") {
      this.activeVoiceTurnId = incomingTurnId || this.activeVoiceTurnId;
      this.voiceUserDraftIndex = null;
      this.voiceModelDraftIndex = null;
      return;
    }
    if (message.type === "turn_complete") {
      if (this.voiceResponseTimer !== null) window.clearTimeout(this.voiceResponseTimer);
      this.voiceResponseTimer = null;
      this.completeVoiceTurn();
      this.closeVoiceSocketAfterTurn();
      return;
    }
    if (message.type === "voice_state") {
      this.setVoiceHud(
        message.state || "idle",
        message.label || "Voice ready",
        message.detail || "",
      );
      return;
    }
    if (message.type === "ready") {
      if (this.voiceReadyTimer !== null) window.clearTimeout(this.voiceReadyTimer);
      this.voiceReadyTimer = null;
      this.setVoiceHud("idle", "Gemini Live voice ready", "Press the mic to ask a spoken question.");
      return;
    }
    if (message.type === "input_transcript" && message.text) {
      this.voiceLastInputText = String(message.text || "").trim() || this.voiceLastInputText;
      this.appendVoiceUserText(message.text, Boolean(message.final));
      this.setVoiceHud("listening", "Heard you", message.text);
      return;
    }
    if (message.type === "output_transcript" && message.text) {
      if (this.voiceResponseTimer !== null) window.clearTimeout(this.voiceResponseTimer);
      this.voiceResponseTimer = null;
      this.appendVoiceModelText(message.text, Boolean(message.final));
      this.setVoiceHud("replying", "Gemini replying", message.text);
      return;
    }
    if (message.type === "tool_result_text" && message.text) {
      if (this.voiceResponseTimer !== null) window.clearTimeout(this.voiceResponseTimer);
      this.voiceResponseTimer = null;
      this.appendVoiceModelText(message.text, true);
      this.setVoiceHud(
        "replying",
        this.audioEnabled ? "Gemini replying" : "Audio off",
        this.audioEnabled ? "Waiting for native Gemini Live audio." : "Showing the grounded response without playback.",
      );
      return;
    }
    if (message.type === "audio" && message.data) {
      if (this.voiceResponseTimer !== null) window.clearTimeout(this.voiceResponseTimer);
      this.voiceResponseTimer = null;
      this.enqueuePcmAudio(
        message.data,
        message.mime_type || "audio/pcm;rate=24000",
      );
      this.setVoiceHud("replying", "Gemini replying", "Native Gemini Live audio is playing.");
      return;
    }
    if (message.type === "audio_chunk" && message.data) {
      if (this.voiceResponseTimer !== null) window.clearTimeout(this.voiceResponseTimer);
      this.voiceResponseTimer = null;
      const id = String(message.id || "voice-audio");
      const total = Number(message.total || 1);
      const index = Number(message.index || 0);
      const buffer = this.audioChunkBuffers.get(id) || {
        mimeType: message.mime_type || "audio/L16;codec=pcm;rate=24000",
        source: message.source || "gemini-live",
        total,
        chunks: new Array(total).fill(""),
      };
      buffer.chunks[index] = message.data;
      this.audioChunkBuffers.set(id, buffer);
      if (buffer.chunks.every(Boolean)) {
        this.audioChunkBuffers.delete(id);
        this.enqueuePcmAudio(
          buffer.chunks.join(""),
          buffer.mimeType,
        );
      }
      this.setVoiceHud("replying", "Gemini replying", "Native Gemini Live audio is streaming.");
      return;
    }
    if (message.type === "tool_calls" && Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls as ChatToolCall[]) {
        this.dispatchToolCall(call);
      }
      this.setVoiceHud("tool", this.voiceToolLabel(message.tool_calls as ChatToolCall[]), "Map and data tools are running.");
      return;
    }
    if (message.type === "interrupted") {
      this.stopAudioPlayback();
      this.setVoiceHud("interrupted", "Interrupted", "Listening for your next question.");
      return;
    }
    if (message.type === "error") {
      this.setVoiceHud("error", "Voice error", message.message || "unknown");
    }
  }

  private beginVoiceTurn(): number {
    this.voiceTurnId += 1;
    this.activeVoiceTurnId = this.voiceTurnId;
    this.voiceUserDraftIndex = null;
    this.voiceModelDraftIndex = null;
    this.audioChunkBuffers.clear();
    this.voiceLastInputText = "";
    if (this.voiceResponseTimer !== null) window.clearTimeout(this.voiceResponseTimer);
    this.voiceResponseTimer = null;
    return this.activeVoiceTurnId;
  }

  private completeVoiceTurn(): void {
    this.voiceUserDraftIndex = null;
    this.voiceModelDraftIndex = null;
  }

  private closeVoiceSocketAfterTurn(): void {
    if (!this.voiceSocket || this.voiceSocket.readyState !== WebSocket.OPEN) return;
    this.voiceClosingAfterTurn = true;
    try {
      this.voiceSocket.send(JSON.stringify({ type: "close" }));
    } catch {}
    this.voiceSocket.close(1000, "turn complete");
  }

  private voiceToolLabel(calls: ChatToolCall[]): string {
    const first = calls[0]?.name || "query_data";
    if (first === "select_hub" || first === "zoom_to_hub") return "Selecting hub";
    if (first === "filter_to_paralympic") return "Filtering Hot Spots";
    if (first === "select_state") return "Opening state";
    if (first === "focus_hometown") return "Opening hometown";
    if (first === "explain_map") return "Explaining map";
    if (first === "reset_view") return "Resetting map";
    return "Checking rankings";
  }

  private appendVoiceUserText(text: string, final: boolean = false): void {
    const clean = String(text || "").trim();
    if (!clean) return;
    if (this.voiceUserDraftIndex === null || !this.chatHistory[this.voiceUserDraftIndex]) {
      this.chatHistory.push({ role: "user", text: clean });
      this.voiceUserDraftIndex = this.chatHistory.length - 1;
    } else {
      const current = this.chatHistory[this.voiceUserDraftIndex].text;
      this.chatHistory[this.voiceUserDraftIndex].text = final
        ? clean
        : current.endsWith(clean) ? current : `${current}${clean.startsWith(" ") ? "" : " "}${clean}`.trim();
    }
    if (final) this.voiceUserDraftIndex = null;
    this.renderChatBody();
  }

  private appendVoiceModelText(text: string, final: boolean = false): void {
    const clean = String(text || "").trim();
    if (!clean) return;
    if (this.voiceModelDraftIndex !== null && this.chatHistory[this.voiceModelDraftIndex]) {
      const current = this.chatHistory[this.voiceModelDraftIndex].text;
      this.chatHistory[this.voiceModelDraftIndex].text = final
        ? (current.includes(clean) ? current : clean)
        : current.endsWith(clean) ? current : `${current}${clean.startsWith(" ") ? "" : " "}${clean}`.trim();
    } else {
      this.chatHistory.push({ role: "model", text: clean });
      this.voiceModelDraftIndex = this.chatHistory.length - 1;
    }
    if (final) this.voiceModelDraftIndex = null;
    this.renderChatBody();
  }

  private async toggleVoiceInput(): Promise<void> {
    if (this.voiceListening) {
      this.stopVoiceStreaming(true);
      return;
    }
    await this.startVoiceStreaming();
  }

  private async startVoiceStreaming(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      this.setVoiceHud("error", "Microphone unavailable", "Use typed Gemini chat in this browser.");
      return;
    }
    try {
      this.stopAudioPlayback();
      if (this.audioEnabled) {
        await this.unlockAudioOutput();
      }
      const socket = await this.ensureVoiceSocket();
      const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextCtor) throw new Error("AudioContext unavailable");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      this.voiceInputStream = stream;
      this.voiceInputContext = new AudioContextCtor();
      if (this.voiceInputContext.state === "suspended") {
        await this.voiceInputContext.resume();
      }
      this.voiceInputSampleRate = this.voiceInputContext.sampleRate;
      this.voiceInputSource = this.voiceInputContext.createMediaStreamSource(stream);
      this.voiceInputBuffer = [];
      this.voiceBufferedSamples = 0;
      const turnId = this.beginVoiceTurn();
      socket.send(JSON.stringify({
        type: "audio_start",
        turn_id: turnId,
        audio_enabled: this.audioEnabled,
      }));

      if (this.voiceInputContext.audioWorklet) {
        const workletCode = `
          class HSMVoiceProcessor extends AudioWorkletProcessor {
            process(inputs) {
              const input = inputs[0];
              if (input && input[0]) this.port.postMessage(input[0].slice(0));
              return true;
            }
          }
          registerProcessor("hsm-voice-processor", HSMVoiceProcessor);
        `;
        this.voiceWorkletUrl = URL.createObjectURL(new Blob([workletCode], { type: "application/javascript" }));
        await this.voiceInputContext.audioWorklet.addModule(this.voiceWorkletUrl);
        const node = new AudioWorkletNode(this.voiceInputContext, "hsm-voice-processor", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [1],
        });
        node.port.onmessage = (event) => {
          this.handleVoiceInputSamples(event.data as Float32Array, this.voiceInputSampleRate);
        };
        const sink = this.voiceInputContext.createGain();
        sink.gain.value = 0;
        this.voiceInputSource.connect(node);
        node.connect(sink);
        sink.connect(this.voiceInputContext.destination);
        this.voiceInputNode = node;
      } else {
        const processor = this.voiceInputContext.createScriptProcessor(2048, 1, 1);
        processor.onaudioprocess = (event) => {
          this.handleVoiceInputSamples(event.inputBuffer.getChannelData(0), this.voiceInputSampleRate);
        };
        this.voiceInputSource.connect(processor);
        processor.connect(this.voiceInputContext.destination);
        this.voiceInputNode = processor;
      }

      this.voiceListening = true;
      this.setVoiceButton(true);
      this.setVoiceHud("listening", "Listening", "Speak naturally. Press the mic again to send.");
    } catch (err) {
      this.stopVoiceStreaming(false);
      this.setVoiceHud("error", "Voice input failed", err instanceof Error ? err.message : "Use typed Gemini chat.");
    }
  }

  private stopVoiceStreaming(sendEnd: boolean): void {
    this.flushVoiceInputBuffer(true);
    if (sendEnd && this.voiceSocket?.readyState === WebSocket.OPEN) {
      this.voiceSocket.send(JSON.stringify({
        type: "audio_end",
        turn_id: this.activeVoiceTurnId,
        audio_enabled: this.audioEnabled,
      }));
      this.setVoiceHud("thinking", "Gemini Live is thinking", "Audio turn ended.");
      this.startVoiceResponseWatchdog();
    }

    try { this.voiceInputNode?.disconnect(); } catch {}
    try { this.voiceInputSource?.disconnect(); } catch {}
    this.voiceInputStream?.getTracks().forEach(track => track.stop());
    if (this.voiceInputContext) void this.voiceInputContext.close();
    if (this.voiceWorkletUrl) URL.revokeObjectURL(this.voiceWorkletUrl);
    this.voiceInputStream = null;
    this.voiceInputContext = null;
    this.voiceInputSource = null;
    this.voiceInputNode = null;
    this.voiceWorkletUrl = null;
    this.voiceInputBuffer = [];
    this.voiceBufferedSamples = 0;
    this.voiceListening = false;
    this.setVoiceButton(false);
  }

  private handleVoiceInputSamples(samples: Float32Array, sampleRate: number): void {
    if (!this.voiceListening || this.voiceSocket?.readyState !== WebSocket.OPEN) return;
    const copy = new Float32Array(samples);
    this.voiceInputBuffer.push(copy);
    this.voiceBufferedSamples += copy.length;
    const targetSamples = Math.floor(sampleRate * 0.04);
    if (this.voiceBufferedSamples >= targetSamples) {
      this.flushVoiceInputBuffer(false);
    }
  }

  private flushVoiceInputBuffer(force: boolean): void {
    if (!this.voiceInputBuffer.length || this.voiceSocket?.readyState !== WebSocket.OPEN) return;
    if (!force && this.voiceBufferedSamples < this.voiceInputSampleRate * 0.02) return;
    const merged = new Float32Array(this.voiceBufferedSamples);
    let offset = 0;
    for (const chunk of this.voiceInputBuffer) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.voiceInputBuffer = [];
    this.voiceBufferedSamples = 0;
    if (this.voiceSocket.bufferedAmount > 1_500_000) {
      this.setVoiceHud("listening", "Network catching up", "Skipping microphone audio for a moment.");
      return;
    }
    const pcm16 = this.downsampleToPcm16(merged, this.voiceInputSampleRate, 16000);
    const data = this.uint8ToBase64(new Uint8Array(pcm16.buffer));
    this.voiceSocket.send(JSON.stringify({
      type: "audio_chunk",
      turn_id: this.activeVoiceTurnId,
      data,
      mime_type: "audio/pcm;rate=16000",
      audio_enabled: this.audioEnabled,
    }));
  }

  private downsampleToPcm16(samples: Float32Array, inputRate: number, outputRate: number): Int16Array {
    if (inputRate === outputRate) {
      const out = new Int16Array(samples.length);
      for (let i = 0; i < samples.length; i += 1) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      return out;
    }
    const ratio = inputRate / outputRate;
    const outputLength = Math.max(1, Math.floor(samples.length / ratio));
    const out = new Int16Array(outputLength);
    for (let i = 0; i < outputLength; i += 1) {
      const start = Math.floor(i * ratio);
      const end = Math.min(samples.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = start; j < end; j += 1) sum += samples[j];
      const sample = sum / Math.max(1, end - start);
      const s = Math.max(-1, Math.min(1, sample));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  private uint8ToBase64(bytes: Uint8Array): string {
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode(...chunk);
    }
    return btoa(binary);
  }

  private speakText(text: string): void {
    if (!this.audioEnabled) return;
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text.replace(/\s+/g, " ").trim());
    utterance.rate = 1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  }

  private async unlockAudioOutput(): Promise<void> {
    if (!this.audioEnabled) return;
    const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextCtor) return;
    if (!this.audioContext) {
      this.audioContext = new AudioContextCtor();
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
    const buffer = this.audioContext.createBuffer(1, 1, this.audioContext.sampleRate);
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);
    source.start();
    this.audioPlayTime = this.audioContext.currentTime;
  }

  private parsePcmSampleRate(mimeType: string): number {
    const rateMatch = /rate=(\d+)/.exec(mimeType);
    return rateMatch ? Number(rateMatch[1]) : 24000;
  }

  private base64ToBytes(base64: string): Uint8Array {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  private wavBlobUrlFromPcm(base64: string, sampleRate: number): string {
    const pcm = this.base64ToBytes(base64);
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    const writeString = (offset: number, value: string) => {
      for (let i = 0; i < value.length; i += 1) {
        view.setUint8(offset + i, value.charCodeAt(i));
      }
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + pcm.byteLength, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, pcm.byteLength, true);
    const pcmBuffer = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength) as ArrayBuffer;
    const blob = new Blob([header, pcmBuffer], { type: "audio/wav" });
    return URL.createObjectURL(blob);
  }

  private async playWavElement(base64: string, sampleRate: number): Promise<void> {
    const url = this.wavBlobUrlFromPcm(base64, sampleRate);
    const audio = new Audio(url);
    audio.preload = "auto";
    this.audioElements.push(audio);
    audio.onended = () => {
      URL.revokeObjectURL(url);
      this.audioElements = this.audioElements.filter((item) => item !== audio);
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      this.audioElements = this.audioElements.filter((item) => item !== audio);
    };
    await audio.play();
  }

  private handleAudioPlaybackError(err: unknown): void {
    const message = err instanceof Error ? err.message : String(err || "unknown playback error");
    this.setVoiceHud("error", "Audio blocked", message.slice(0, 120));
  }

  private enqueuePcmAudio(base64: string, mimeType: string): void {
    this.audioQueue = this.audioQueue
      .catch(() => undefined)
      .then(() => this.playPcmAudio(base64, mimeType))
      .catch((err) => this.handleAudioPlaybackError(err));
  }

  private async playPcmAudio(base64: string, mimeType: string): Promise<void> {
    if (!this.audioEnabled) return;
    const sampleRate = this.parsePcmSampleRate(mimeType);
    const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextCtor) {
      await this.playWavElement(base64, sampleRate);
      return;
    }
    if (!this.audioContext) {
      this.audioContext = new AudioContextCtor();
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }

    const binary = atob(base64);
    const sampleCount = Math.floor(binary.length / 2);
    const buffer = this.audioContext.createBuffer(1, sampleCount, sampleRate);
    const channel = buffer.getChannelData(0);

    for (let i = 0; i < sampleCount; i += 1) {
      const lo = binary.charCodeAt(i * 2);
      const hi = binary.charCodeAt(i * 2 + 1);
      const sample = (hi << 8) | lo;
      const signed = sample >= 0x8000 ? sample - 0x10000 : sample;
      channel[i] = signed / 0x8000;
    }

    const bufferSource = this.audioContext.createBufferSource();
    bufferSource.buffer = buffer;
    bufferSource.connect(this.audioContext.destination);
    this.audioSources.push(bufferSource);
    bufferSource.onended = () => {
      this.audioSources = this.audioSources.filter(s => s !== bufferSource);
    };
    const startAt = Math.max(this.audioContext.currentTime + 0.02, this.audioPlayTime);
    bufferSource.start(startAt);
    this.audioPlayTime = startAt + buffer.duration;
  }

  private toggleChat(): void {
    this.chatOpen = !this.chatOpen;
    const panel = this.querySelector("#hubmap-chat-panel") as HTMLElement;
    const pill = this.querySelector("#hubmap-chat-pill") as HTMLElement;
    const statePanel = this.querySelector("#hubmap-state-panel") as HTMLElement;
    if (panel) {
      panel.style.display = this.chatOpen ? "flex" : "none";
      if (this.chatOpen) {
        panel.classList.remove("hsm-chat-open");
        void panel.offsetWidth;
        panel.classList.add("hsm-chat-open");
      }
    }
    if (pill) pill.style.display = this.chatOpen ? "none" : "flex";
    if (statePanel && this.chatOpen) {
      statePanel.style.display = "none";
    } else if (statePanel && !this.chatOpen && this.selectedStateCode) {
      this.renderStatePanel();
    }
    if (this.chatOpen) {
      const input = this.querySelector("#hubmap-chat-input") as HTMLInputElement;
      input?.focus();
    }
  }

  private wireChatUI(): void {
    const pill = this.querySelector("#hubmap-chat-pill");
    if (pill) {
      pill.addEventListener("click", () => this.toggleChat());
    }

    const closeBtn = this.querySelector("#hubmap-chat-close");
    if (closeBtn) {
      closeBtn.addEventListener("click", () => this.toggleChat());
    }

    const voiceBtn = this.querySelector("#hubmap-voice-btn");
    if (voiceBtn) {
      voiceBtn.addEventListener("click", () => this.toggleVoiceInput());
    }
    const audioBtn = this.querySelector("#hubmap-audio-btn");
    if (audioBtn) {
      audioBtn.addEventListener("click", () => this.toggleAudioOutput());
      this.setAudioButton();
    }

    const form = this.querySelector("#hubmap-chat-form") as HTMLFormElement;
    if (form) {
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const input = this.querySelector("#hubmap-chat-input") as HTMLInputElement;
        if (input?.value) {
          const msg = input.value;
          input.value = "";
          void this.sendChatMessage(msg);
        }
      });
    }

    this.renderChatBody();
  }

  private renderChatBody(): void {
    const body = this.querySelector("#hubmap-chat-body") as HTMLElement;
    if (!body) return;

    const suggestionPills = `
      <div style="padding: 12px; border-bottom: 1px solid #efeae6;
            background: #ffffff; display: flex; flex-wrap: wrap; gap: 6px;">
        ${SUGGESTED_PROMPTS.map(p => `
          <button class="hubmap-chat-suggestion" type="button"
            style="background: #efeae6; border: 1px solid #b9bfd2;
                   border-radius: 14px; padding: 5px 10px;
                   font-size: 11px; color: #152969; cursor: pointer;
                   font-family: system-ui, -apple-system, sans-serif;
                   transition: background 0.15s;"
            onmouseover="this.style.background='#d7d3cf';"
            onmouseout="this.style.background='#efeae6';">
            ${p}
          </button>
        `).join("")}
      </div>
    `;

    if (this.chatHistory.length === 0) {
      body.innerHTML = `
        ${suggestionPills}
        <div style="padding: 16px; color: #484645;">
          <div style="font-size: 13px; line-height: 1.55;">
            Hi! I'm <strong>Gemini</strong>, your guide to the Hometown Success Engine. Ask about a hub, state, sport, ranking, or Paralympic Hot Spot, and I can answer with grounded data or move the map for you.
          </div>
        </div>
      `;
    } else {
      const turns = this.chatHistory.map(turn => {
        const isUser = turn.role === "user";
        return `
          <div style="display: flex;
                justify-content: ${isUser ? "flex-end" : "flex-start"};
                margin-bottom: 8px;">
            <div style="max-width: 82%;
                  background: ${isUser ? "#152969" : "#efeae6"};
                  color: ${isUser ? "#ffffff" : "#484645"};
                  padding: 8px 12px; border-radius: 12px;
                  font-size: 13px; line-height: 1.5;
                  ${isUser ? "border-bottom-right-radius: 4px;" : "border-bottom-left-radius: 4px;"}">
              ${this.escapeHtml(turn.text)}
            </div>
          </div>
        `;
      }).join("");

      const loading = this.chatLoading
        ? `
          <div style="display: flex; justify-content: flex-start;
                margin-bottom: 8px;">
            <div style="background: #efeae6; color: #b9bfd2;
                  padding: 8px 12px; border-radius: 12px;
                  border-bottom-left-radius: 4px;
                  font-size: 13px; font-style: italic;">
              Thinking...
            </div>
          </div>
        `
        : "";

      body.innerHTML = `
        ${suggestionPills}
        <div style="padding: 12px;">
          ${turns}${loading}
        </div>
      `;
      body.scrollTop = body.scrollHeight;
    }

    body.querySelectorAll(".hubmap-chat-suggestion").forEach((el) => {
      el.addEventListener("click", () => {
        const prompt = el.textContent?.trim() || "";
        if (prompt) void this.sendChatMessage(prompt);
      });
    });
  }

  private escapeHtml(s: string): string {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  private wireResetButton(): void {
    const resetBtn = this.querySelector("#hubmap-reset-btn");
    if (resetBtn) {
      resetBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.resetView();
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
      zoomControl: true,
      zoomControlOptions: {
        position: google.maps.ControlPosition.TOP_LEFT,
      },
      rotateControl: true,
      rotateControlOptions: {
        position: google.maps.ControlPosition.TOP_LEFT,
      },
      cameraControl: true,
      cameraControlOptions: {
        position: google.maps.ControlPosition.TOP_LEFT,
      },
      mapTypeControl: false,
      streetViewControl: false,
      fullscreenControl: false,
      keyboardShortcuts: true,
      clickableIcons: false,
      gestureHandling: "greedy",
      backgroundColor: "#ffffff",
      minZoom: 3,
      maxZoom: 12,
      restriction: {
        latLngBounds: { north: 72, south: 17, west: -180, east: -60 },
        strictBounds: false,
      },
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
        getLineColor: (f: any) => {
          const stateName = f.properties?.name || "";
          const code = STATE_NAME_TO_CODE[stateName];
          if (code === this.selectedStateCode) return [255, 200, 50, 255];
          return [255, 255, 255, 255];
        },
        getLineWidth: (f: any) => {
          const stateName = f.properties?.name || "";
          const code = STATE_NAME_TO_CODE[stateName];
          return code === this.selectedStateCode ? 3 : 1;
        },
        lineWidthUnits: "pixels",
        stroked: true,
        filled: true,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 40],
        onClick: (info: any) => {
          if (info.object?.properties?.name) {
            const name = info.object.properties.name;
            const code = STATE_NAME_TO_CODE[name];
            if (code) {
              this.selectedHometown = null;
              this.selectedStateCode = code;
              this.selectedStateName = name;
              this.renderHometownPanel();
              this.renderStatePanel();
              this.updateLayers();
              this.centerOnState(code);
            }
          }
        },
        updateTriggers: {
          getLineColor: [this.selectedStateCode],
          getLineWidth: [this.selectedStateCode],
        },
      }));
    }

    const visibleAthletes = this.selectedStateCode
      ? this.athletes.filter(a => a.state === this.selectedStateCode)
      : this.athletes;

    if (visibleAthletes.length > 0) {
      layers.push(new ScatterplotLayer({
        id: "athlete-constellation",
        data: visibleAthletes,
        getPosition: (a: AthleteGeoPoint) => [a.lon, a.lat],
        getRadius: 1500,
        radiusUnits: "meters",
        radiusMinPixels: 1.5,
        radiusMaxPixels: 3,
        getFillColor: (a: AthleteGeoPoint) => {
          const paraFilter = state.filters?.paralympic_focus === true;
          const isPara = a.status === "paralympic" || a.status === "both";
          if (paraFilter) {
            return isPara ? [211, 17, 24, 220] : [21, 41, 105, 30];
          }
          return isPara ? [211, 17, 24, 180] : [21, 41, 105, 100];
        },
        stroked: false,
        filled: true,
        pickable: false,
        updateTriggers: {
          getFillColor: [state.filters?.paralympic_focus],
        },
      }));
    }

    if (this.selectedHometown?.lat !== undefined && this.selectedHometown?.lon !== undefined) {
      layers.push(new ScatterplotLayer({
        id: "hometown-focus",
        data: [this.selectedHometown],
        getPosition: (h: HometownFocus) => [Number(h.lon), Number(h.lat)],
        getRadius: 8000,
        radiusUnits: "meters",
        radiusMinPixels: 8,
        radiusMaxPixels: 16,
        getFillColor: [255, 200, 50, 230],
        getLineColor: [21, 41, 105, 255],
        getLineWidth: 2,
        lineWidthUnits: "pixels",
        stroked: true,
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
        if (h.hub_id === state.selectedHubId) {
          return base * (h.is_paralympic_hot_spot ? 1.5 : 1.3);
        }
        if (h.hub_id === this.hoveredHubId) {
          return base * (h.is_paralympic_hot_spot ? 1.5 : 1.25);
        }
        return base;
      },
      radiusUnits: "meters",
      radiusMinPixels: 10,
      radiusMaxPixels: 50,
      getFillColor: (h: Hub) => {
        const paraFilter = state.filters?.paralympic_focus === true;
        if (paraFilter && !h.is_paralympic_hot_spot) return [21, 41, 105, 60];
        if (h.is_paralympic_hot_spot) return [211, 17, 24, 230];
        return [21, 41, 105, 230];
      },
      getLineColor: (h: Hub) => {
        const paraFilter = state.filters?.paralympic_focus === true;
        if (h.hub_id === state.selectedHubId) {
          // Gold ring for selected distinct from hub fill colors
          return [255, 200, 50, 255];
        }
        if (paraFilter && !h.is_paralympic_hot_spot) return [255, 255, 255, 80];
        return [255, 255, 255, 255];
      },
      getLineWidth: (h: Hub) => {
        if (h.hub_id === state.selectedHubId) return 6;
        if (h.hub_id === this.hoveredHubId) return 4;
        return h.is_paralympic_hot_spot ? 3 : 2;
      },
      lineWidthUnits: "pixels",
      stroked: true,
      filled: true,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 60],
      onHover: (info: any) => {
        const canvas = this.querySelector("#hubmap-canvas") as HTMLElement;
        if (canvas) {
          canvas.style.cursor = info.object ? "pointer" : "grab";
        }
        const newHoveredId = info.object ? info.object.hub_id : null;
        if (newHoveredId !== this.hoveredHubId) {
          this.hoveredHubId = newHoveredId;
          this.updateLayers();
        }
      },
      onClick: (info: any) => {
        if (info.object) this.selectHub(info.object.hub_id);
      },
      updateTriggers: {
        getRadius: [state.selectedHubId, this.hoveredHubId],
        getFillColor: [state.selectedHubId, state.filters?.paralympic_focus],
        getLineColor: [state.selectedHubId, state.filters?.paralympic_focus],
        getLineWidth: [state.selectedHubId, this.hoveredHubId],
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
  ? `<div style="color: #d31118; font-weight: 700; font-size: 10px; letter-spacing: 1px; margin-top: 4px;">PARALYMPIC HOT SPOT</div>`
  : "";
          return {
            html: `
              <div style="font-family: system-ui, sans-serif;">
                <div style="font-weight: 700; font-size: 14px; color: #152969;">${hub.display_name}</div>
                <div style="font-size: 12px; color: #171fbe; margin-top: 2px;">${hub.region_name}</div>
                <div style="font-size: 11px; color: #484645; margin-top: 4px;">
                  ${hub.total_athletes} athletes ${paraPct}% Paralympic
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
      <style>
        hometown-hub-map .hsm-stats-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
        }
        hometown-hub-map .hsm-stat-card {
          background: #ffffff;
          padding: 18px 20px;
          border-radius: 8px;
          border: 1px solid #b9bfd2;
          border-left: 4px solid #171fbe;
          transition: transform 0.15s, box-shadow 0.15s;
        }
        hometown-hub-map .hsm-stat-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        }
        hometown-hub-map .hsm-stat-card.hsm-stat-hotspot {
          border-left-color: #d31118;
        }
        hometown-hub-map .hsm-stat-label {
          color: #484645;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 1.2px;
          font-weight: 600;
          margin-bottom: 8px;
          line-height: 1.3;
        }
        hometown-hub-map .hsm-stat-value {
          color: #152969;
          font-size: 32px;
          font-weight: 800;
          line-height: 1;
          margin-bottom: 4px;
        }
        hometown-hub-map .hsm-stat-sub {
          color: #6b6b6b;
          font-size: 11px;
          line-height: 1.3;
        }
        @keyframes hsm-chat-slide-up {
          from {
            opacity: 0;
            transform: translateY(20px) scale(0.96);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
        hometown-hub-map .hsm-chat-panel.hsm-chat-open {
          animation: hsm-chat-slide-up 0.22s cubic-bezier(0.2, 0.8, 0.2, 1);
          transform-origin: bottom right;
        }
      

        @media (max-width: 1024px) {
          hometown-hub-map .hsm-map-area { height: 520px !important; }
          hometown-hub-map .hsm-stats-grid {
            grid-template-columns: repeat(2, 1fr) !important;
          }
        }

        @media (max-width: 640px) {
          hometown-hub-map .hsm-map-area { height: 480px !important; }
          hometown-hub-map .hsm-header-title { font-size: 16px !important; gap: 8px !important; }
          hometown-hub-map .hsm-header-title svg { width: 26px !important; height: 14px !important; }
          hometown-hub-map .hsm-header-tag { display: none !important; }
          hometown-hub-map .hsm-header-sub { font-size: 12px !important; }
          hometown-hub-map .hsm-insets-wrapper {
            bottom: 28px !important;
            left: 8px !important;
            gap: 6px !important;
          }
          hometown-hub-map .hsm-color-legend { display: none !important; }
          hometown-hub-map .hsm-insets-row {
            padding: 4px !important;
            gap: 4px !important;
          }
          hometown-hub-map .hsm-insets-row svg {
            width: 70px !important;
            height: 50px !important;
          }
          hometown-hub-map .hsm-chat-panel {
            width: calc(100vw - 24px) !important;
            max-width: 420px;
            height: 78vh !important;
            max-height: 580px;
            right: 12px !important;
            bottom: 28px !important;
          }
          hometown-hub-map .hsm-chat-pill {
            bottom: 28px !important;
            right: 12px !important;
            font-size: 13px !important;
            padding: 9px 14px !important;
          }
          hometown-hub-map .hsm-stats-grid {
            grid-template-columns: 1fr 1fr !important;
            gap: 8px !important;
          }
          hometown-hub-map .hsm-stat-card {
            padding: 12px 14px !important;
          }
          hometown-hub-map .hsm-stat-value { font-size: 26px !important; }
          hometown-hub-map .hsm-stat-label { font-size: 10px !important; letter-spacing: 0.8px !important; }
          hometown-hub-map .hsm-stats-section { padding: 16px !important; }
          hometown-hub-map .hsm-narrative-section { padding: 16px !important; }
        }
      </style>

      <div style="display: flex; flex-direction: column;
            background: #ffffff; font-family: system-ui, -apple-system, sans-serif;
            position: relative;">

        <header style="background: #152969; color: #ffffff; padding: 16px 24px;">
          <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px;">
            <div class="hsm-header-title" style="font-size: 20px; font-weight: 700; letter-spacing: 0.5px; display: flex; align-items: center; gap: 10px;">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 32" width="32" height="17" style="border-radius: 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); flex-shrink: 0;" aria-label="United States flag" role="img">
        <rect width="60" height="32" fill="#B22234"/>
        <path d="M0,3.692h60m-60,4.923h60m-60,4.923h60m-60,4.923h60m-60,4.923h60m-60,4.923h60" stroke="#FFFFFF" stroke-width="2.4615"/>
        <rect width="24" height="17.231" fill="#3C3B6E"/>
        <defs>
          <polygon id="s" fill="#FFFFFF" points="0,-0.8 0.18,-0.25 0.76,-0.25 0.29,0.09 0.47,0.65 0,0.31 -0.47,0.65 -0.29,0.09 -0.76,-0.25 -0.18,-0.25"/>
          <g id="r6"><use href="#s" x="2"/><use href="#s" x="6"/><use href="#s" x="10"/><use href="#s" x="14"/><use href="#s" x="18"/><use href="#s" x="22"/></g>
          <g id="r5"><use href="#s" x="4"/><use href="#s" x="8"/><use href="#s" x="12"/><use href="#s" x="16"/><use href="#s" x="20"/></g>
        </defs>
        <use href="#r6" y="1.723"/>
        <use href="#r5" y="3.446"/>
        <use href="#r6" y="5.169"/>
        <use href="#r5" y="6.892"/>
        <use href="#r6" y="8.615"/>
        <use href="#r5" y="10.339"/>
        <use href="#r6" y="12.062"/>
        <use href="#r5" y="13.785"/>
        <use href="#r6" y="15.508"/>
      </svg>
      <span>Hometown Success Engine</span>
    </div>
            <div class="hsm-header-tag" style="font-size: 13px; color: #b9bfd2;
                  text-transform: uppercase; letter-spacing: 1px;">
              Team USA Hometown Hub Map
            </div>
          </div>
          <div class="hsm-header-sub" style="font-size: 14px; color: #b9bfd2;
                margin-top: 6px; font-weight: 400; letter-spacing: 0.3px;">
            Mapping 5,119 Olympians and Paralympians across 40 Team USA hometown hubs from Tokyo 2020 through Milan-Cortina 2026.
          </div>
        </header>

        <div class="hsm-map-area" style="position: relative; width: 100%; height: 600px;">
          <div id="hubmap-canvas"
              style="width: 100%; height: 100%; background: #ffffff;"></div>

          <div id="hubmap-insets-wrapper"
              class="hsm-insets-wrapper"
              style="position: absolute; bottom: 32px; left: 12px;
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
              <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2"
                   stroke-linecap="round" stroke-linejoin="round"
                   style="display: block; flex: 0 0 auto;">
                <circle cx="12" cy="12" r="7" />
                <circle cx="12" cy="12" r="2" />
                <path d="M12 3v2" />
                <path d="M12 19v2" />
                <path d="M3 12h2" />
                <path d="M19 12h2" />
              </svg>
              Reset View
            </button>
            <div id="hubmap-color-legend"
                class="hsm-color-legend"
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
                  Paralympian
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
                class="hsm-insets-row"
                style="display: flex; gap: 8px;
                   background: rgba(255, 255, 255, 0.95);
                   border: 1px solid #b9bfd2;
                   border-radius: 6px; padding: 8px;
                   box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);">
            </div>
          </div>

          <button id="hubmap-chat-pill"
                  class="hsm-chat-pill"
                  type="button"
                  aria-label="Ask Gemini"
                  style="position: absolute; bottom: 32px; right: 20px;
                     background: #ffffff; color: #171fbe;
                     border: 1px solid #b9bfd2; border-radius: 22px;
                     padding: 11px 18px;
                     font-size: 14px; font-weight: 700;
                     letter-spacing: 0.3px;
                     font-family: system-ui, -apple-system, sans-serif;
                     cursor: pointer;
                     display: flex; align-items: center; gap: 8px;
                     box-shadow: 0 4px 14px rgba(21, 41, 105, 0.18);
                     transition: box-shadow 0.15s, background 0.15s;
                     z-index: 101;"
                  onmouseover="this.style.background='#f5f3f0'; this.style.boxShadow='0 6px 18px rgba(21, 41, 105, 0.25)';"
                  onmouseout="this.style.background='#ffffff'; this.style.boxShadow='0 4px 14px rgba(21, 41, 105, 0.18)';">
            Ask Gemini
          </button>

          <div id="hubmap-chat-panel"
              class="hsm-chat-panel"
              style="position: absolute; bottom: 32px; right: 20px;
                 width: 420px; height: 580px;
                 background: #ffffff;
                 border: 1px solid #b9bfd2;
                 border-radius: 12px;
                 box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
                 display: none;
                 flex-direction: column;
                 z-index: 99;
                 overflow: hidden;">
            <div style="background: linear-gradient(90deg, #4285F4 0%, #9B72CB 50%, #D96570 100%);
                  color: #ffffff;
                  padding: 6px 16px;
                  display: flex; align-items: center; gap: 8px;
                  font-size: 11px; font-weight: 600;
                  letter-spacing: 0.8px; text-transform: uppercase;">
              Powered by Google Gemini
            </div>
            <div style="background: #152969; color: #ffffff;
                  padding: 12px 16px;
                  display: flex; align-items: center; justify-content: space-between;">
              <div>
                <div style="font-size: 14px; font-weight: 700;">
                  Ask Gemini
                </div>
                <div style="font-size: 11px; color: #b9bfd2;
                      margin-top: 1px;">
                  Data guide for the map
                </div>
              </div>
              <button id="hubmap-chat-close"
                      type="button"
                      aria-label="Close chat"
                      style="background: transparent; border: none;
                         color: #ffffff; cursor: pointer; padding: 4px;
                         display: flex; align-items: center;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                     stroke="#ffffff" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <div id="hubmap-chat-body"
                style="flex: 1; overflow-y: auto; background: #ffffff;">
            </div>
            <div id="hubmap-voice-hud"
                data-state="idle"
                style="padding: 8px 12px; border-top: 1px solid #efeae6;
                   background: #f8f7f5; color: #484645;
                   font-size: 11px; line-height: 1.35;
                   display: flex; align-items: center; gap: 10px;
                   transition: background 140ms ease, border-color 140ms ease;">
              <div id="hubmap-voice-dot"
                   style="width: 9px; height: 9px; border-radius: 50%;
                      background: #7886ac; flex: 0 0 auto;"></div>
              <div style="min-width: 0; flex: 1;">
                <div id="hubmap-voice-status"
                     style="color: #152969; font-weight: 700; font-size: 11px;">
                  Voice ready
                </div>
                <div id="hubmap-voice-detail"
                     style="color: #484645; margin-top: 1px; overflow: hidden;
                        text-overflow: ellipsis; white-space: nowrap;">
                  Press the mic to ask Gemini with voice.
                </div>
              </div>
              <div id="hubmap-voice-wave"
                   aria-hidden="true"
                   style="height: 18px; width: 34px; display: flex;
                      align-items: center; justify-content: flex-end; gap: 3px;
                      flex: 0 0 auto;">
                <span style="display: block; width: 3px; height: 5px; border-radius: 2px; background: #7886ac; opacity: 0.25;"></span>
                <span style="display: block; width: 3px; height: 5px; border-radius: 2px; background: #7886ac; opacity: 0.25;"></span>
                <span style="display: block; width: 3px; height: 5px; border-radius: 2px; background: #7886ac; opacity: 0.25;"></span>
                <span style="display: block; width: 3px; height: 5px; border-radius: 2px; background: #7886ac; opacity: 0.25;"></span>
              </div>
            </div>
            <form id="hubmap-chat-form"
                  style="display: flex; gap: 8px; padding: 12px;
                     border-top: 1px solid #b9bfd2; background: #ffffff;">
              <input id="hubmap-chat-input"
                     type="text"
                     placeholder="Ask about hubs, states, hometowns, rankings..."
                     autocomplete="off"
                     style="flex: 1; padding: 8px 12px;
                        border: 1px solid #b9bfd2; border-radius: 16px;
                        font-size: 13px; color: #484645;
                        font-family: system-ui, -apple-system, sans-serif;
                        outline: none;" />
              <button id="hubmap-voice-btn"
                      type="button"
                      aria-label="Speak to Gemini"
                      title="Speak to Gemini"
                      style="background: #ffffff; color: #152969;
                         border: 1.5px solid #152969; border-radius: 50%;
                         width: 36px; min-width: 36px; height: 36px;
                         padding: 0; cursor: pointer;
                         display: flex; align-items: center; justify-content: center;
                         transition: background 140ms ease, border-color 140ms ease, box-shadow 140ms ease;">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                     stroke="#152969" stroke-width="2.4"
                     stroke-linecap="round" stroke-linejoin="round"
                     aria-hidden="true">
                  <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <path d="M12 19v3" />
                </svg>
              </button>
              <button id="hubmap-audio-btn"
                      type="button"
                      aria-label="Turn audio off"
                      title="Audio on"
                      style="background: #ffffff; color: #152969;
                         border: 1.5px solid #152969; border-radius: 50%;
                         width: 36px; min-width: 36px; height: 36px;
                         padding: 0; cursor: pointer;
                         display: flex; align-items: center; justify-content: center;
                         transition: background 140ms ease, border-color 140ms ease, opacity 140ms ease;">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                     stroke="#152969" stroke-width="2.3"
                     stroke-linecap="round" stroke-linejoin="round"
                     aria-hidden="true">
                  <path d="M11 5 6 9H3v6h3l5 4V5Z" />
                  <path d="M15.5 8.5a5 5 0 0 1 0 7" />
                  <path d="M18.5 5.5a9 9 0 0 1 0 13" />
                </svg>
              </button>
              <button type="submit"
                      aria-label="Send"
                      style="background: #152969; color: #ffffff;
                         border: none; border-radius: 50%;
                         width: 36px; height: 36px;
                         cursor: pointer;
                         display: flex; align-items: center; justify-content: center;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                     stroke="#ffffff" stroke-width="2.5"
                     stroke-linecap="round" stroke-linejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </form>
          </div>

          <div id="hubmap-state-panel"
              class="hsm-state-panel"
              style="position: absolute; top: 12px; right: 12px;
                 display: none; flex-direction: column;
                 background: rgba(255, 255, 255, 0.97);
                 border: 1px solid #b9bfd2; border-radius: 8px;
                 padding: 14px 16px;
                 min-width: 220px; max-width: 280px;
                 box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                 font-family: system-ui, -apple-system, sans-serif;
                 z-index: 102;">
          </div>
          <div id="hubmap-hometown-panel"
              class="hsm-hometown-panel"
              style="position: absolute; top: 12px; right: 12px;
                 display: none; flex-direction: column;
                 background: rgba(255, 255, 255, 0.97);
                 border: 1px solid #b9bfd2; border-radius: 8px;
                 padding: 14px 16px;
                 min-width: 240px; max-width: 310px;
                 box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                 font-family: system-ui, -apple-system, sans-serif;
                 z-index: 103;">
          </div>
        </div>

        <section id="hubmap-legend"
              class="hsm-stats-section"
              style="padding: 24px;
                 background: #efeae6;
                 border-top: 1px solid #b9bfd2;">
          <div style="font-size: 11px; color: #484645;
                text-transform: uppercase; letter-spacing: 1.5px;
                font-weight: 600; margin-bottom: 14px;">
            Hometown hub intelligence
          </div>
          <div class="hsm-stats-grid"></div>
        </section>

        <section id="hubmap-narrative"
              class="hsm-narrative-section"
              style="padding: 24px; background: #ffffff;
                 min-height: 100px;
                 border-top: 1px solid #b9bfd2;">
          <div style="color: #b9bfd2; font-size: 14px;
                font-style: italic;">
            Select a hub on the map to see its profile.
          </div>
        </section>

      </div>
    `;
  }

  private updateLegendCards(state: WidgetState): void {
    const grid = this.querySelector("#hubmap-legend .hsm-stats-grid");
    if (!grid) return;

    let hotSpots = 0;
    let totalAthletes = 0;
    let totalParalympic = 0;

    for (const h of state.hubs) {
      if (h.is_paralympic_hot_spot) hotSpots++;
      totalAthletes += h.total_athletes;
      totalParalympic += h.composition.paralympic_share * h.total_athletes;
    }

    const overallParaPct = totalAthletes > 0
      ? (totalParalympic / totalAthletes) * 100
      : 0;
    const hubsDiscovered = state.hubs.length;

    const fmtNumber = (n: number) => n.toLocaleString();

    const card = (label: string, value: string | number, sub: string, isHotSpot = false) => `
      <div class="hsm-stat-card${isHotSpot ? " hsm-stat-hotspot" : ""}">
        <div class="hsm-stat-label">${label}</div>
        <div class="hsm-stat-value">${value}</div>
        <div class="hsm-stat-sub">${sub}</div>
      </div>
    `;

    grid.innerHTML = `
      ${card("Paralympic Hot Spots", hotSpots, "hubs with 7.5%+ Paralympic share", true)}
      ${card("Athletes Mapped", fmtNumber(totalAthletes), "mapped Olympians and Paralympians, Tokyo 2020 through Milan-Cortina 2026")}
      ${card("Hometown Hubs", hubsDiscovered, "Team USA hometown hubs grouped from public roster geography")}
      ${card("Paralympic Share", overallParaPct.toFixed(1) + "%", "of mapped athletes")}
    `;
  }

  private updateNarrativeCard(state: WidgetState): void {
    const card = this.querySelector("#hubmap-narrative") as HTMLElement;
    if (!card) return;

    if (!state.selectedHubId) {
      card.innerHTML = `
        <div style="color: #b9bfd2; font-size: 14px;
              font-style: italic;">
          Select a hub on the map to see its profile.
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
       HOT SPOT
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
       Loading hub profile...
      </p>`;

    const headline = narrative
      ? `<h2 style="margin: 8px 0 0 0; color: #171fbe;
             font-size: 20px; font-weight: 700;">
       ${narrative.headline}
      </h2>`
      : "";

    card.innerHTML = `
      <div>
       <div style="display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;">
        <span style="color: #152969; font-size: 24px;
               font-weight: 800;">
         ${hub.display_name}
        </span>
        ${hotSpotBadge}
       </div>
       <div style="color: #171fbe; font-size: 15px;
             margin-top: 4px;">
        ${hub.region_name} <span style="color: #484645;
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
             border-top: 1px solid #b9bfd2; flex-wrap: wrap;">
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
       ${this.renderGeographicBlock(narrative)}
      </div>
     `;
  }

  private renderGeographicBlock(narrative: Narrative | undefined): string {
    if (!narrative) return "";
    const climate = narrative.climate;
    const context = narrative.geographic_context;
    if (!climate && !context) return "";

    const stat = (label: string, value: string) => `
      <div style="display: flex; flex-direction: column; gap: 2px;">
        <div style="font-size: 10px; color: #484645; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;">${label}</div>
        <div style="color: #152969; font-size: 16px; font-weight: 700;">${value}</div>
      </div>
    `;

    const climateRow = climate ? `
      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; padding: 12px; background: #efeae6; border-radius: 6px; margin-bottom: 12px;">
        ${climate.annual_avg_temp_f != null ? stat("Avg Temp", `${climate.annual_avg_temp_f}°F`) : ""}
        ${climate.annual_precipitation_in != null ? stat("Precip", `${climate.annual_precipitation_in}″`) : ""}
        ${climate.annual_sunshine_hours != null ? stat("Sunshine", `${Math.round((climate.annual_sunshine_hours / 4380) * 100)}% of yr`) : ""}
        ${climate.elevation_ft != null ? stat("Elevation", `${Math.round(climate.elevation_ft).toLocaleString()}ft`) : ""}
      </div>
    ` : "";

    const contextBlock = context ? `
      <p style="margin: 0; line-height: 1.6; color: #484645; font-size: 13px;">
        ${context}
      </p>
    ` : "";

    return `
      <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid #b9bfd2;">
        <div style="font-size: 11px; color: #484645; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 12px;">
          Why this hub
        </div>
        ${climateRow}
        ${contextBlock}
      </div>
    `;
  }

  private renderHometownPanel(): void {
    const panel = this.querySelector("#hubmap-hometown-panel") as HTMLElement;
    if (!panel) return;
    const focus = this.selectedHometown;
    if (!focus) {
      panel.style.display = "none";
      panel.innerHTML = "";
      return;
    }

    panel.style.display = "flex";
    const para = (focus.paralympic_count || 0) + (focus.both_count || 0);
    const share = focus.total_athletes > 0
      ? ((focus.paralympic_share ?? (para / focus.total_athletes)) * 100).toFixed(1)
      : "0.0";
    const sports = (focus.top_sports || [])
      .slice(0, 3)
      .map(s => `${s.sport} (${s.count})`)
      .join(", ");
    const locationLabel = [focus.hometown, focus.state || focus.state_code].filter(Boolean).join(", ");

    if (focus.ambiguous) {
      const options = (focus.options || [])
        .map(o => `<li style="margin: 4px 0;">${o.hometown}, ${o.state}: <strong>${o.total_athletes}</strong> mapped athletes${o.hub_name ? `, ${o.hub_name}` : ""}</li>`)
        .join("");
      panel.innerHTML = `
        <div style="display: flex; justify-content: space-between; gap: 10px;">
          <div>
            <div style="font-size: 10px; color: #484645; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Hometown lookup</div>
            <div style="color: #152969; font-size: 17px; font-weight: 800; line-height: 1.2; margin-top: 2px;">Clarify state</div>
          </div>
          ${this.closeHometownButtonHtml()}
        </div>
        <div style="margin-top: 10px; color: #484645; font-size: 12px; line-height: 1.45;">
          Multiple mapped hometowns match. Ask with a state for a deterministic zoom.
          <ul style="padding-left: 18px; margin: 8px 0 0;">${options}</ul>
        </div>
      `;
      this.wireHometownPanelClose();
      return;
    }

    panel.innerHTML = `
      <div style="display: flex; justify-content: space-between; gap: 10px;">
        <div>
          <div style="font-size: 10px; color: #484645; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Hometown focus</div>
          <div style="color: #152969; font-size: 17px; font-weight: 800; line-height: 1.2; margin-top: 2px;">${locationLabel || "Requested place"}</div>
        </div>
        ${this.closeHometownButtonHtml()}
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px 12px; padding: 12px 0; margin-top: 10px; border-top: 1px solid #e4e4e7; border-bottom: 1px solid #e4e4e7;">
        <div>
          <div style="font-size: 10px; color: #484645; text-transform: uppercase; letter-spacing: 0.8px;">Mapped Athletes</div>
          <div style="color: #152969; font-size: 22px; font-weight: 800; line-height: 1.1;">${focus.total_athletes}</div>
        </div>
        <div>
          <div style="font-size: 10px; color: #d31118; text-transform: uppercase; letter-spacing: 0.8px;">Paralympians</div>
          <div style="color: #d31118; font-size: 22px; font-weight: 800; line-height: 1.1;">${para}</div>
          <div style="font-size: 11px; color: #484645; margin-top: 2px;">${share}% share</div>
        </div>
      </div>
      <div style="margin-top: 12px; color: #484645; font-size: 12px; line-height: 1.45;">
        ${sports ? `<div><strong style="color: #152969;">Top sports:</strong> ${sports}</div>` : ""}
        ${focus.hub_name ? `<div style="margin-top: 6px;"><strong style="color: #152969;">Hub context:</strong> ${focus.hub_name}${focus.distance_to_hub_km ? `, about ${Math.round(focus.distance_to_hub_km)} km from this hometown` : ""}.</div>` : ""}
        ${focus.nearest_hub_name ? `<div style="margin-top: 6px;"><strong style="color: #152969;">Nearest hub:</strong> ${focus.nearest_hub_name}${focus.nearest_hub_distance_km ? `, about ${focus.nearest_hub_distance_km} km away` : ""}.</div>` : ""}
        ${focus.total_athletes === 0 ? `<div style="margin-top: 6px;">No mapped athletes were found for this exact hometown label in the public dataset.</div>` : ""}
      </div>
    `;
    this.wireHometownPanelClose();
  }

  private closeHometownButtonHtml(): string {
    return `
      <button class="hubmap-hometown-close" type="button" aria-label="Close"
              style="background: transparent; border: none; cursor: pointer; padding: 2px; color: #484645; line-height: 1;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    `;
  }

  private wireHometownPanelClose(): void {
    const close = this.querySelector(".hubmap-hometown-close") as HTMLButtonElement | null;
    close?.addEventListener("click", () => {
      this.selectedHometown = null;
      this.renderHometownPanel();
      this.updateLayers();
    });
  }

  private renderStatePanel(): void {
    const panel = this.querySelector("#hubmap-state-panel") as HTMLElement;
    if (!panel) return;

    if (!this.selectedStateCode || !this.selectedStateName) {
      panel.style.display = "none";
      return;
    }

    const agg = this.stateAggregates.find(s => s.state === this.selectedStateCode);

    const state = this.store.getState();
    const stateHubs = state.hubs.filter(h => h.states.includes(this.selectedStateCode!));
    const topHub = stateHubs.length > 0
      ? stateHubs.reduce((max, h) => h.total_athletes > max.total_athletes ? h : max)
      : null;

    const rankUniverseCount = this.stateAggregates.length;
    let totalRank: string | number = "-";
    let paraRank: string | number = "-";
    if (agg) {
      const stateNameFor = (code: string): string =>
        Object.keys(STATE_NAME_TO_CODE).find(k => STATE_NAME_TO_CODE[k] === code) || code;
      const sortedByTotal = [...this.stateAggregates].sort((a, b) =>
        (b.total_athletes - a.total_athletes) || stateNameFor(a.state).localeCompare(stateNameFor(b.state)));
      totalRank = sortedByTotal.findIndex(s => s.state === this.selectedStateCode) + 1;
      const sortedByPara = [...this.stateAggregates].sort((a, b) =>
        ((b.paralympic_count + b.both_count) - (a.paralympic_count + a.both_count)) ||
        (b.total_athletes - a.total_athletes) ||
        stateNameFor(a.state).localeCompare(stateNameFor(b.state)));
      paraRank = sortedByPara.findIndex(s => s.state === this.selectedStateCode) + 1;
    }

    const paraTotal = agg ? (agg.paralympic_count + agg.both_count) : 0;
    const paraPct = agg ? (agg.paralympic_share * 100).toFixed(1) : "0.0";
    const totalAthletes = agg ? agg.total_athletes : 0;

    panel.style.display = "flex";
    panel.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="background: #152969; color: #ffffff; font-size: 14px; font-weight: 800; letter-spacing: 0.5px; padding: 6px 9px; border-radius: 4px; line-height: 1; min-width: 32px; text-align: center;">${this.selectedStateCode}</div>
          <div>
            <div style="font-size: 10px; color: #484645; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">State</div>
            <div style="color: #152969; font-size: 17px; font-weight: 800; line-height: 1.15; margin-top: 1px;">${this.selectedStateName}</div>
          </div>
        </div>
        <button class="hubmap-state-close" type="button" aria-label="Close"
          style="background: transparent; border: none; cursor: pointer; padding: 2px; color: #484645; line-height: 1;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px 12px; padding: 12px 0; border-top: 1px solid #e4e4e7; border-bottom: 1px solid #e4e4e7;">
        <div>
          <div style="font-size: 10px; color: #484645; text-transform: uppercase; letter-spacing: 0.8px;">Total Athletes</div>
          <div style="color: #152969; font-size: 20px; font-weight: 700; line-height: 1.1;">${totalAthletes}</div>
          <div style="font-size: 11px; color: #484645; margin-top: 2px;">Rank <strong style="color: #152969;">#${totalRank}</strong> of ${rankUniverseCount} in-scope regions</div>
          </div>
        <div>
          <div style="font-size: 10px; color: #d31118; text-transform: uppercase; letter-spacing: 0.8px;">Paralympic</div>
          <div style="color: #d31118; font-size: 20px; font-weight: 700; line-height: 1.1;">${paraTotal}</div>
          <div style="font-size: 11px; color: #484645; margin-top: 2px;">Rank <strong style="color: #d31118;">#${paraRank}</strong> of ${rankUniverseCount} in-scope regions</div>
        </div>
        <div style="grid-column: span 2;">
          <div style="font-size: 10px; color: #484645; text-transform: uppercase; letter-spacing: 0.8px;">Paralympic Share</div>
          <div style="color: #152969; font-size: 16px; font-weight: 700; line-height: 1.2;">${paraPct}%</div>
        </div>
      </div>
      ${topHub ? `
        <div style="margin-top: 12px;">
          <div style="font-size: 10px; color: #484645; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;">Top Hub</div>
          <button class="hubmap-state-hub-link" type="button" data-hub-id="${topHub.hub_id}"
            style="background: transparent; border: none; padding: 0; color: #171fbe; font-size: 13px; font-weight: 600; cursor: pointer; text-align: left; font-family: inherit;">
            ${topHub.display_name}
          </button>
        </div>
      ` : ''}
    `;

    panel.querySelector(".hubmap-state-close")?.addEventListener("click", (e) => {
      e.stopPropagation();
      this.selectedStateCode = null;
      this.selectedStateName = null;
      this.renderStatePanel();
      this.updateLayers();
    });

    panel.querySelector(".hubmap-state-hub-link")?.addEventListener("click", (e) => {
      const hubId = (e.currentTarget as HTMLElement).getAttribute("data-hub-id");
      if (hubId) this.zoomToHub(hubId);
    });
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
      { label: "Alaska", stateName: "Alaska", hubId: "HUB_AK_ANCHORAGE", width: 110, height: 80 },
      { label: "Hawaii", stateName: "Hawaii", hubId: "HUB_HI_HONOLULU", width: 90, height: 70 },
      { label: "Puerto Rico", stateName: "Puerto Rico", hubId: "HUB_PR_SAN_JUAN", width: 90, height: 60 },
    ];

    container.innerHTML = insets.map(inset => {
      const feature = this.stateGeoJson.features.find(
        (f: any) => f.properties?.name === inset.stateName
      );
      if (!feature) return "";

      const hub = state.hubs.find(h => h.hub_id === inset.hubId);

      const displayFeature = inset.stateName === "Alaska"
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

      let athleteDots = "";
      if (this.athletes.length > 0) {
        const stateAthletes = this.athletes.filter(a => a.state === stateCode);
        athleteDots = stateAthletes.map(a => {
          const projected = projection([a.lon, a.lat]);
          if (!projected) return "";
          const [ax, ay] = projected;
          if (ax < 0 || ax > inset.width || ay < 0 || ay > inset.height) return "";
          const isPara = a.status === "paralympic" || a.status === "both";
          const dotFill = isPara ? "#d31118" : "#152969";
          const dotOpacity = isPara ? "0.7" : "0.4";
          return `<circle cx="${ax.toFixed(2)}" cy="${ay.toFixed(2)}" r="1.2"
            fill="${dotFill}" opacity="${dotOpacity}" />`;
        }).join("");
      }

      let hubDot = "";
      if (hub) {
        const [hx, hy] = projection([hub.centroid_longitude, hub.centroid_latitude]) || [0, 0];
        const isSelected = hub.hub_id === state.selectedHubId;
        const isHotSpot = hub.is_paralympic_hot_spot;
        const dotFill = isHotSpot ? "#d31118" : "#152969";
        const dotStroke = isSelected ? "#ffc832" : "#ffffff";
        const dotStrokeWidth = isSelected ? 2.5 : 1.5;
        const dotR = isSelected ? 5 : 4;
        const dotRHover = dotR + 2;
        hubDot = `<circle cx="${hx}" cy="${hy}" r="${dotR}"
          fill="${dotFill}" stroke="${dotStroke}" stroke-width="${dotStrokeWidth}"
          style="cursor: pointer; transition: r 0.15s ease;"
          data-hub-id="${hub.hub_id}"
          onmouseover="this.setAttribute('r', '${dotRHover}')"
          onmouseout="this.setAttribute('r', '${dotR}')" />`;
      }

      return `
        <div data-inset="${inset.label}" data-hub-id="${inset.hubId}" data-state-code="${stateCode}" data-state-name="${inset.stateName}"
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
          ${athleteDots}
          ${hubDot}
         </svg>
         <div style="font-size: 10px; color: #484645;
               font-weight: 600; letter-spacing: 0.5px;
               margin-top: 4px; white-space: nowrap;">
          ${inset.label}
         </div>
        </div>
      `;
    }).join("");

    container.querySelectorAll("[data-inset]").forEach(el => {
      const hubId = el.getAttribute("data-hub-id");
      const stateCode = el.getAttribute("data-state-code");
      const stateName = el.getAttribute("data-state-name");
      el.addEventListener("click", (e: Event) => {
        e.preventDefault();
        e.stopPropagation();
        if (stateCode && stateName) {
          this.selectedStateCode = stateCode;
          this.selectedStateName = stateName;
          this.renderStatePanel();
          this.updateLayers();
          this.centerOnState(stateCode);
        }
        if (hubId && hubId !== "null") this.zoomToHub(hubId);
      });
    });
  }
}
