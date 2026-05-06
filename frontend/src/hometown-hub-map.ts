// frontend/src/hometown-hub-map.ts
import { Action, Hub, FilterUpdate, WidgetState } from "./widget-contract";
import { Store } from "./store";
import { ApiClient } from "./api-client";

const DEFAULT_API_URL = "https://hometown-success-engine-yumatgk63a-uc.a.run.app";

export class HometownHubMap extends HTMLElement {
  private store: Store;
  private api: ApiClient | null = null;
  private unsubscribe: (() => void) | null = null;

  constructor() {
    super();
    this.store = new Store();
  }

  static get observedAttributes(): string[] {
    return ["api-url"];
  }

  connectedCallback(): void {
    const apiUrl = this.getAttribute("api-url") || DEFAULT_API_URL;
    this.api = new ApiClient(apiUrl);

    this.unsubscribe = this.store.subscribe(state => this.render(state));

    this.render(this.store.getState());
    void this.loadHubs();
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
  }

  filterToParalympic(macro_region?: FilterUpdate["macro_region"]): void {
    this.dispatch({
      type: "SET_FILTER",
      filter: { paralympic_focus: true, macro_region }
    });
  }

  zoomToHub(hub_id: string): void {
    this.dispatch({ type: "SELECT_HUB", hub_id });
  }

  resetView(): void {
    this.dispatch({ type: "CLEAR_SELECTION" });
    this.dispatch({ type: "CLEAR_FILTERS" });
  }

  // ===== INTERNAL =====

  private async loadHubs(): Promise<void> {
    if (!this.api) return;
    this.dispatch({ type: "DATA_LOADED", hubs: [] }); // Set load status to loading (in custom reducer, it's just a state change before true data arrives, though here we directly await)
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

  private render(state: WidgetState): void {
    const filterParts: string[] = [];
    if (state.filters.macro_region) filterParts.push(`macro_region=${state.filters.macro_region}`);
    if (state.filters.paralympic_focus) filterParts.push("paralympic_focus=true");
    if (state.filters.sport_category) filterParts.push(`sport=${state.filters.sport_category}`);
    const filterStr = filterParts.length > 0 ? filterParts.join(", ") : "none";

    const selectedHub = state.hubs.find(h => h.hub_id === state.selectedHubId);

    this.innerHTML = `
      <div style="font-family: system-ui, sans-serif; padding: 16px;
           border: 1px solid #ddd; border-radius: 8px;
           background: #fafafa;">
        <h3 style="margin: 0 0 8px 0;">Hometown Hub Map (Phase 4 Day 1 stub)</h3>
        <p style="margin: 4px 0; color: #666; font-size: 14px;">
         Status: <strong>${state.loadStatus}</strong> ·
         Hubs loaded: <strong>${state.hubs.length}</strong> ·
         Filters: <strong>${filterStr}</strong>
        </p>
        ${state.loadError ? `<p style="color: red; margin: 4px 0;">Error: ${state.loadError}</p>` : ""}
        ${selectedHub ? `
         <div style="margin-top: 12px; padding: 12px;
               background: white; border-radius: 4px;
               border: 1px solid #e0e0e0;">
          <strong>${selectedHub.display_name}</strong>
          <span style="color: #666; font-size: 13px;">
           · ${selectedHub.region_name} · ${selectedHub.macro_region}
          </span>
          <p style="margin: 4px 0; font-size: 14px;">
           ${selectedHub.total_athletes} athletes ·
           ${(selectedHub.composition.paralympic_share * 100).toFixed(1)}% Paralympic
           ${selectedHub.is_paralympic_hot_spot ? '<span style="color: #E69F00;">★ HOT SPOT</span>' : ""}
          </p>
         </div>
        ` : `<p style="color: #999; font-size: 14px; margin: 12px 0;">No hub selected.</p>`}
        <p style="color: #999; font-size: 12px; margin: 12px 0 0 0;">
         deck.gl map renders here in Phase 4 Day 2
        </p>
      </div>
    `;
  }
}