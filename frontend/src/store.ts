// frontend/src/store.ts
import { Action, WidgetState, INITIAL_STATE } from "./widget-contract";

export type Listener = (state: WidgetState) => void;

export class Store {
  private state: WidgetState = { ...INITIAL_STATE };
  private listeners: Set<Listener> = new Set();

  getState(): WidgetState {
    return this.state;
  }

  dispatch(action: Action): WidgetState {
    this.state = reduce(this.state, action);
    this.listeners.forEach(l => l(this.state));
    return this.state;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

function reduce(state: WidgetState, action: Action): WidgetState {
  switch (action.type) {
    case "SELECT_HUB":
      return { ...state, selectedHubId: action.hub_id };
    case "CLEAR_SELECTION":
      return { ...state, selectedHubId: null };
    case "SET_FILTER":
      return { ...state, filters: { ...state.filters, ...action.filter } };
    case "CLEAR_FILTERS":
      return { ...state, filters: {} };
    case "SET_VIEW":
      return { ...state, view: action.view };
    case "DATA_LOADED":
      return { ...state, hubs: action.hubs, loadStatus: "loaded", loadError: null };
    case "DATA_ERROR":
      return { ...state, loadStatus: "error", loadError: action.error };
    default:
      return state;
  }
}