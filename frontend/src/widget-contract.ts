// frontend/src/widget-contract.ts
export type FilterUpdate = {
  macro_region?:
    | "Northeast" | "Mid-Atlantic" | "South" | "Midwest"
    | "Southwest" | "Mountain West" | "Pacific" | "Alaska"
    | "Hawaii" | "Territories";
  region_name?: string;
  paralympic_focus?: boolean;
  sport_category?: "summer" | "winter" | "all";
};

export type Action =
  | { type: "SELECT_HUB"; hub_id: string }
  | { type: "CLEAR_SELECTION" }
  | { type: "SET_FILTER"; filter: FilterUpdate }
  | { type: "CLEAR_FILTERS" }
  | { type: "SET_VIEW"; view: "map" | "table" }
  | { type: "DATA_LOADED"; hubs: Hub[] }
  | { type: "DATA_ERROR"; error: string };

export type HubComposition = {
  olympic_count: number;
  paralympic_count: number;
  both_count: number;
  paralympic_share: number;
  composition_label: string;
};

export type SportInHub = {
  sport: string;
  count: number;
  paralympic_count: number;
  track_type: string;
};

export type Hub = {
  hub_id: string;
  display_name: string;
  centroid_latitude: number;
  centroid_longitude: number;
  medoid_hometown: string;
  radius_km: number;
  region: string;
  region_name: string;
  macro_region: string;
  states: string[];
  total_athletes: number;
  composition: HubComposition;
  is_paralympic_hot_spot: boolean;
  top_sports: SportInHub[];
  sport_diversity_index: number;
  tags: string[];
  search_aliases: string[];
};

export type WidgetState = {
  hubs: Hub[];
  selectedHubId: string | null;
  filters: FilterUpdate;
  view: "map" | "table";
  loadStatus: "idle" | "loading" | "loaded" | "error";
  loadError: string | null;
};

export const INITIAL_STATE: WidgetState = {
  hubs: [],
  selectedHubId: null,
  filters: {},
  view: "map",
  loadStatus: "idle",
  loadError: null
};