// frontend/src/main.ts
import { HometownHubMap } from "./hometown-hub-map";

if (!customElements.get("hometown-hub-map")) {
  customElements.define("hometown-hub-map", HometownHubMap);
}

export { HometownHubMap };
export type { Action, Hub, WidgetState, FilterUpdate } from "./widget-contract";