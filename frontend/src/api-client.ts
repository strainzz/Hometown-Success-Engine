// frontend/src/api-client.ts
import { Hub } from "./widget-contract";

export class ApiClient {
  constructor(private baseUrl: string) {}

  async fetchHubs(): Promise<Hub[]> {
    const res = await fetch(`${this.baseUrl}/hubs`);
    if (!res.ok) {
      throw new Error(`Hubs fetch failed: ${res.status} ${res.statusText}`);
    }
    return await res.json() as Hub[];
  }

  async fetchNarrative(hub_id: string): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}/hubs/${hub_id}/narrative`);
    if (!res.ok) {
      throw new Error(`Narrative fetch failed: ${res.status}`);
    }
    return await res.json();
  }

  async prewarmVoice(): Promise<void> {
    await fetch(`${this.baseUrl}/voice/prewarm`, { method: "GET" });
  }
}
