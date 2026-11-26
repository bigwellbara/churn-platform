import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { forward, forwardPost } from "./upstream";

function reply(body: unknown, status = 200): Promise<Response> {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

type FetchMock = (url: string, init: RequestInit) => Promise<Response>;

describe("forward", () => {
  beforeEach(() => {
    vi.stubEnv("CHURN_API_URL", "https://api.example.com/");
    vi.stubEnv("CHURN_API_KEY", "");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("returns 500 when the API URL is not configured", async () => {
    vi.stubEnv("CHURN_API_URL", "");
    const res = await forward("/health", { method: "GET" });
    expect(res.status).toBe(500);
    expect((await res.json()).error).toBe("not_configured");
  });

  it("passes the API's status and body through, without a double slash", async () => {
    const fetchImpl = vi.fn<FetchMock>(() => reply({ error: "validation_error", detail: [] }, 422));
    const res = await forward("/predict", { method: "POST", body: "{}" }, fetchImpl as typeof fetch);
    expect(res.status).toBe(422);
    expect((await res.json()).error).toBe("validation_error");
    expect(fetchImpl.mock.calls[0]![0]).toBe("https://api.example.com/predict");
  });

  it("adds the API key header only when one is configured", async () => {
    const fetchImpl = vi.fn<FetchMock>(() => reply({ status: "ok" }));
    await forward("/health", { method: "GET" }, fetchImpl as typeof fetch);
    expect(fetchImpl.mock.calls[0]![1].headers).not.toHaveProperty("X-API-Key");

    vi.stubEnv("CHURN_API_KEY", "secret");
    await forward("/health", { method: "GET" }, fetchImpl as typeof fetch);
    expect(fetchImpl.mock.calls[1]![1].headers).toHaveProperty("X-API-Key", "secret");
  });

  it("turns a timeout into a 504 with a wake-up hint", async () => {
    const fetchImpl = vi.fn<FetchMock>(() =>
      Promise.reject(new DOMException("timed out", "TimeoutError")),
    );
    const res = await forward("/health", { method: "GET" }, fetchImpl as typeof fetch);
    expect(res.status).toBe(504);
    expect((await res.json()).detail).toMatch(/waking up/);
  });

  it("turns a network failure into a 502", async () => {
    const fetchImpl = vi.fn<FetchMock>(() => Promise.reject(new TypeError("fetch failed")));
    const res = await forward("/health", { method: "GET" }, fetchImpl as typeof fetch);
    expect(res.status).toBe(502);
  });
});

describe("forwardPost", () => {
  beforeEach(() => vi.stubEnv("CHURN_API_URL", "https://api.example.com"));
  afterEach(() => vi.unstubAllEnvs());

  const post = (body: string) => new Request("http://x/api/predict", { method: "POST", body });

  it("rejects a body that is not JSON before calling the API", async () => {
    const fetchImpl = vi.fn<FetchMock>();
    const res = await forwardPost(post("not json"), "/predict", fetchImpl as typeof fetch);
    expect(res.status).toBe(400);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("rejects an oversized body", async () => {
    const fetchImpl = vi.fn<FetchMock>();
    const big = JSON.stringify({ pad: "x".repeat(1_100_000) });
    const res = await forwardPost(post(big), "/predict", fetchImpl as typeof fetch);
    expect(res.status).toBe(413);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("forwards valid JSON", async () => {
    const fetchImpl = vi.fn<FetchMock>(() => reply({ churn_probability: 0.5 }));
    const res = await forwardPost(post("{}"), "/predict", fetchImpl as typeof fetch);
    expect(res.status).toBe(200);
  });
});
