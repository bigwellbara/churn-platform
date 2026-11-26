// Server-side only: forwards dashboard requests to the prediction API.
//
// Why a proxy instead of calling the API from the browser:
// - the browser talks only to this site, so the API needs no CORS rules;
// - an API key (if the API has one) stays on the server, never in page code;
// - one place to turn timeouts and outages into clear messages.

/** Free Render instances take up to ~2 minutes to wake; stay just under Vercel's limit. */
export const UPSTREAM_TIMEOUT_MS = 110_000;
/** Largest request body forwarded (a 500-customer chunk is ~250 KB). */
export const MAX_BODY_BYTES = 1_000_000;

export function apiBaseUrl(): string | null {
  const url = process.env.CHURN_API_URL?.trim();
  return url ? url.replace(/\/+$/, "") : null;
}

function json(status: number, error: string, detail: string): Response {
  return Response.json({ error, detail }, { status });
}

export async function forward(
  path: string,
  init: { method: "GET" | "POST"; body?: string },
  fetchImpl: typeof fetch = fetch,
): Promise<Response> {
  const base = apiBaseUrl();
  if (!base) return json(500, "not_configured", "CHURN_API_URL is not set on the server");

  const headers: Record<string, string> = { Accept: "application/json" };
  if (init.body !== undefined) headers["Content-Type"] = "application/json";
  const key = process.env.CHURN_API_KEY?.trim();
  if (key) headers["X-API-Key"] = key;

  try {
    const upstream = await fetchImpl(`${base}${path}`, {
      method: init.method,
      headers,
      body: init.body,
      cache: "no-store",
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" },
    });
  } catch (err) {
    if (err instanceof DOMException && (err.name === "TimeoutError" || err.name === "AbortError")) {
      return json(
        504,
        "upstream_timeout",
        "The prediction API did not answer in time. It may be waking up; try again in a minute.",
      );
    }
    return json(502, "upstream_unreachable", "Could not reach the prediction API.");
  }
}

/** Read a JSON request body with a size cap, then forward it as a POST. */
export async function forwardPost(
  request: Request,
  path: string,
  fetchImpl: typeof fetch = fetch,
): Promise<Response> {
  const body = await request.text();
  if (new TextEncoder().encode(body).length > MAX_BODY_BYTES) {
    return json(413, "too_large", "Request body is too large; send at most 500 customers at a time.");
  }
  try {
    JSON.parse(body);
  } catch {
    return json(400, "invalid_json", "Request body must be JSON.");
  }
  return forward(path, { method: "POST", body }, fetchImpl);
}
