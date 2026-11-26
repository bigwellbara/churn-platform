import { forward } from "@/lib/upstream";

export const maxDuration = 120;
export const dynamic = "force-dynamic";

export async function GET() {
  return forward("/health", { method: "GET" });
}
