import { forwardPost } from "@/lib/upstream";

// A cold free-tier API can take up to two minutes to answer the first request.
export const maxDuration = 120;

export async function POST(request: Request) {
  return forwardPost(request, "/predict");
}
