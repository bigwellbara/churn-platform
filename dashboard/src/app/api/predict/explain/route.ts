import { forwardPost } from "@/lib/upstream";

export const maxDuration = 120;

export async function POST(request: Request) {
  return forwardPost(request, "/predict/explain");
}
