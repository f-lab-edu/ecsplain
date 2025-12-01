// src/api.ts
export type QueryResponse = {
  answer?: string;
  sources?: { source?: string; page?: number }[];
  error?: string;
};

// 1순위: 빌드 시 주입된 VITE_API_BASE
const envBase = import.meta.env.VITE_API_BASE;

// 2순위: 안 들어왔으면, 프론트가 떠 있는 서버 IP 기준으로 자동 구성
const API_BASE =
  envBase ||
  `${window.location.protocol}//${window.location.hostname}:8000`;

console.log("[RAG] API_BASE =", API_BASE);

export async function ask(question: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await res.json();
  if (!res.ok) {
    console.error("[RAG] /query error resp:", data);
    throw new Error(data?.error || data?.detail || "Request failed");
  }
  console.log("[RAG] /query ok resp:", data);
  return data as QueryResponse;
}