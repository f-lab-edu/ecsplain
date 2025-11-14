import { useState } from "react";
import { ask } from "./api";

export default function App() {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<{source?: string; page?: number}[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onAsk = async () => {
    setLoading(true);
    setAnswer(null);
    setSources(null);
    setError(null);
    try {
      const r = await ask(q);
      setAnswer(r.answer ?? "");
      setSources(r.sources ?? []);
    } catch (e: any) {
      setError(e.message || "요청 실패");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ maxWidth: 860, margin: "40px auto", padding: 16, fontFamily: "ui-sans-serif" }}>
      <h1 style={{ fontSize: 26, fontWeight: 700, marginBottom: 12 }}>RAG React SPA</h1>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <input
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder="질문을 입력하세요"
          style={{ flex: 1, padding: 10, border: "1px solid #ccc", borderRadius: 8 }}
        />
        <button onClick={onAsk} disabled={loading || !q.trim()} style={{ padding: "10px 16px", borderRadius: 8 }}>
          {loading ? "질의중..." : "질의"}
        </button>
      </div>

      {error && <div style={{ color: "crimson", marginBottom: 12 }}>⚠️ {error}</div>}

      {answer && (
        <section style={{ padding: 16, border: "1px solid #eee", borderRadius: 8, background: "#fafafa" }}>
          <h2 style={{ marginTop: 0 }}>답변</h2>
          <pre style={{ whiteSpace: "pre-wrap" }}>{answer}</pre>
          {sources && sources.length > 0 && (
            <>
              <h3>출처</h3>
              <ul>
                {sources.map((s, i) => (
                  <li key={i}>
                    {s.source ?? "unknown"} {s.page != null ? `(p.${s.page})` : ""}
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}
    </main>
  );
}
