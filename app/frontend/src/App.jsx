import { useEffect, useMemo, useRef, useState } from "react";
import { fetchAdapters, fetchSystemPrompt, sendChat } from "./api";

function pretty(value) {
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

function ChatToolCalls({ calls }) {
  return (
    <div className="tool-stack">
      {calls.map((tc, i) => (
        <div key={tc.id || i} className="tool-card call">
          <div className="tool-head">
            <span className="pill call">tool_call</span>
            <code>{tc.name}</code>
          </div>
          <pre>{pretty(tc.args ?? {})}</pre>
        </div>
      ))}
    </div>
  );
}

function ChatToolResult({ item }) {
  return (
    <div className="tool-card result">
      <div className="tool-head">
        <span className="pill result">tool_result</span>
        <code>{item.name || "tool"}</code>
        {item.tool_call_id ? (
          <span className="muted-id">{item.tool_call_id}</span>
        ) : null}
      </div>
      <pre>{pretty(item.content)}</pre>
    </div>
  );
}

function ChatView({ messages, loading }) {
  return (
    <>
      {messages.map((m, i) => {
        if (m.role === "tool") {
          return <ChatToolResult key={i} item={m} />;
        }
        if (m.tool_calls?.length) {
          return (
            <div key={i} className="turn-block">
              {m.content ? (
                <div className="bubble assistant">
                  <span className="role">assistant</span>
                  <p>{m.content}</p>
                </div>
              ) : null}
              <ChatToolCalls calls={m.tool_calls} />
            </div>
          );
        }
        return (
          <div key={i} className={`bubble ${m.role}`}>
            <span className="role">{m.role}</span>
            <p>{m.content}</p>
          </div>
        );
      })}
      {loading ? <div className="bubble assistant pending">Generazione…</div> : null}
    </>
  );
}

function RawView({ messages, loading, lastExchange }) {
  return (
    <div className="raw-feed">
      {lastExchange ? (
        <details className="raw-block exchange" open>
          <summary>ultima richiesta / risposta API</summary>
          <div className="raw-grid">
            <div>
              <h3>request</h3>
              <pre>{pretty(lastExchange.request)}</pre>
            </div>
            <div>
              <h3>response</h3>
              <pre>{pretty(lastExchange.response)}</pre>
            </div>
          </div>
        </details>
      ) : null}

      {messages.map((m, i) => {
        const label =
          m.role === "tool"
            ? `tool_result · ${m.name || "tool"}`
            : m.tool_calls?.length
              ? `assistant · ${m.tool_calls.length} tool_call(s)`
              : m.role;

        return (
          <details key={i} className={`raw-block ${m.role}`} open>
            <summary>
              <span className="idx">#{i}</span>
              <span>{label}</span>
            </summary>

            {m.raw_completion ? (
              <div className="raw-section">
                <h3>raw model completion</h3>
                <pre className="raw-text">{m.raw_completion}</pre>
              </div>
            ) : null}

            {m.tool_calls?.length ? (
              <div className="raw-section">
                <h3>parsed tool_calls</h3>
                <pre>{pretty(m.tool_calls)}</pre>
              </div>
            ) : null}

            {m.role === "tool" ? (
              <div className="raw-section">
                <h3>tool payload</h3>
                <pre>
                  {pretty({
                    name: m.name,
                    tool_call_id: m.tool_call_id,
                    content: (() => {
                      try {
                        return JSON.parse(m.content);
                      } catch {
                        return m.content;
                      }
                    })(),
                  })}
                </pre>
              </div>
            ) : null}

            {(m.role === "user" ||
              (m.role === "assistant" && !m.tool_calls?.length)) && (
              <div className="raw-section">
                <h3>message</h3>
                <pre>{pretty({ role: m.role, content: m.content })}</pre>
              </div>
            )}

            {m.role === "assistant" && m.tool_calls?.length && m.content ? (
              <div className="raw-section">
                <h3>assistant content</h3>
                <pre>{pretty(m.content)}</pre>
              </div>
            ) : null}
          </details>
        );
      })}

      {loading ? (
        <div className="raw-block pending">Generazione in corso…</div>
      ) : null}
    </div>
  );
}

export default function App() {
  const [adapters, setAdapters] = useState([]);
  const [adapterId, setAdapterId] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState("chat");
  const [lastExchange, setLastExchange] = useState(null);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [showSystem, setShowSystem] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    fetchAdapters()
      .then((list) => {
        setAdapters(list);
        if (list.length && !adapterId) setAdapterId(list[0].id);
      })
      .catch((err) => setError(String(err.message || err)));
    fetchSystemPrompt()
      .then(setSystemPrompt)
      .catch((err) => setError(String(err.message || err)));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, view]);

  const toolCallCount = useMemo(
    () =>
      messages.reduce(
        (n, m) => n + (Array.isArray(m.tool_calls) ? m.tool_calls.length : 0),
        0,
      ),
    [messages],
  );

  const canSend = useMemo(
    () => Boolean(adapterId && input.trim() && !loading),
    [adapterId, input, loading],
  );

  async function onSend(e) {
    e?.preventDefault();
    if (!canSend) return;
    const text = input.trim();
    setInput("");
    setError("");
    const next = [...messages, { role: "user", content: text }];
    setMessages(next);
    setLoading(true);
    try {
      const history = next
        .filter(
          (m) =>
            m.role === "user" ||
            (m.role === "assistant" && !m.tool_calls?.length),
        )
        .map((m) => ({ role: m.role, content: m.content || "" }));
      const request = { adapter_id: adapterId, messages: history };
      const result = await sendChat(adapterId, history);
      setLastExchange({ request, response: result });
      const trace = result.trace || [];
      setMessages([...next, ...trace]);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  }

  function onReset() {
    setMessages([]);
    setError("");
    setLastExchange(null);
  }

  return (
    <div className="shell">
      <header className="top">
        <div>
          <p className="brand">Agent Tuning</p>
          <h1>Playground</h1>
          {messages.length ? (
            <p className="meta">
              tool_calls in sessione: <strong>{toolCallCount}</strong>
              {toolCallCount === 0
                ? " — il modello non ne ha emesse (controlla Raw)"
                : null}
            </p>
          ) : null}
        </div>
        <div className="controls">
          <label>
            Adapter
            <select
              value={adapterId}
              onChange={(e) => {
                setAdapterId(e.target.value);
                setMessages([]);
                setLastExchange(null);
                setError("");
              }}
              disabled={loading || !adapters.length}
            >
              {!adapters.length ? <option value="">Nessun adapter</option> : null}
              {adapters.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.label}
                </option>
              ))}
            </select>
          </label>

          <div className="view-switch" role="group" aria-label="Vista">
            <button
              type="button"
              className={view === "chat" ? "active" : ""}
              onClick={() => setView("chat")}
            >
              Chat
            </button>
            <button
              type="button"
              className={view === "raw" ? "active" : ""}
              onClick={() => setView("raw")}
            >
              Raw
            </button>
          </div>

          <button
            type="button"
            className={`ghost ${showSystem ? "active-ghost" : ""}`}
            onClick={() => setShowSystem((v) => !v)}
          >
            System
          </button>

          <button type="button" className="ghost" onClick={onReset} disabled={loading}>
            Reset
          </button>
        </div>
      </header>

      {showSystem ? (
        <section className="system-panel">
          <div className="system-panel-head">
            <h2>System prompt</h2>
            <button type="button" className="ghost" onClick={() => setShowSystem(false)}>
              Chiudi
            </button>
          </div>
          <pre>{systemPrompt || "(vuoto)"}</pre>
        </section>
      ) : null}

      <main className="chat">
        {!messages.length && !loading ? (
          <div className="empty">
            Seleziona il modello base o un adapter LoRA. In vista Raw vedi
            completion grezza, tool_calls e risposta API.
          </div>
        ) : view === "chat" ? (
          <ChatView messages={messages} loading={loading} />
        ) : (
          <RawView
            messages={messages}
            loading={loading}
            lastExchange={lastExchange}
          />
        )}

        {error ? <div className="error">{error}</div> : null}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={onSend}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Es. Ciao, a che ora aprite? / Vorrei un taglio capelli…"
          disabled={loading || !adapterId}
        />
        <button type="submit" disabled={!canSend}>
          Invia
        </button>
      </form>
    </div>
  );
}
