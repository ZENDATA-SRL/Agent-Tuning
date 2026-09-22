const API = "";

export async function fetchAdapters() {
  const res = await fetch(`${API}/api/adapters`);
  if (!res.ok) throw new Error(`adapters: ${res.status}`);
  const data = await res.json();
  return data.adapters || [];
}

export async function fetchSystemPrompt() {
  const res = await fetch(`${API}/api/system-prompt`);
  if (!res.ok) throw new Error(`system-prompt: ${res.status}`);
  const data = await res.json();
  return data.system_prompt || "";
}

export async function sendChat(adapterId, messages) {
  const res = await fetch(`${API}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      adapter_id: adapterId,
      messages,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `chat: ${res.status}`);
  }
  return res.json();
}
