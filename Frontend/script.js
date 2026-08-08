// ---------------------------------------------------------------------------
// Point this at your deployed FastAPI backend.
// Locally:  "http://127.0.0.1:8000"
// Deployed: "https://your-backend-project.vercel.app"  (no trailing slash)
// ---------------------------------------------------------------------------
const API_BASE_URL = "http://127.0.0.1:8000";

const transcript = document.getElementById("transcript");
const composer = document.getElementById("composer");
const input = document.getElementById("question");
const sendBtn = document.getElementById("sendBtn");
const startTime = document.getElementById("startTime");

let questionCount = 0;

function formatTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function qLabel(n) {
  return "Q" + String(n).padStart(2, "0");
}

startTime.textContent = formatTime();

function addEntry({ type, label, text, html }) {
  const entry = document.createElement("div");
  entry.className = `entry ${type}`;

  const meta = document.createElement("div");
  meta.className = "entry-meta";
  meta.innerHTML = `<span class="q-index">${label}</span><span class="timestamp">${formatTime()}</span>`;

  const body = document.createElement("p");
  body.className = "entry-text";
  if (html) {
    body.innerHTML = html;
  } else {
    body.textContent = text;
  }

  entry.appendChild(meta);
  entry.appendChild(body);
  transcript.appendChild(entry);
  transcript.scrollTop = transcript.scrollHeight;
  return entry;
}

function addTypingIndicator() {
  const entry = document.createElement("div");
  entry.className = "entry answer";
  entry.id = "typing-indicator";
  entry.innerHTML = `
    <div class="entry-meta"><span class="q-index">A</span><span class="timestamp">${formatTime()}</span></div>
    <p class="entry-text"><span class="typing"><span></span><span></span><span></span></span></p>
  `;
  transcript.appendChild(entry);
  transcript.scrollTop = transcript.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

async function askBackend(question) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const err = await response.json();
      if (err.detail) detail = err.detail;
    } catch (_) {}
    throw new Error(detail);
  }

  const data = await response.json();
  return data.answer;
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = input.value.trim();
  if (!question) return;

  questionCount += 1;
  addEntry({ type: "question", label: qLabel(questionCount), text: question });

  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;
  addTypingIndicator();

  try {
    const answer = await askBackend(question);
    removeTypingIndicator();
    addEntry({ type: "answer", label: "A", text: answer });
  } catch (err) {
    removeTypingIndicator();
    addEntry({
      type: "error",
      label: "ERROR",
      text: `Couldn't reach the backend — ${err.message}. Check that the API is running and CORS is configured.`,
    });
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
});