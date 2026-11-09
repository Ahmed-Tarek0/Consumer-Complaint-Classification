// Point this at wherever the FastAPI backend is running.
const API_BASE_URL = "http://127.0.0.1:8000";

const SAMPLE_COMPLAINTS = [
  "I was charged twice for the same transaction and the bank won't refund the duplicate charge.",
  "My mortgage application was denied without any clear explanation from the loan officer.",
  "Debt collectors keep calling me at work even after I asked them to stop.",
  "There's a credit inquiry on my report from a loan I never applied for.",
];

const form = document.getElementById("complaint-form");
const textarea = document.getElementById("complaint-text");
const sampleBtn = document.getElementById("sample-btn");
const submitBtn = document.getElementById("submit-btn");
const btnLabel = submitBtn.querySelector(".btn__label");
const btnSpinner = submitBtn.querySelector(".btn__spinner");

const resultSection = document.getElementById("result");
const resultCategory = document.getElementById("result-category");
const resultConfidence = document.getElementById("result-confidence");
const resultLedger = document.getElementById("result-ledger");

const errorSection = document.getElementById("error");
const errorText = document.getElementById("error-text");

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  btnLabel.hidden = isLoading;
  btnSpinner.hidden = !isLoading;
}

function showError(message) {
  resultSection.hidden = true;
  resultSection.classList.remove("is-visible");
  errorText.textContent = message;
  errorSection.hidden = false;
}

function clearError() {
  errorSection.hidden = true;
}

function renderResult(data) {
  clearError();

  resultCategory.textContent = data.category;
  resultConfidence.textContent = `${Math.round(data.confidence * 100)}% confidence`;

  resultLedger.innerHTML = "";
  data.top_predictions.forEach((prediction) => {
    const pct = Math.round(prediction.probability * 100);

    const item = document.createElement("li");
    item.innerHTML = `
      <span class="ledger__name">${prediction.category}</span>
      <span class="ledger__value">${pct}%</span>
      <span class="ledger__bar"><span></span></span>
    `;
    resultLedger.appendChild(item);

    // Animate the bar in after it's in the DOM.
    requestAnimationFrame(() => {
      item.querySelector(".ledger__bar span").style.width = `${pct}%`;
    });
  });

  resultSection.hidden = false;
  requestAnimationFrame(() => resultSection.classList.add("is-visible"));
}

async function classifyComplaint(text) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }

  return response.json();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = textarea.value.trim();
  if (!text) return;

  setLoading(true);
  try {
    const data = await classifyComplaint(text);
    renderResult(data);
  } catch (err) {
    showError(
      err.message === "Failed to fetch"
        ? "Can't reach the API. Is the backend running?"
        : err.message
    );
  } finally {
    setLoading(false);
  }
});

sampleBtn.addEventListener("click", () => {
  const sample =
    SAMPLE_COMPLAINTS[Math.floor(Math.random() * SAMPLE_COMPLAINTS.length)];
  textarea.value = sample;
  textarea.focus();
});

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();

    if (data.model_loaded) {
      statusDot.className = "status-dot ok";
      statusText.textContent = `API connected — running on ${data.device}`;
    } else {
      statusDot.className = "status-dot down";
      statusText.textContent = "API is up but the model hasn't loaded yet";
    }
  } catch {
    statusDot.className = "status-dot down";
    statusText.textContent = "Can't reach the API — start the backend first";
  }
}

checkHealth();
