const elements = {
  healthCard: document.getElementById("health-card"),
  asrPrimary: document.getElementById("asr-primary"),
  asrFallback: document.getElementById("asr-fallback"),
  mtPrimary: document.getElementById("mt-primary"),
  mtLanguage: document.getElementById("mt-language"),
  ttsPrimary: document.getElementById("tts-primary"),
  ttsFallback: document.getElementById("tts-fallback"),
  asrModel: document.getElementById("asr-model"),
  asrLanguage: document.getElementById("asr-language"),
  mtModel: document.getElementById("mt-model"),
  mtDirection: document.getElementById("mt-direction"),
  ttsModel: document.getElementById("tts-model"),
  ttsVoice: document.getElementById("tts-voice"),
  sourceLang: document.getElementById("source-lang"),
  targetLang: document.getElementById("target-lang"),
  sourceText: document.getElementById("source-text"),
  resultStatus: document.getElementById("result-status"),
  resultText: document.getElementById("result-text"),
  translateForm: document.getElementById("translate-form"),
  useRuntimeDirection: document.getElementById("use-runtime-direction"),
  startPipeline: document.getElementById("start-pipeline"),
  stopPipeline: document.getElementById("stop-pipeline"),
  refreshPipeline: document.getElementById("refresh-pipeline"),
  pipelineState: document.getElementById("pipeline-state"),
  pipelinePid: document.getElementById("pipeline-pid"),
  pipelineLogs: document.getElementById("pipeline-logs"),
  pipelineStatusNote: document.getElementById("pipeline-status-note"),
  runtimeSourceLang: document.getElementById("runtime-source-lang"),
  runtimeTargetLang: document.getElementById("runtime-target-lang"),
  directionForm: document.getElementById("direction-form"),
  reloadDirection: document.getElementById("reload-direction"),
  directionStatus: document.getElementById("direction-status"),
  voiceForm: document.getElementById("voice-form"),
  voiceSelect: document.getElementById("voice-select"),
  reloadVoices: document.getElementById("reload-voices"),
  voiceStatus: document.getElementById("voice-status"),
};

let runtimeStack = null;
let pipelinePollTimer = null;
let voiceEntries = [];

function setHealth(status, subtitle, className) {
  const value = elements.healthCard.querySelector(".status-value");
  const sub = elements.healthCard.querySelector(".status-sub");
  value.textContent = status;
  value.className = `status-value ${className || ""}`.trim();
  sub.textContent = subtitle;
}

function formatFallback(value) {
  return value ? `fallback: ${value}` : "fallback: none";
}

function setDirectionStatus(text, className = "") {
  elements.directionStatus.textContent = text;
  elements.directionStatus.className = className;
}

function setVoiceStatus(text, className = "") {
  elements.voiceStatus.textContent = text;
  elements.voiceStatus.className = className;
}

function applyStack(stack) {
  runtimeStack = stack;

  elements.asrPrimary.textContent = stack.routes.asr_primary || "--";
  elements.asrFallback.textContent = formatFallback(stack.routes.asr_fallback);
  elements.mtPrimary.textContent = stack.routes.mt_primary || "--";
  elements.mtLanguage.textContent =
    `${stack.languages.translation.source_name} -> ${stack.languages.translation.target_name}`;
  elements.ttsPrimary.textContent = stack.routes.tts_primary || "--";
  elements.ttsFallback.textContent = formatFallback(stack.routes.tts_fallback);

  elements.asrModel.textContent = stack.models.asr || "--";
  elements.asrLanguage.textContent =
    `language: ${stack.languages.asr.name} (${stack.languages.asr.code})`;
  elements.mtModel.textContent = stack.models.mt || "--";
  elements.mtDirection.textContent =
    `${stack.languages.translation.source_code} -> ${stack.languages.translation.target_code}`;
  elements.ttsModel.textContent = stack.models.tts || "--";
  elements.ttsVoice.textContent = stack.voice.configured
    ? `voice sample: ${stack.voice.reference_sample}`
    : "voice: not configured";

  elements.sourceLang.value = stack.languages.translation.source_code;
  elements.targetLang.value = stack.languages.translation.target_code;
  elements.runtimeSourceLang.value = stack.languages.translation.source_code;
  elements.runtimeTargetLang.value = stack.languages.translation.target_code;
}

function renderVoiceOptions(entries) {
  voiceEntries = entries;
  elements.voiceSelect.innerHTML = "";

  if (!entries.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No saved voices";
    elements.voiceSelect.appendChild(option);
    return;
  }

  entries.forEach((entry, idx) => {
    const option = document.createElement("option");
    option.value = String(idx + 1);
    const activeMark = entry.active ? " [active]" : "";
    option.textContent = `#${idx + 1} ${entry.sample_path}${activeMark}`;
    option.selected = !!entry.active;
    elements.voiceSelect.appendChild(option);
  });
}

function applyPipelineStatus(payload) {
  const status = payload.status || "unknown";
  const pid = payload.pid || "--";
  const logs = Array.isArray(payload.logs) ? payload.logs : [];

  elements.pipelineState.textContent = status;
  elements.pipelinePid.textContent = `PID: ${pid}`;
  elements.pipelineStatusNote.textContent = status;
  elements.pipelineStatusNote.className =
    status === "running" ? "ok" : status === "stopped" ? "" : "loading";
  elements.pipelineLogs.textContent = logs.length
    ? logs.join("\n")
    : "No pipeline logs yet.";
  elements.pipelineLogs.scrollTop = elements.pipelineLogs.scrollHeight;
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    if (data.status === "ok") {
      const subtitle = data.local_translation_model_loaded
        ? "Backend ready, local MT model loaded"
        : "Backend ready, local MT model not loaded yet";
      setHealth("Online", subtitle, "ok");
    } else {
      setHealth("Loading", "Backend is still warming up", "loading");
    }
  } catch (error) {
    setHealth("Offline", "Cannot reach backend", "error");
  }
}

async function loadStack() {
  try {
    const response = await fetch("/api/stack");
    if (!response.ok) {
      throw new Error("Failed to fetch runtime stack.");
    }
    const data = await response.json();
    applyStack(data);
  } catch (error) {
    elements.resultStatus.textContent = "Stack load failed";
    elements.resultStatus.className = "error";
    elements.resultText.textContent = String(error);
  }
}

async function loadPipelineStatus() {
  try {
    const response = await fetch("/api/pipeline/status");
    if (!response.ok) {
      throw new Error("Failed to fetch pipeline status.");
    }
    const data = await response.json();
    applyPipelineStatus(data);
  } catch (error) {
    elements.pipelineStatusNote.textContent = "status error";
    elements.pipelineStatusNote.className = "error";
    elements.pipelineLogs.textContent = String(error);
  }
}

async function loadVoices() {
  try {
    const response = await fetch("/api/voices");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Failed to load voices");
    }
    renderVoiceOptions(data.voices || []);
    setVoiceStatus("Voice list loaded.", "ok");
  } catch (error) {
    setVoiceStatus(String(error), "error");
  }
}

async function controlPipeline(action) {
  const endpoint = action === "start" ? "/api/pipeline/start" : "/api/pipeline/stop";
  elements.pipelineStatusNote.textContent = action === "start" ? "starting..." : "stopping...";
  elements.pipelineStatusNote.className = "loading";

  try {
    const response = await fetch(endpoint, { method: "POST" });
    const data = await response.json();
    applyPipelineStatus({
      status: data.status,
      pid: data.pid,
      logs: elements.pipelineLogs.textContent === "No pipeline logs yet."
        ? []
        : elements.pipelineLogs.textContent.split("\n"),
    });
    await loadPipelineStatus();
  } catch (error) {
    elements.pipelineStatusNote.textContent = "control error";
    elements.pipelineStatusNote.className = "error";
    elements.pipelineLogs.textContent = String(error);
  }
}

async function submitTranslation(event) {
  event.preventDefault();

  const payload = {
    text: elements.sourceText.value.trim(),
    source_lang: elements.sourceLang.value,
    target_lang: elements.targetLang.value,
  };

  if (!payload.text) {
    elements.resultStatus.textContent = "Empty";
    elements.resultStatus.className = "error";
    elements.resultText.textContent = "Please enter some text first.";
    return;
  }

  elements.resultStatus.textContent = "Translating...";
  elements.resultStatus.className = "loading";
  elements.resultText.textContent = "Working...";

  try {
    const response = await fetch("/translate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Translation failed.");
    }

    elements.resultStatus.textContent = "Done";
    elements.resultStatus.className = "ok";
    elements.resultText.textContent = data.translated_text || "(empty)";
    await loadHealth();
  } catch (error) {
    elements.resultStatus.textContent = "Error";
    elements.resultStatus.className = "error";
    elements.resultText.textContent = String(error);
  }
}

async function saveDirection(event) {
  event.preventDefault();

  const payload = {
    source_lang: elements.runtimeSourceLang.value,
    target_lang: elements.runtimeTargetLang.value,
  };

  setDirectionStatus("Saving...", "loading");

  try {
    const response = await fetch("/api/config/languages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Failed to update .env");
    }

    setDirectionStatus(
      `Saved: ${data.source_lang} -> ${data.target_lang}`,
      "ok",
    );
    await loadStack();
  } catch (error) {
    setDirectionStatus(String(error), "error");
  }
}

async function activateSelectedVoice(event) {
  event.preventDefault();

  const index = Number(elements.voiceSelect.value);
  if (!index) {
    setVoiceStatus("No voice selected.", "error");
    return;
  }

  setVoiceStatus("Activating...", "loading");

  try {
    const response = await fetch("/api/voices/activate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ index }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Failed to activate voice");
    }

    setVoiceStatus(data.message || "Voice activated.", "ok");
    await Promise.all([loadVoices(), loadStack()]);
  } catch (error) {
    setVoiceStatus(String(error), "error");
  }
}

function applyRuntimeDirection() {
  if (!runtimeStack) {
    return;
  }
  elements.sourceLang.value = runtimeStack.languages.translation.source_code;
  elements.targetLang.value = runtimeStack.languages.translation.target_code;
  elements.resultStatus.textContent = "Synced";
  elements.resultStatus.className = "ok";
  elements.resultText.textContent =
    "Synced to the current runtime translation direction.";
}

async function reloadDirection() {
  await loadStack();
  setDirectionStatus("Reloaded from current .env", "ok");
}

function startPollingPipeline() {
  if (pipelinePollTimer) {
    clearInterval(pipelinePollTimer);
  }
  pipelinePollTimer = setInterval(loadPipelineStatus, 1500);
}

async function boot() {
  await Promise.all([loadHealth(), loadStack(), loadPipelineStatus(), loadVoices()]);

  elements.translateForm.addEventListener("submit", submitTranslation);
  elements.useRuntimeDirection.addEventListener("click", applyRuntimeDirection);
  elements.startPipeline.addEventListener("click", () => controlPipeline("start"));
  elements.stopPipeline.addEventListener("click", () => controlPipeline("stop"));
  elements.refreshPipeline.addEventListener("click", loadPipelineStatus);
  elements.directionForm.addEventListener("submit", saveDirection);
  elements.reloadDirection.addEventListener("click", reloadDirection);
  elements.voiceForm.addEventListener("submit", activateSelectedVoice);
  elements.reloadVoices.addEventListener("click", loadVoices);

  startPollingPipeline();
}

boot();
