const elements = {
  connectionStatus: document.getElementById("connection-status"),
  connectionText: document.getElementById("connection-text"),
  serviceStatus: document.getElementById("service-status"),
  serviceSubtitle: document.getElementById("service-subtitle"),
  routeAsrBadge: document.getElementById("route-asr-badge"),
  routeMtBadge: document.getElementById("route-mt-badge"),
  routeTtsBadge: document.getElementById("route-tts-badge"),
  pipelineState: document.getElementById("pipeline-state"),
  pipelinePid: document.getElementById("pipeline-pid"),
  mtDirectionCard: document.getElementById("mt-direction-card"),
  asrLanguageCard: document.getElementById("asr-language-card"),
  ttsVoiceCard: document.getElementById("tts-voice-card"),
  ttsModelCard: document.getElementById("tts-model-card"),
  feedStatus: document.getElementById("feed-status"),
  turnFeed: document.getElementById("turn-feed"),
  turnFeedEmpty: document.getElementById("turn-feed-empty"),
  asrModel: document.getElementById("asr-model"),
  asrLanguage: document.getElementById("asr-language"),
  mtModel: document.getElementById("mt-model"),
  mtDirection: document.getElementById("mt-direction"),
  ttsModel: document.getElementById("tts-model"),
  ttsVoice: document.getElementById("tts-voice"),
  pipelineStatusNote: document.getElementById("pipeline-status-note"),
  startPipeline: document.getElementById("start-pipeline"),
  stopPipeline: document.getElementById("stop-pipeline"),
  refreshPipeline: document.getElementById("refresh-pipeline"),
  pipelineLogs: document.getElementById("pipeline-logs"),
  consoleLogMeta: document.getElementById("console-log-meta"),
  directionForm: document.getElementById("direction-form"),
  runtimeSourceLang: document.getElementById("runtime-source-lang"),
  runtimeTargetLang: document.getElementById("runtime-target-lang"),
  reloadDirection: document.getElementById("reload-direction"),
  directionStatus: document.getElementById("direction-status"),
  voiceForm: document.getElementById("voice-form"),
  voiceSelect: document.getElementById("voice-select"),
  reloadVoices: document.getElementById("reload-voices"),
  voiceStatus: document.getElementById("voice-status"),
  translateForm: document.getElementById("translate-form"),
  sourceLang: document.getElementById("source-lang"),
  targetLang: document.getElementById("target-lang"),
  sourceText: document.getElementById("source-text"),
  useRuntimeDirection: document.getElementById("use-runtime-direction"),
  resultStatus: document.getElementById("result-status"),
  resultText: document.getElementById("result-text"),
  toastHost: document.getElementById("toast-host"),
  sheetOverlay: document.getElementById("sheet-overlay"),
};

const state = {
  runtimeStack: null,
  pipelinePollTimer: null,
  pipelineLogFingerprint: "",
  speakerLayout: new Map(),
  nextSpeakerSlot: 0,
};

function setConnectionStatus(status, text) {
  elements.connectionStatus.className = `connection-status ${status}`;
  elements.connectionText.textContent = text;
}

function setHealth(status, subtitle, className = "") {
  elements.serviceStatus.textContent = status;
  elements.serviceStatus.className = `hero-stat-value ${className}`.trim();
  elements.serviceSubtitle.textContent = subtitle;

  if (className === "ok") {
    setConnectionStatus("connected", "后端已连接");
  } else if (className === "loading") {
    setConnectionStatus("connecting", "连接中");
  } else if (className === "error") {
    setConnectionStatus("error", "后端不可达");
  } else {
    setConnectionStatus("disconnected", "未连接");
  }
}

function setInlineStatus(element, text, className = "") {
  element.textContent = text;
  element.className = `inline-status-card ${className}`.trim();
}

function formatFallback(value) {
  return value || "none";
}

function clipText(text, limit = 88) {
  const normalized = (text || "").replace(/\s+/g, " ").trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, limit - 1)}…`;
}

function applyStack(stack) {
  state.runtimeStack = stack;

  elements.routeAsrBadge.textContent = `ASR ${stack.routes.asr_primary || "--"}`;
  elements.routeMtBadge.textContent = `MT ${stack.routes.mt_primary || "--"}`;
  elements.routeTtsBadge.textContent = `TTS ${stack.routes.tts_primary || "--"}`;

  const directionText =
    `${stack.languages.translation.source_code} -> ${stack.languages.translation.target_code}`;
  elements.mtDirectionCard.textContent = directionText;
  elements.asrLanguageCard.textContent =
    `ASR ${stack.languages.asr.name} (${stack.languages.asr.code})`;
  elements.ttsVoiceCard.textContent = stack.voice.current_voice || "未配置";
  elements.ttsModelCard.textContent = stack.models.tts || "--";

  elements.asrModel.textContent = stack.models.asr || "--";
  elements.asrLanguage.textContent =
    `language: ${stack.languages.asr.name} (${stack.languages.asr.code})`;
  elements.mtModel.textContent = stack.models.mt || "--";
  elements.mtDirection.textContent = `${stack.languages.translation.source_name} -> ${stack.languages.translation.target_name}`;
  elements.ttsModel.textContent = stack.models.tts || "--";
  elements.ttsVoice.textContent = stack.voice.configured
    ? `${stack.voice.current_voice || "configured"}`
    : "voice: not configured";

  elements.runtimeSourceLang.value = stack.languages.translation.source_code;
  elements.runtimeTargetLang.value = stack.languages.translation.target_code;
  elements.sourceLang.value = stack.languages.translation.source_code;
  elements.targetLang.value = stack.languages.translation.target_code;
}

function renderVoiceOptions(entries) {
  elements.voiceSelect.innerHTML = "";

  if (!entries.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No saved voices";
    elements.voiceSelect.appendChild(option);
    return;
  }

  entries.forEach((entry, index) => {
    const option = document.createElement("option");
    option.value = String(index + 1);
    option.selected = !!entry.active;
    option.textContent =
      `#${index + 1} ${entry.sample_path}${entry.active ? " [active]" : ""}`;
    elements.voiceSelect.appendChild(option);
  });
}

function extractField(segment, key) {
  const pattern = new RegExp(`${key}=([^|]+?)(?= \\| [a-z0-9_]+=|$)`, "i");
  const match = segment.match(pattern);
  return match ? match[1].trim() : "";
}

function parseTurnLine(line) {
  const turnIndex = line.indexOf("turn | ");
  if (turnIndex === -1) {
    return null;
  }

  const timeMatch = line.match(/^(\d{2}:\d{2}:\d{2})/);
  const body = line.slice(turnIndex);
  const asrMarker = " | asr=";
  const mtMarker = " | mt=";
  const asrIndex = body.indexOf(asrMarker);
  const mtIndex = body.indexOf(mtMarker);

  if (asrIndex === -1 || mtIndex === -1 || mtIndex <= asrIndex) {
    return null;
  }

  const metaSegment = body.slice(0, asrIndex);
  const asrText = body.slice(asrIndex + asrMarker.length, mtIndex).trim();
  const mtText = body.slice(mtIndex + mtMarker.length).trim();

  return {
    time: timeMatch ? timeMatch[1] : "--:--:--",
    id: extractField(metaSegment, "id"),
    speaker: extractField(metaSegment, "speaker") || "speaker_unknown",
    route: extractField(metaSegment, "route") || "none",
    voice: extractField(metaSegment, "voice") || "default",
    top1: extractField(metaSegment, "top1") || "none:0.000",
    top2: extractField(metaSegment, "top2") || "none:0.000",
    asr: asrText,
    mt: mtText,
  };
}

function getSpeakerSide(speakerId) {
  if (state.speakerLayout.has(speakerId)) {
    return state.speakerLayout.get(speakerId);
  }

  const side = state.nextSpeakerSlot === 0 ? "right" : "left";
  state.speakerLayout.set(speakerId, side);
  state.nextSpeakerSlot += 1;
  return side;
}

function formatSpeakerAvatar(speakerId) {
  const normalized = String(speakerId || "").trim();
  if (normalized.startsWith("speaker_")) {
    return normalized.replace("speaker_", "S");
  }
  if (normalized.startsWith("guest_")) {
    return normalized.replace("guest_", "G");
  }
  return normalized || "?";
}

function renderTurnFeedFromLogs(logs) {
  const turns = logs
    .map((line) => parseTurnLine(line))
    .filter(Boolean);

  state.speakerLayout.clear();
  state.nextSpeakerSlot = 0;

  if (!turns.length) {
    elements.turnFeed.innerHTML = "";
    elements.turnFeed.appendChild(elements.turnFeedEmpty);
    elements.feedStatus.textContent = "等待主流程输出";
    return;
  }

  elements.turnFeed.innerHTML = "";
  const latestTurn = turns[turns.length - 1];
  elements.feedStatus.textContent =
    `最近一条 turn: ${latestTurn.speaker} -> ${latestTurn.route}`;

  turns.forEach((turn) => {
    const side = getSpeakerSide(turn.speaker);
    const avatarText = formatSpeakerAvatar(turn.speaker);

    const row = document.createElement("article");
    row.className = `turn-row ${side}`;
    row.innerHTML = `
      <div class="turn-avatar">${avatarText}</div>
      <div class="turn-body">
        <div class="speaker-line">
          <span>${escapeHtml(turn.speaker)}</span>
          <span class="route-pill">${escapeHtml(turn.route)}</span>
        </div>
        <div class="bubble bubble-source">
          <div class="bubble-label">ASR</div>
          <p class="bubble-text">${escapeHtml(turn.asr)}</p>
        </div>
        <div class="bubble bubble-target">
          <div class="bubble-label">MT</div>
          <p class="bubble-text">${escapeHtml(turn.mt)}</p>
        </div>
        <div class="turn-meta">
          <span class="turn-meta-chip">${escapeHtml(turn.time)}</span>
          <span class="turn-meta-chip">top1 ${escapeHtml(turn.top1)}</span>
          <span class="turn-meta-chip">top2 ${escapeHtml(turn.top2)}</span>
        </div>
      </div>
    `;
    elements.turnFeed.appendChild(row);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function applyPipelineStatus(payload) {
  const status = payload.status || "unknown";
  const pid = payload.pid || "--";
  const logs = Array.isArray(payload.logs) ? payload.logs : [];
  const compactLogs = logs.slice(-120);
  const fingerprint = compactLogs.join("\n");

  elements.pipelineState.textContent = status;
  elements.pipelinePid.textContent = `PID: ${pid}`;
  elements.pipelineStatusNote.textContent =
    status === "running" ? "主流程运行中" : status === "stopped" ? "主流程已停止" : status;
  elements.pipelineStatusNote.className =
    `speaking-indicator ${status === "running" ? "ok" : status === "stopped" ? "" : "loading"}`.trim();

  elements.consoleLogMeta.textContent =
    `${compactLogs.length} lines / ${status}`;
  elements.pipelineLogs.textContent = compactLogs.length
    ? compactLogs.join("\n")
    : "No pipeline logs yet.";
  elements.pipelineLogs.scrollTop = elements.pipelineLogs.scrollHeight;

  if (state.pipelineLogFingerprint !== fingerprint) {
    state.pipelineLogFingerprint = fingerprint;
    renderTurnFeedFromLogs(logs);
  }
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();

    if (data.status === "ok") {
      const subtitle = data.local_translation_model_loaded
        ? "Backend ready, local MT loaded"
        : "Backend ready, local MT idle";
      setHealth("Online", subtitle, "ok");
      return;
    }

    setHealth("Loading", "Backend is still warming up", "loading");
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
    elements.pipelineStatusNote.textContent = "状态读取失败";
    elements.pipelineStatusNote.className = "speaking-indicator error";
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
    setInlineStatus(elements.voiceStatus, "Voice list loaded.", "ok");
  } catch (error) {
    setInlineStatus(elements.voiceStatus, String(error), "error");
  }
}

async function controlPipeline(action) {
  const endpoint = action === "start" ? "/api/pipeline/start" : "/api/pipeline/stop";
  elements.pipelineStatusNote.textContent = action === "start" ? "正在启动..." : "正在停止...";
  elements.pipelineStatusNote.className = "speaking-indicator loading";

  try {
    const response = await fetch(endpoint, { method: "POST" });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `Failed to ${action} pipeline.`);
    }

    showToast(data.message || "操作已提交");
    await loadPipelineStatus();
  } catch (error) {
    elements.pipelineStatusNote.textContent = String(error);
    elements.pipelineStatusNote.className = "speaking-indicator error";
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
    elements.resultText.textContent = "请先输入一句话。";
    return;
  }

  elements.resultStatus.textContent = "Translating...";
  elements.resultStatus.className = "loading";
  elements.resultText.textContent = "Working...";

  try {
    const response = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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

  setInlineStatus(elements.directionStatus, "Saving...", "loading");

  try {
    const response = await fetch("/api/config/languages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_lang: elements.runtimeSourceLang.value,
        target_lang: elements.runtimeTargetLang.value,
      }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Failed to update .env");
    }

    setInlineStatus(
      elements.directionStatus,
      `Saved: ${data.source_lang} -> ${data.target_lang}`,
      "ok",
    );
    showToast("语言方向已更新");
    await loadStack();
  } catch (error) {
    setInlineStatus(elements.directionStatus, String(error), "error");
  }
}

async function activateSelectedVoice(event) {
  event.preventDefault();

  const index = Number(elements.voiceSelect.value);
  if (!index) {
    setInlineStatus(elements.voiceStatus, "No voice selected.", "error");
    return;
  }

  setInlineStatus(elements.voiceStatus, "Activating...", "loading");

  try {
    const response = await fetch("/api/voices/activate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Failed to activate voice");
    }

    setInlineStatus(elements.voiceStatus, data.message || "Voice activated.", "ok");
    showToast(data.message || "Voice activated");
    await Promise.all([loadVoices(), loadStack()]);
  } catch (error) {
    setInlineStatus(elements.voiceStatus, String(error), "error");
  }
}

function applyRuntimeDirection() {
  if (!state.runtimeStack) {
    return;
  }

  elements.sourceLang.value = state.runtimeStack.languages.translation.source_code;
  elements.targetLang.value = state.runtimeStack.languages.translation.target_code;
  elements.resultStatus.textContent = "Synced";
  elements.resultStatus.className = "ok";
  elements.resultText.textContent = "已同步到当前运行时翻译方向。";
}

async function reloadDirection() {
  await loadStack();
  setInlineStatus(elements.directionStatus, "Reloaded from current .env", "ok");
}

function openSheet(sheetId) {
  closeSheets();
  const sheet = document.getElementById(sheetId);
  if (!sheet) {
    return;
  }
  sheet.classList.add("active");
  sheet.setAttribute("aria-hidden", "false");
  elements.sheetOverlay.classList.add("active");
}

function closeSheets() {
  document.querySelectorAll(".sheet.active").forEach((sheet) => {
    sheet.classList.remove("active");
    sheet.setAttribute("aria-hidden", "true");
  });
  elements.sheetOverlay.classList.remove("active");
}

function bindSheetTriggers() {
  document.querySelectorAll("[data-sheet-open]").forEach((button) => {
    button.addEventListener("click", () => openSheet(button.dataset.sheetOpen));
  });

  document.querySelectorAll("[data-sheet-close]").forEach((button) => {
    button.addEventListener("click", closeSheets);
  });

  elements.sheetOverlay.addEventListener("click", closeSheets);
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  elements.toastHost.appendChild(toast);

  window.setTimeout(() => {
    toast.remove();
  }, 2400);
}

function startPollingPipeline() {
  if (state.pipelinePollTimer) {
    window.clearInterval(state.pipelinePollTimer);
  }

  state.pipelinePollTimer = window.setInterval(loadPipelineStatus, 1500);
}

async function boot() {
  bindSheetTriggers();

  await Promise.all([loadHealth(), loadStack(), loadPipelineStatus(), loadVoices()]);

  elements.directionForm.addEventListener("submit", saveDirection);
  elements.reloadDirection.addEventListener("click", reloadDirection);
  elements.voiceForm.addEventListener("submit", activateSelectedVoice);
  elements.reloadVoices.addEventListener("click", loadVoices);
  elements.translateForm.addEventListener("submit", submitTranslation);
  elements.useRuntimeDirection.addEventListener("click", applyRuntimeDirection);
  elements.startPipeline.addEventListener("click", () => controlPipeline("start"));
  elements.stopPipeline.addEventListener("click", () => controlPipeline("stop"));
  elements.refreshPipeline.addEventListener("click", loadPipelineStatus);

  startPollingPipeline();
}

boot();
