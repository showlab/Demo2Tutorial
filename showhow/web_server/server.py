"""ShowHow web UI server.

Serves a browser-based recording interface and coordinates with the recorder
service + tutorial generator.  Runs on port 18090 by default.
"""

from __future__ import annotations

import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
import logging


import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Embedded frontend HTML
# ---------------------------------------------------------------------------

_FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ShowHow Recording Control</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --cream:     #F7F3ED;
      --warm-white:#FEFCF9;
      --sand:      #E8E0D4;
      --stone:     #C4B9A8;
      --ink:       #1A1612;
      --ink-light: #3D362E;
      --ink-faded: #6B6158;
      --vermillion:#C4432A;
      --verm-soft: #D4654F;
      --verm-glow: rgba(196,67,42,0.10);
    }

    body {
      font-family: 'DM Sans', system-ui, sans-serif;
      font-weight: 300;
      background: var(--cream);
      color: var(--ink);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: clamp(3rem, 8vh, 6rem) 1.5rem clamp(4rem, 10vh, 8rem);
    }

    /* ── Wordmark ── */
    .logo {
      font-family: 'Cormorant Garamond', Georgia, serif;
      font-weight: 300;
      font-size: clamp(2.8rem, 6vw, 4.2rem);
      letter-spacing: -0.01em;
      color: var(--ink);
    }
    .logo em {
      font-style: italic;
      color: var(--vermillion);
    }
    .tagline {
      font-size: 0.72rem;
      font-weight: 400;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--stone);
      margin-top: 0.55rem;
      margin-bottom: clamp(2.5rem, 6vh, 4.5rem);
    }

    /* ── Card ── */
    .card {
      background: var(--warm-white);
      border: 1px solid rgba(200,185,168,0.25);
      border-radius: 2px;
      box-shadow: 0 4px 24px rgba(26,22,18,0.07);
      padding: clamp(2rem, 4vw, 3rem) clamp(1.75rem, 4vw, 3rem);
      width: 100%;
      max-width: 780px;
      display: grid;
      grid-template-columns: 1fr 1px 1fr;
      gap: 0 2.5rem;
      align-items: center;
    }

    /* ── Left column: topic + record ── */
    .col-record {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1.25rem;
      padding: clamp(1rem, 3vw, 2rem) 0;
    }
    .col-record .field { width: 100%; }

    /* ── Vertical divider ── */
    .col-divider {
      background: rgba(200,185,168,0.3);
      align-self: stretch;
    }

    /* ── Right column: generate ── */
    .col-generate {
      position: relative;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      padding: clamp(1rem, 3vw, 2rem) 0;
    }

    /* ── Frosted overlay ── */
    .col-overlay {
      position: absolute;
      inset: -0.5rem;
      border-radius: 4px;
      backdrop-filter: blur(5px);
      -webkit-backdrop-filter: blur(5px);
      background: rgba(254,252,249,0.72);
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
      opacity: 1;
      transition: opacity 0.3s;
      z-index: 10;
    }
    .col-overlay.hidden {
      opacity: 0;
      pointer-events: none;
    }
    .col-overlay-text {
      font-family: 'Cormorant Garamond', serif;
      font-style: italic;
      font-weight: 300;
      font-size: 1rem;
      color: var(--stone);
      letter-spacing: 0.04em;
      text-align: center;
    }

    @media (max-width: 600px) {
      .card { grid-template-columns: 1fr; gap: 1.5rem 0; }
      .col-divider { display: none; }
      .col-overlay { display: none; }
    }

    /* ── Field ── */
    .field { width: 100%; }
    .field label {
      font-size: 0.62rem;
      font-weight: 500;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--stone);
      display: block;
      margin-bottom: 0.5rem;
    }
    .field input {
      width: 100%;
      border: 1px solid var(--sand);
      border-radius: 100px;
      padding: 0.6rem 1.1rem;
      font-size: 0.88rem;
      font-family: 'DM Sans', sans-serif;
      font-weight: 300;
      background: var(--warm-white);
      color: var(--ink);
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .field input::placeholder { color: var(--stone); }
    .field input:focus {
      border-color: var(--vermillion);
      box-shadow: 0 0 0 3px var(--verm-glow);
    }
    .field input:disabled { background: var(--cream); color: var(--stone); }
    .field select {
      width: 100%;
      border: 1px solid var(--sand);
      border-radius: 100px;
      padding: 0.6rem 1.1rem;
      font-size: 0.88rem;
      font-family: 'DM Sans', sans-serif;
      font-weight: 300;
      background: var(--warm-white);
      color: var(--ink);
      outline: none;
      appearance: none;
      cursor: pointer;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .field select:focus {
      border-color: var(--vermillion);
      box-shadow: 0 0 0 3px var(--verm-glow);
    }
    .field select:disabled { background: var(--cream); color: var(--stone); }

    /* ── Record button ── */
    .btn-record {
      width: 120px; height: 120px;
      border-radius: 50%;
      border: 1.5px solid var(--sand);
      cursor: pointer;
      font-family: 'DM Sans', sans-serif;
      font-size: 0.68rem;
      font-weight: 500;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 7px;
      outline: none;
      transition: transform 0.15s, box-shadow 0.25s, background 0.2s, border-color 0.2s;
      user-select: none;
      background: var(--warm-white);
      color: var(--ink-faded);
    }
    .btn-record:active { transform: scale(0.96); }

    .btn-record.idle {
      background: var(--vermillion);
      border-color: var(--vermillion);
      color: #fff;
      box-shadow: 0 4px 20px rgba(196,67,42,0.22);
    }
    .btn-record.idle:hover {
      background: var(--verm-soft);
      border-color: var(--verm-soft);
      box-shadow: 0 6px 28px rgba(196,67,42,0.32);
      transform: translateY(-1px);
    }
    .btn-record.recording {
      background: var(--warm-white);
      border-color: var(--vermillion);
      color: var(--vermillion);
      animation: pulse 1.8s ease-out infinite;
    }
    @keyframes pulse {
      0%   { box-shadow: 0 0 0 0   rgba(196,67,42,0.35); }
      70%  { box-shadow: 0 0 0 18px rgba(196,67,42,0);   }
      100% { box-shadow: 0 0 0 0   rgba(196,67,42,0);    }
    }
    .btn-record.busy {
      background: var(--sand);
      border-color: var(--sand);
      color: var(--stone);
      cursor: default;
    }
    .btn-icon { font-size: 1.4rem; line-height: 1; }

    /* ── Timer ── */
    .timer {
      font-family: 'Cormorant Garamond', serif;
      font-weight: 300;
      font-size: 2.4rem;
      font-variant-numeric: tabular-nums;
      letter-spacing: 0.06em;
      color: var(--vermillion);
      display: none;
    }
    .timer.visible { display: block; }

    /* ── Status ── */
    .status-text {
      font-size: 0.75rem;
      font-weight: 400;
      letter-spacing: 0.06em;
      color: var(--stone);
      text-align: center;
    }
    .status-text.recording { color: var(--ink-light); font-style: italic; }
    .status-text.error { color: var(--vermillion); }
    .blink { animation: blink 1s step-end infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

    /* ── Divider ── */
    .divider { width: 100%; height: 1px; background: rgba(200,185,168,0.3); }

    /* ── Generate section (col-generate handles layout) ── */

    .btn-generate {
      width: 100%;
      background: var(--ink);
      color: var(--cream);
      border: 1px solid var(--ink);
      border-radius: 100px;
      padding: 0.72rem 1rem;
      font-size: 0.75rem;
      font-weight: 500;
      font-family: 'DM Sans', sans-serif;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      cursor: pointer;
      transition: background 0.2s, color 0.2s;
    }
    .btn-generate:hover {
      background: var(--ink-light);
      border-color: var(--ink-light);
    }
    .btn-generate:disabled {
      background: var(--sand);
      border-color: var(--sand);
      color: var(--stone);
      cursor: default;
    }

    /* ── Progress ── */
    .progress-wrap { display: none; flex-direction: column; align-items: center; gap: 0.7rem; }
    .progress-wrap.visible { display: flex; }
    .spinner {
      width: 22px; height: 22px;
      border: 1.5px solid var(--sand);
      border-top-color: var(--ink-light);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .progress-label {
      font-size: 0.72rem;
      font-weight: 400;
      letter-spacing: 0.08em;
      color: var(--ink-faded);
      text-align: center;
    }

    /* ── Result ── */
    .result-wrap { display: none; flex-direction: column; align-items: center; gap: 0.85rem; }
    .result-wrap.visible { display: flex; }
    .result-badge {
      font-size: 0.62rem;
      font-weight: 500;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--ink-faded);
      border: 1px solid var(--sand);
      border-radius: 100px;
      padding: 0.3rem 1rem;
    }
    .btn-view {
      display: inline-block;
      background: transparent;
      color: var(--vermillion);
      border: 1px solid var(--vermillion);
      border-radius: 100px;
      padding: 0.65rem 2rem;
      font-size: 0.72rem;
      font-weight: 500;
      font-family: 'DM Sans', sans-serif;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      cursor: pointer;
      text-decoration: none;
      transition: background 0.2s, color 0.2s;
    }
    .btn-view:hover {
      background: var(--vermillion);
      color: #fff;
    }
    .session-meta {
      font-size: 0.65rem;
      letter-spacing: 0.06em;
      color: var(--stone);
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="logo">Show<em>How</em></div>
  <div class="tagline">Record &mdash; Generate &mdash; Share</div>

  <div class="card">
    <!-- Left: topic + record -->
    <div class="col-record">
      <div class="field">
        <label for="api-key">OpenAI API Key</label>
        <input id="api-key" type="password" placeholder="Required before recording" autocomplete="off">
      </div>
      <div class="field">
        <label for="topic">Topic</label>
        <input id="topic" type="text" placeholder="e.g. How to create a pivot table" autocomplete="off">
      </div>
      <button id="btn-record" class="btn-record idle" onclick="toggleRecord()">
        <span class="btn-icon" id="btn-icon">&#9210;</span>
        <span id="btn-label">Start</span>
      </button>
      <div id="timer" class="timer">00:00</div>
      <div id="status-text" class="status-text">Ready to record</div>
    </div>

    <!-- Vertical divider -->
    <div class="col-divider"></div>

    <!-- Right: generate -->
    <div class="col-generate">
      <!-- Frosted overlay shown before recording -->
      <div class="col-overlay" id="col-overlay">
        <span class="col-overlay-text">Please record first 💭</span>
      </div>

      <div class="field">
        <label for="model-select">Model</label>
        <select id="model-select" onchange="syncModelInput()">
          <option value="gpt-4o">gpt-4o</option>
          <option value="gpt-4o-mini">gpt-4o-mini</option>
          <option value="gpt-4.1">gpt-4.1</option>
          <option value="gpt-5.1">gpt-5.1</option>
          <option value="gpt-5">gpt-5</option>
          <option value="gpt-5-mini">gpt-5-mini</option>
          <option value="gpt-5-nano">gpt-5-nano</option>
          <option value="custom">Custom model...</option>
        </select>
      </div>
      <div class="field" id="custom-model-wrap" style="display:none">
        <label for="custom-model">Custom Model</label>
        <input id="custom-model" type="text" placeholder="e.g. gpt-4.1">
      </div>

      <button id="btn-generate" class="btn-generate" onclick="generateTutorial()">
        Generate Tutorial
      </button>

      <div id="progress-wrap" class="progress-wrap">
        <div class="spinner"></div>
        <div class="progress-label" id="progress-label">Generating tutorial&hellip;</div>
      </div>

      <div id="result-wrap" class="result-wrap">
        <div class="result-badge">Tutorial ready</div>
        <a id="btn-view" class="btn-view" href="#" target="_blank">Open Tutorial</a>
        <div class="session-meta" id="session-meta"></div>
      </div>
    </div>
  </div>

  <script>
    var state = 'idle';
    var sessionId = null;
    var currentJobId = null;
    var jobPollId = null;
    var timerStart = null;
    var timerInterval = null;

    function pad2(n) { return String(n).padStart(2, '0'); }
    function fmtMs(ms) {
      var s = Math.floor(ms / 1000);
      return pad2(Math.floor(s / 60)) + ':' + pad2(s % 60);
    }
    function tickTimer() {
      if (timerStart) document.getElementById('timer').textContent = fmtMs(Date.now() - timerStart);
    }

    function setUI(s) {
      state = s;
      var btn  = document.getElementById('btn-record');
      var icon = document.getElementById('btn-icon');
      var lbl  = document.getElementById('btn-label');
      var st   = document.getElementById('status-text');
      var tmr  = document.getElementById('timer');
      var bgen = document.getElementById('btn-generate');
      var prg  = document.getElementById('progress-wrap');
      var res  = document.getElementById('result-wrap');
      var inp  = document.getElementById('topic');
      var key  = document.getElementById('api-key');
      var ovl  = document.getElementById('col-overlay');
      var mdl  = document.getElementById('model-select');
      var customModel = document.getElementById('custom-model');

      btn.className  = 'btn-record';
      st.className   = 'status-text';
      tmr.className  = 'timer';
      prg.className  = 'progress-wrap';
      res.className  = 'result-wrap';

      if (s === 'idle') {
        btn.classList.add('idle'); btn.disabled = false;
        icon.textContent = '\\u23FA'; lbl.textContent = 'Start';
        st.textContent = 'Ready to record';
        inp.disabled = false;
        key.disabled = false;
        mdl.disabled = false;
        customModel.disabled = false;
        bgen.disabled = false;
        ovl.classList.remove('hidden');
        clearInterval(timerInterval); timerStart = null;

      } else if (s === 'recording') {
        btn.classList.add('recording'); btn.disabled = false;
        icon.textContent = '\\u23F9'; lbl.textContent = 'Stop';
        st.classList.add('recording');
        st.innerHTML = 'Recording<span class="blink">\u2026</span>';
        tmr.classList.add('visible');
        inp.disabled = true;
        key.disabled = true;
        mdl.disabled = true;
        customModel.disabled = true;
        ovl.classList.remove('hidden');
        if (!timerStart) timerStart = Date.now();
        clearInterval(timerInterval);
        timerInterval = setInterval(tickTimer, 500);

      } else if (s === 'stopped') {
        btn.classList.add('idle'); btn.disabled = false;
        icon.textContent = '\\u23FA'; lbl.textContent = 'New';
        st.textContent = 'Recording complete \u2014 ready to generate';
        inp.disabled = false;
        key.disabled = false;
        mdl.disabled = false;
        customModel.disabled = false;
        clearInterval(timerInterval);
        bgen.disabled = false;
        ovl.classList.add('hidden');

      } else if (s === 'generating') {
        btn.classList.add('busy'); btn.disabled = true;
        st.textContent = 'Generating\u2026';
        bgen.disabled = true;
        key.disabled = true;
        mdl.disabled = true;
        customModel.disabled = true;
        prg.classList.add('visible');
        ovl.classList.add('hidden');

      } else if (s === 'done') {
        btn.classList.add('idle'); btn.disabled = false;
        icon.textContent = '\\u23FA'; lbl.textContent = 'New';
        st.textContent = 'Tutorial ready';
        inp.disabled = false;
        key.disabled = false;
        mdl.disabled = false;
        customModel.disabled = false;
        clearInterval(timerInterval);
        bgen.disabled = false;
        res.classList.add('visible');
        ovl.classList.add('hidden');
      }
    }

    async function toggleRecord() {
      if (state === 'idle' || state === 'stopped' || state === 'done') {
        await startRecording();
      } else if (state === 'recording') {
        await stopRecording();
      }
    }

    function syncModelInput() {
      var select = document.getElementById('model-select');
      var wrap = document.getElementById('custom-model-wrap');
      wrap.style.display = select.value === 'custom' ? 'block' : 'none';
    }

    function selectedModel() {
      var select = document.getElementById('model-select');
      if (select.value !== 'custom') return select.value;
      return document.getElementById('custom-model').value.trim();
    }

    async function startRecording() {
      var topic = document.getElementById('topic').value.trim();
      var apiKey = document.getElementById('api-key').value.trim();
      if (!apiKey) { showError('Enter your OpenAI API key before recording'); return; }
      try {
        var r = await fetch('/api/start', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({topic: topic || null, api_key: apiKey})
        });
        var d = await r.json();
        if (d.error) { showError(d.error); return; }
        sessionId = d.session_id;
        timerStart = Date.now();
        setUI('recording');
      } catch(e) { showError('Cannot reach server'); }
    }

    async function stopRecording() {
      try {
        var r = await fetch('/api/stop', {method: 'POST'});
        var d = await r.json();
        if (d.error) { showError(d.error); return; }
        sessionId = d.session_id || sessionId;
        setUI('stopped');
      } catch(e) { showError('Cannot reach server'); }
    }

    async function generateTutorial() {
      var apiKey = document.getElementById('api-key').value.trim();
      if (!apiKey) { showError('Please enter your OpenAI API key'); return; }
      if (!sessionId) { showError('Please record first'); return; }
      var model = selectedModel();
      if (!model) { showError('Please enter a model name'); return; }
      setUI('generating');
      try {
        var r = await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({session_id: sessionId, api_key: apiKey, model: model})
        });
        var d = await r.json();
        if (!r.ok || d.error || d.detail) {
          showError(d.error || d.detail || 'Generation request failed');
          setUI('stopped');
          return;
        }
        currentJobId = d.job_id;
        pollJob();
      } catch(e) { showError('Cannot reach server'); setUI(sessionId ? 'stopped' : 'idle'); }
    }

    function pollJob() {
      if (jobPollId) clearInterval(jobPollId);
      jobPollId = setInterval(async function() {
        try {
          var r = await fetch('/api/jobs/' + currentJobId);
          var d = await r.json();
          if (d.status === 'done') {
            clearInterval(jobPollId);
            document.getElementById('btn-view').href = '/view/' + sessionId;
            document.getElementById('session-meta').textContent = 'Session: ' + sessionId;
            setUI('done');
          } else if (d.status === 'failed') {
            clearInterval(jobPollId);
            showError('Generation failed: ' + (d.error || 'unknown error'));
            setUI('stopped');
          } else {
            var label = d.stage ? 'Processing: ' + d.stage + '\u2026' : 'Generating tutorial\u2026';
            document.getElementById('progress-label').textContent = label;
          }
        } catch(e) {}
      }, 2000);
    }

    function showError(msg) {
      var st = document.getElementById('status-text');
      st.textContent = msg;
      st.className = 'status-text error';
    }

    // On load: sync state with server
    (async function() {
      try {
        var r = await fetch('/api/status');
        var d = await r.json();
        if (d.recording) {
          sessionId = d.session_id;
          timerStart = Date.now() - Math.round((d.duration || 0) * 1000);
          setUI('recording');
        } else if (d.last_session_id) {
          sessionId = d.last_session_id;
          setUI('stopped');
        }
      } catch(e) {}
    })();
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_service: Any = None  # ShowHowService
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="showhow-gen")
_last_session_id: Optional[str] = None
_last_tutorial_dir: Optional[Path] = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class StartRequest(BaseModel):
    topic: Optional[str] = None
    api_key: Optional[str] = None


class GenerateRequest(BaseModel):
    session_id: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


# ---------------------------------------------------------------------------
# Background generation task
# ---------------------------------------------------------------------------

_logger = logging.getLogger("showhow.web")


def _run_generation(
    job_id: str,
    session_id: str,
    data_path: Optional[str],
    api_key: str,
    model: Optional[str] = None,
) -> None:
    import os

    global _last_tutorial_dir
    selected_model = (model or "gpt-4o").strip()
    # Make the key available to all OpenAI clients in this process
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_MODEL"] = selected_model
    # Reset cached client so it picks up the new key
    try:
        import showhow.tutorial_generator.smart_task_graph as _stg

        _stg.openai_client = None
    except Exception:
        pass
    job = _jobs[job_id]

    def _set_stage(stage: str) -> None:
        job["stage"] = stage

    try:
        _logger.info("Generation started — session=%s", session_id)
        result = _service.generate_tutorial_with_options(
            session_id=session_id,
            data_path=data_path,
            caption_model=selected_model,
            planner_model=selected_model,
            critic_model=selected_model,
            on_stage=_set_stage,
        )

        # Build the self-contained HTML
        tutorial_dir_str = result.get("tutorial_output_dir")
        if tutorial_dir_str:
            from showhow.web_server.html_exporter import export_html

            tutorial_dir = Path(tutorial_dir_str)
            export_html(tutorial_dir, session_id=session_id)
            _last_tutorial_dir = tutorial_dir

        _logger.info("Generation complete — output: %s", tutorial_dir_str)
        job["status"] = "done"
        job["result"] = result
    except Exception as exc:
        _logger.exception("Generation failed for session=%s", session_id)
        job["status"] = "failed"
        job["error"] = str(exc)


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    global _service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _service
        from showhow.runtime.service import ShowHowService

        _service = ShowHowService()
        _service.startup()
        yield
        try:
            _service.shutdown()
        except Exception:
            pass

    app = FastAPI(title="ShowHow Web UI", lifespan=lifespan)

    # -- Frontend -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def frontend() -> HTMLResponse:
        return HTMLResponse(_FRONTEND_HTML)

    # -- Recorder API -------------------------------------------------------

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        try:
            status = _service.recorder_client.status()
            status["last_session_id"] = _last_session_id
            if _last_session_id:
                status["last_data_path"] = _service._session_data_paths.get(
                    _last_session_id
                )
            return JSONResponse(status)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/start")
    async def api_start(req: StartRequest) -> JSONResponse:
        global _last_session_id
        if not req.api_key:
            return JSONResponse(
                {"error": "OpenAI API key is required before recording"},
                status_code=400,
            )
        try:
            result = _service.start_recording(topic=req.topic)
            _last_session_id = result.get("session_id")
            return JSONResponse(result)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)

    @app.post("/api/stop")
    async def api_stop() -> JSONResponse:
        global _last_session_id
        try:
            result = _service.stop_recording()
            _last_session_id = result.get("session_id") or _last_session_id
            return JSONResponse(result)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)

    # -- Tutorial generation ------------------------------------------------

    @app.post("/api/generate")
    async def api_generate(req: GenerateRequest) -> JSONResponse:
        session_id = req.session_id or _last_session_id
        if not session_id:
            raise HTTPException(
                status_code=400, detail="No active session to generate from"
            )
        if not req.api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required")

        data_path = None
        if _service._session_data_paths.get(session_id):
            data_path = _service._session_data_paths[session_id]

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {
            "status": "running",
            "stage": "init",
            "error": None,
            "result": None,
        }
        _executor.submit(
            _run_generation, job_id, session_id, data_path, req.api_key, req.model
        )
        return JSONResponse({"job_id": job_id})

    @app.get("/api/jobs/{job_id}")
    async def api_job_status(job_id: str) -> JSONResponse:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(
            {
                "status": job["status"],
                "stage": job.get("stage"),
                "error": job.get("error"),
            }
        )

    # -- Serve generated tutorial -------------------------------------------

    @app.get("/view/{session_id}", response_class=HTMLResponse)
    async def view_tutorial(session_id: str) -> HTMLResponse:
        # Locate the tutorial HTML file
        tutorial_html = _find_tutorial_html(session_id)
        if tutorial_html is None:
            raise HTTPException(
                status_code=404, detail="Tutorial not found. Generate it first."
            )
        content = tutorial_html.read_text(encoding="utf-8")
        return HTMLResponse(content)

    return app


def _find_tutorial_html(session_id: str) -> Optional[Path]:
    """Search for tutorial.html in known locations."""
    import os

    # Try the known service data path
    if _service and session_id in _service._session_data_paths:
        data_path = Path(_service._session_data_paths[session_id])
        candidates = list(data_path.glob("**/tutorial.html"))
        if candidates:
            return candidates[0]

    # Fallback: search under record root
    record_root_str = os.getenv("SHOWHOW_RECORD_ROOT", "")
    record_root = (
        Path(record_root_str).expanduser()
        if record_root_str
        else Path.home() / "Downloads" / "record_save"
    )
    session_dir = record_root / session_id
    if session_dir.exists():
        candidates = list(session_dir.glob("**/tutorial.html"))
        if candidates:
            return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    host: str = "127.0.0.1",
    port: int = 18090,
    open_browser: bool = True,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if open_browser:
        threading.Timer(1.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
