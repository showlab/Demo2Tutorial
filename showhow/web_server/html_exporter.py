"""Convert tutorial pipeline output into a self-contained, editable HTML file."""

from __future__ import annotations

import base64
import json
from pathlib import Path


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _img_b64(img_path: Path) -> str:
    if img_path.exists():
        raw = img_path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode()
    return ""


def export_html(tutorial_output_dir: Path, session_id: str = "") -> Path:
    """
    Build a self-contained editable HTML tutorial from a composed output directory.

    Reads ``dataset.jsonl`` + ``step_images/`` produced by ``quality_composer``.
    Returns the path to the written ``tutorial.html`` file inside *tutorial_output_dir*.
    """
    output_dir = tutorial_output_dir
    dataset_path = output_dir / "dataset.jsonl"
    html_path = output_dir / "tutorial.html"

    # -- title ----------------------------------------------------------
    md_path = output_dir / "index.md"
    title = session_id or "ShowHow Tutorial"
    if md_path.exists():
        for line in md_path.read_text(encoding="utf-8").splitlines():
            stripped = line.lstrip("# ").strip()
            if stripped:
                title = stripped
                break

    # -- parse steps from dataset.jsonl ---------------------------------
    steps: list[dict] = []
    if dataset_path.exists():
        for raw in dataset_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                try:
                    steps.append(json.loads(raw))
                except Exception:
                    pass

    # -- group into chapters --------------------------------------------
    chapters: dict[int, dict] = {}
    for step in steps:
        ch_idx = int(step.get("chapter_index", 1))
        if ch_idx not in chapters:
            chapters[ch_idx] = {
                "title": step.get("chapter_title", f"Chapter {ch_idx}"),
                "steps": [],
            }
        chapters[ch_idx]["steps"].append(step)

    # -- build chapter HTML ---------------------------------------------
    chapters_html_parts: list[str] = []
    for ch_idx in sorted(chapters.keys()):
        ch = chapters[ch_idx]
        steps_html_parts: list[str] = []

        for step in ch["steps"]:
            instruction = _esc(step.get("instruction") or "")
            rationale = _esc(step.get("rationale") or "")
            expected = _esc(step.get("expected_result") or "")
            image_rel = step.get("image_path") or ""
            candidate_rels: list[str] = step.get("candidate_images") or []
            ch_i = step.get("chapter_index", 1)
            st_i = step.get("step_index", 1)
            step_id = f"{ch_i}-{st_i}"
            meta = step.get("highlight_metadata") or {}
            focus_points = []
            for point in (meta.get("clicks") or []) + (meta.get("drag") or []):
                try:
                    x = float(point.get("x"))
                    y = float(point.get("y"))
                    focus_points.append((x, y))
                except Exception:
                    pass

            # default main image (middle candidate or image_path)
            main_src = ""
            if image_rel:
                main_src = _img_b64(output_dir / image_rel)

            # embed all candidate images
            candidate_srcs: list[str] = []
            for cr in candidate_rels:
                s = _img_b64(output_dir / cr)
                if s:
                    candidate_srcs.append(s)

            # if we have candidates, rebuild main_src as middle one
            if candidate_srcs:
                mid = len(candidate_srcs) // 2
                main_src = candidate_srcs[mid]

            img_tag = ""
            if main_src:
                mag_html = ""
                if focus_points:
                    mag_x, mag_y = focus_points[0]
                    mag_html = (
                        f'<div class="magnifier" data-x="{mag_x:.2f}" data-y="{mag_y:.2f}"></div>'
                    )
                img_tag = (
                    f'<div class="image-stage" data-step-id="{step_id}">'
                    f'<img src="{main_src}" alt="Step {ch_i}.{st_i}" '
                    f'class="step-img step-main-img" id="main-{step_id}">'
                    f'{mag_html}'
                    f'</div>'
                    f'<div class="image-tools">'
                    f'<button type="button" onclick="addMagnifier(\'{step_id}\')">Add magnifier</button>'
                    f'<button type="button" onclick="removeMagnifier(\'{step_id}\')">Remove magnifier</button>'
                    f'</div>'
                )

            # filmstrip (only if more than one candidate)
            filmstrip_html = ""
            if len(candidate_srcs) > 1:
                thumbs: list[str] = []
                mid = len(candidate_srcs) // 2
                for ci, cs in enumerate(candidate_srcs):
                    active = " active" if ci == mid else ""
                    thumbs.append(
                        f'<img src="{cs}" class="filmstrip-thumb{active}" '
                        f"onclick=\"selectFrame('{step_id}',this,'{cs}')\" "
                        f'alt="Candidate {ci + 1}" title="Candidate {ci + 1}">'
                    )
                filmstrip_html = (
                    f'<div class="filmstrip" id="film-{step_id}">'
                    + "".join(thumbs)
                    + "</div>"
                )

            meta_parts: list[str] = []
            if rationale:
                meta_parts.append(
                    f'<p class="meta"><span class="meta-label">Why</span>'
                    f'<span contenteditable="true">{rationale}</span></p>'
                )
            if expected:
                meta_parts.append(
                    f'<p class="meta"><span class="meta-label">Expected</span>'
                    f'<span contenteditable="true">{expected}</span></p>'
                )
            meta_html = "\n".join(meta_parts)

            steps_html_parts.append(
                f"""
  <div class="step">
    <div class="step-header">
      <span class="step-num">{ch_i}.{st_i}</span>
      <span class="step-instruction" contenteditable="true">{instruction}</span>
    </div>
    {meta_html}
    {img_tag}
    {filmstrip_html}
  </div>"""
            )

        chapters_html_parts.append(
            f"""
<section class="chapter">
  <h2 contenteditable="true">{_esc(ch["title"])}</h2>
  {"".join(steps_html_parts)}
</section>"""
        )

    if not chapters_html_parts:
        chapters_html_parts.append(
            '<p class="empty-note">No steps were found in this tutorial.</p>'
        )

    chapters_html = "\n".join(chapters_html_parts)

    # -- assemble HTML --------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_esc(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --cream: #F7F3ED;
      --warm-white: #FEFCF9;
      --sand: #E8E0D4;
      --stone: #C4B9A8;
      --ink: #1A1612;
      --ink-light: #3D362E;
      --ink-faded: #6B6158;
      --vermillion: #C4432A;
      --verm-soft: #D4654F;
      --verm-glow: rgba(196,67,42,0.10);
    }}
    body {{
      font-family: 'DM Sans', system-ui, sans-serif;
      font-weight: 300;
      font-size: 16px;
      line-height: 1.65;
      background: var(--cream);
      color: var(--ink);
    }}

    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(254,252,249,0.92);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      color: var(--ink-faded);
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.8rem clamp(1rem, 4vw, 2rem);
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      border-bottom: 1px solid rgba(200,185,168,0.35);
    }}
    .toolbar .brand {{
      font-family: 'Cormorant Garamond', Georgia, serif;
      font-weight: 300;
      font-size: 1.6rem;
      letter-spacing: 0;
      color: var(--ink);
      margin-right: auto;
      line-height: 1;
    }}
    .toolbar .brand em {{ color: var(--vermillion); font-style: italic; }}
    .toolbar .hint {{ color: var(--stone); }}
    .toolbar button {{
      background: var(--warm-white);
      color: var(--ink-light);
      border: 1px solid var(--sand);
      border-radius: 100px;
      padding: 0.45rem 0.95rem;
      font-size: 0.68rem;
      font-weight: 500;
      font-family: inherit;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      cursor: pointer;
      transition: border-color 0.15s, color 0.15s, background 0.15s;
    }}
    .toolbar button:hover {{ border-color: var(--vermillion); color: var(--vermillion); }}

    .doc {{
      max-width: 880px;
      margin: clamp(2.4rem, 6vh, 4rem) auto;
      padding: 0 clamp(1.25rem, 4vw, 2rem) 5rem;
    }}

    h1.title {{
      font-family: 'Cormorant Garamond', Georgia, serif;
      font-size: clamp(2.4rem, 6vw, 4rem);
      font-weight: 300;
      color: var(--ink);
      margin-bottom: 0.55rem;
      line-height: 1.05;
      letter-spacing: 0;
    }}
    h1.title::after {{
      content: "";
      display: block;
      width: 4.5rem;
      height: 1px;
      background: var(--vermillion);
      margin-top: 1rem;
    }}

    .chapter {{
      margin-top: 2.75rem;
    }}
    .chapter h2 {{
      font-size: 0.72rem;
      font-weight: 500;
      color: var(--ink-faded);
      letter-spacing: 0.16em;
      text-transform: uppercase;
      border-bottom: 1px solid rgba(200,185,168,0.45);
      padding-bottom: 0.65rem;
      margin-bottom: 1.25rem;
    }}

    .step {{
      background: var(--warm-white);
      border: 1px solid rgba(200,185,168,0.28);
      border-radius: 2px;
      box-shadow: 0 4px 24px rgba(26,22,18,0.07);
      padding: clamp(1.25rem, 3vw, 1.75rem);
      margin-bottom: 1.2rem;
    }}
    .step-header {{
      display: flex;
      align-items: baseline;
      gap: 0.75rem;
      margin-bottom: 0.6rem;
    }}
    .step-num {{
      flex-shrink: 0;
      background: var(--vermillion);
      color: #fff;
      border-radius: 50%;
      width: 2.05rem;
      height: 2.05rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.8rem;
      font-weight: 700;
    }}
    .step-instruction {{
      font-size: 1.05rem;
      font-weight: 400;
      color: var(--ink);
      flex: 1;
    }}
    .meta {{
      display: flex;
      gap: 0.5rem;
      font-size: 0.875rem;
      color: var(--ink-faded);
      margin-top: 0.3rem;
      align-items: baseline;
    }}
    .meta-label {{
      flex-shrink: 0;
      font-weight: 500;
      font-size: 0.62rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--stone);
      border: 1px solid var(--sand);
      border-radius: 100px;
      padding: 0.08rem 0.45rem;
    }}
    .step-img {{
      display: block;
      width: 100%;
      border-radius: 2px;
      box-shadow: 0 4px 20px rgba(26,22,18,0.09);
      border: 1px solid var(--sand);
    }}
    .image-stage {{
      position: relative;
      margin-top: 1rem;
      overflow: hidden;
      border-radius: 2px;
      touch-action: none;
    }}
    .magnifier {{
      position: absolute;
      left: 62%;
      top: 42%;
      width: 168px;
      height: 168px;
      border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.92);
      box-shadow: 0 8px 28px rgba(26,22,18,0.32), inset 0 0 0 1px rgba(26,22,18,0.18);
      cursor: grab;
      transform: translate(-50%, -50%);
      background-repeat: no-repeat;
      background-size: auto;
      z-index: 3;
    }}
    .magnifier:active {{ cursor: grabbing; }}
    .image-tools {{
      display: flex;
      justify-content: flex-end;
      gap: 0.45rem;
      margin-top: 0.45rem;
    }}
    .image-tools button {{
      background: var(--warm-white);
      color: var(--ink-faded);
      border: 1px solid var(--sand);
      border-radius: 100px;
      padding: 0.32rem 0.7rem;
      font-size: 0.62rem;
      font-family: inherit;
      font-weight: 500;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      cursor: pointer;
    }}
    .image-tools button:hover {{ border-color: var(--vermillion); color: var(--vermillion); }}

    .filmstrip {{
      display: flex;
      gap: 0.35rem;
      margin-top: 0.6rem;
      overflow-x: auto;
      padding: 0.3rem 0.1rem;
      scrollbar-width: thin;
    }}
    .filmstrip-thumb {{
      height: 64px;
      width: auto;
      flex-shrink: 0;
      border-radius: 2px;
      border: 2px solid var(--sand);
      cursor: pointer;
      opacity: 0.65;
      transition: opacity 0.12s, border-color 0.12s;
      object-fit: cover;
    }}
    .filmstrip-thumb:hover {{ opacity: 0.9; border-color: var(--stone); }}
    .filmstrip-thumb.active {{ opacity: 1; border-color: var(--vermillion); }}

    [contenteditable]:focus {{
      outline: 2px dashed var(--vermillion);
      border-radius: 3px;
      background: var(--verm-glow);
    }}
    [contenteditable]:hover:not(:focus) {{
      outline: 1px dashed var(--stone);
      border-radius: 3px;
    }}

    .empty-note {{
      color: var(--stone);
      font-style: italic;
      margin-top: 2rem;
      text-align: center;
    }}

    @media print {{
      .toolbar {{ display: none; }}
      .filmstrip {{ display: none; }}
      .image-tools {{ display: none; }}
      .magnifier {{ box-shadow: inset 0 0 0 1px rgba(26,22,18,0.18); }}
      body {{ background: #fff; }}
      .doc {{ margin: 0; padding: 1cm 1.5cm; max-width: none; }}
      .step {{ box-shadow: none; border: 1px solid #ddd; break-inside: avoid; }}
      [contenteditable]:focus, [contenteditable]:hover {{ outline: none; background: none; }}
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <span class="brand">Show<em>How</em></span>
    <span class="hint">Click text to edit</span>
    <button onclick="window.print()">Print / Save PDF</button>
    <button onclick="downloadHTML()">Download HTML</button>
  </div>

  <div class="doc" id="doc">
    <h1 class="title" contenteditable="true">{_esc(title)}</h1>
    {chapters_html}
  </div>

  <script>
    const MAGNIFIER_ZOOM = 1.8;

    function selectFrame(stepId, thumbEl, src) {{
      const main = document.getElementById('main-' + stepId);
      if (main) main.src = src;
      const stage = main ? main.closest('.image-stage') : null;
      if (stage) updateMagnifier(stage);
      const film = document.getElementById('film-' + stepId);
      if (film) film.querySelectorAll('.filmstrip-thumb').forEach(t => t.classList.remove('active'));
      thumbEl.classList.add('active');
    }}

    function ensureMagnifier(stage, x, y) {{
      let mag = stage.querySelector('.magnifier');
      if (!mag) {{
        mag = document.createElement('div');
        mag.className = 'magnifier';
        stage.appendChild(mag);
      }}
      mag.dataset.x = String(x);
      mag.dataset.y = String(y);
      bindMagnifier(stage, mag);
      updateMagnifier(stage);
      return mag;
    }}

    function addMagnifier(stepId) {{
      const stage = document.querySelector('.image-stage[data-step-id="' + stepId + '"]');
      if (!stage) return;
      const img = stage.querySelector('.step-main-img');
      const iw = img ? (img.naturalWidth || img.clientWidth || 1) : 1;
      const ih = img ? (img.naturalHeight || img.clientHeight || 1) : 1;
      ensureMagnifier(stage, iw / 2, ih / 2);
    }}

    function removeMagnifier(stepId) {{
      const stage = document.querySelector('.image-stage[data-step-id="' + stepId + '"]');
      if (!stage) return;
      const mag = stage.querySelector('.magnifier');
      if (mag) mag.remove();
    }}

    function updateMagnifier(stage) {{
      const img = stage.querySelector('.step-main-img');
      const mag = stage.querySelector('.magnifier');
      if (!img || !mag) return;
      const iw = img.naturalWidth || img.clientWidth || 1;
      const ih = img.naturalHeight || img.clientHeight || 1;
      const x = parseFloat(mag.dataset.x || String(iw / 2));
      const y = parseFloat(mag.dataset.y || String(ih / 2));
      const stageRect = stage.getBoundingClientRect();
      const imgRect = img.getBoundingClientRect();
      const imgLeft = imgRect.left - stageRect.left;
      const imgTop = imgRect.top - stageRect.top;
      const imageX = Math.max(0, Math.min(imgRect.width, (x / iw) * imgRect.width));
      const imageY = Math.max(0, Math.min(imgRect.height, (y / ih) * imgRect.height));
      const displayX = imgLeft + imageX;
      const displayY = imgTop + imageY;
      const magW = mag.offsetWidth || 168;
      const magH = mag.offsetHeight || 168;
      const bgX = (magW / 2) - (imageX * MAGNIFIER_ZOOM);
      const bgY = (magH / 2) - (imageY * MAGNIFIER_ZOOM);
      mag.style.left = displayX + 'px';
      mag.style.top = displayY + 'px';
      mag.style.backgroundImage = 'url("' + img.src + '")';
      mag.style.backgroundSize = (imgRect.width * MAGNIFIER_ZOOM) + 'px ' + (imgRect.height * MAGNIFIER_ZOOM) + 'px';
      mag.style.backgroundPosition = bgX + 'px ' + bgY + 'px';
    }}

    function bindMagnifier(stage, mag) {{
      if (mag.dataset.bound === '1') return;
      mag.dataset.bound = '1';
      function moveTo(clientX, clientY) {{
        const img = stage.querySelector('.step-main-img');
        const rect = img ? img.getBoundingClientRect() : stage.getBoundingClientRect();
        const iw = img ? (img.naturalWidth || rect.width || 1) : 1;
        const ih = img ? (img.naturalHeight || rect.height || 1) : 1;
        const x = Math.max(0, Math.min(iw, ((clientX - rect.left) / rect.width) * iw));
        const y = Math.max(0, Math.min(ih, ((clientY - rect.top) / rect.height) * ih));
        mag.dataset.x = x.toFixed(2);
        mag.dataset.y = y.toFixed(2);
        updateMagnifier(stage);
      }}
      mag.addEventListener('pointerdown', ev => {{
        ev.preventDefault();
        mag.setPointerCapture(ev.pointerId);
        moveTo(ev.clientX, ev.clientY);
        const onMove = e => moveTo(e.clientX, e.clientY);
        const onUp = () => {{
          mag.removeEventListener('pointermove', onMove);
          mag.removeEventListener('pointerup', onUp);
        }};
        mag.addEventListener('pointermove', onMove);
        mag.addEventListener('pointerup', onUp);
      }});
    }}

    function initMagnifiers() {{
      document.querySelectorAll('.image-stage').forEach(stage => {{
        const img = stage.querySelector('.step-main-img');
        const mag = stage.querySelector('.magnifier');
        if (!img) return;
        const refresh = () => updateMagnifier(stage);
        img.addEventListener('load', refresh);
        if (mag) bindMagnifier(stage, mag);
        refresh();
      }});
    }}

    function downloadHTML() {{
      const clone = document.documentElement.cloneNode(true);
      const tb = clone.querySelector('.toolbar');
      if (tb) tb.remove();
      const blob = new Blob(['<!DOCTYPE html>\\n' + clone.outerHTML], {{type: 'text/html'}});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = {json.dumps(title.replace(" ", "_") + ".html")};
      a.click();
    }}

    initMagnifiers();
  </script>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    return html_path
