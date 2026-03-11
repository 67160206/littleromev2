"""
Littlelome AI Vision — Streamlit App
UI เหมือน HTML ทุกอย่าง · ไม่ใช้ cv2

requirements.txt:
  streamlit
  ultralytics
  pillow
  pandas
  numpy
"""

import streamlit as st
import time, os, json, tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────
# PAGE CONFIG — ต้องมาก่อนทุกอย่าง
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Littlelome AI Vision",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — copy จาก HTML ทุก token
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:#0d1117; --bg2:#161b22; --bg3:#1c2230; --bg4:#21262d;
  --border:#30363d; --border2:#3a4049;
  --text:#e6edf3; --text2:#8b949e; --text3:#484f58;
  --blue:#2188ff; --red:#f85149; --green:#3fb950; --orange:#e3b341;
  --font:'Syne',sans-serif; --mono:'JetBrains Mono',monospace;
}

/* ── GLOBAL ── */
*, html, body { font-family: var(--font) !important; }
.stApp { background: var(--bg) !important; color: var(--text) !important; }
#MainMenu, footer { visibility: hidden; }

/* ── REMOVE STREAMLIT TOP PADDING ── */
[data-testid="stAppViewBlockContainer"] { padding-top: 0 !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Nav buttons in sidebar */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: 1px solid transparent !important;
  color: var(--text2) !important;
  text-align: left !important;
  width: 100% !important;
  border-radius: 6px !important;
  padding: 8px 12px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  margin-bottom: 2px !important;
  justify-content: flex-start !important;
  gap: 8px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--bg3) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] .stButton > button:focus {
  box-shadow: none !important;
}

/* ── METRICS (fallback styling) ── */
div[data-testid="metric-container"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 16px !important;
}

/* ── MAIN BUTTONS ── */
.stButton > button {
  background: var(--bg4) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 6px !important;
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
}
.stButton > button:hover { background: var(--bg3) !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary { font-weight: 600 !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--bg2) !important;
  border: 2px dashed var(--border2) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--blue) !important; }

/* ── SLIDER ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: var(--blue) !important; }

/* ── SELECT ── */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { background: var(--bg2) !important; }

/* ───── CUSTOM HTML COMPONENTS ───── */

/* Full-width top header bar */
.top-bar {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 0 20px;
  height: 52px;
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 -1rem 0 -1rem;
  position: sticky;
  top: 0;
  z-index: 999;
}
.logo-icon {
  width: 30px; height: 30px;
  background: linear-gradient(135deg, #2188ff, #1a5fb4);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; flex-shrink: 0;
}
.logo-text { font-size: 15px; font-weight: 800; letter-spacing: .3px; }
.logo-text em { font-style: normal; color: var(--blue); }
.sep { width: 1px; height: 24px; background: var(--border); margin: 0 4px; }
.conn-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green); box-shadow: 0 0 8px var(--green);
  display: inline-block; animation: pulse-g 2s infinite;
}
@keyframes pulse-g { 0%,100%{opacity:1} 50%{opacity:.4} }
.conn-label { font-family: var(--mono); font-size: 12px; color: var(--text2); }
.hdr-spacer { flex: 1; }
.hdr-time { font-family: var(--mono); font-size: 12px; color: var(--text2); }
.avatar {
  width: 30px; height: 30px; border-radius: 50%;
  background: linear-gradient(135deg, #2188ff, #7c3aed);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: white;
}

/* Sidebar logo header */
.sb-logo {
  padding: 14px 12px 12px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
}
.sb-logo-icon {
  width: 28px; height: 28px;
  background: linear-gradient(135deg, #2188ff, #1a5fb4);
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px;
}
.sb-logo-name { font-size: 13px; font-weight: 800; }
.sb-logo-name em { font-style: normal; color: var(--blue); }
.sb-logo-sub { font-size: 10px; color: var(--text3); }

/* Sidebar section label */
.nav-section {
  font-size: 10px; font-weight: 700;
  color: var(--text3); letter-spacing: 1px;
  text-transform: uppercase;
  padding: 10px 12px 4px;
}

/* Active nav item */
.nav-active {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 12px; border-radius: 6px;
  font-size: 13px; font-weight: 600;
  color: var(--blue) !important;
  background: rgba(33,136,255,.12);
  border: 1px solid rgba(33,136,255,.2);
  margin-bottom: 2px;
  cursor: default;
}

/* Sidebar bottom stats */
.sb-stats {
  display: flex; gap: 8px;
  padding: 0 10px 10px;
}
.sb-stat-box {
  flex: 1; background: var(--bg4);
  border-radius: 8px; padding: 10px;
  text-align: center;
}
.sb-stat-val { font-family: var(--mono); font-size: 18px; font-weight: 700; }
.sb-stat-label { font-size: 10px; color: var(--text3); margin-top: 2px; }
.sb-footer { font-size: 10px; color: var(--text3); text-align: center; padding: 8px 0 12px; }
.model-status { font-size: 11px; padding: 4px 12px 10px; }

/* KPI CARD — exact copy from HTML */
.kpi-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 20px;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent, var(--blue));
}
.kpi-label {
  font-size: 11px; font-weight: 600; color: var(--text2);
  letter-spacing: .5px; text-transform: uppercase; margin-bottom: 8px;
}
.kpi-val { font-size: 30px; font-weight: 800; font-family: var(--mono); }
.kpi-sub { font-size: 11px; color: var(--text2); margin-top: 4px; }
.kpi-icon {
  position: absolute; right: 16px; top: 50%;
  transform: translateY(-50%);
  font-size: 40px; opacity: .12;
}

/* Section card */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 16px;
}
.card-header {
  display: flex; align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.card-title {
  font-size: 13px; font-weight: 700;
  display: flex; align-items: center; gap: 8px;
}

/* System status rows */
.sys-row {
  display: flex; align-items: center;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid var(--bg4);
  font-size: 13px;
}
.sys-row:last-child { border: none; }
.sdot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block; margin-right: 5px;
}
.sdot-g { background: var(--green); box-shadow: 0 0 6px var(--green); }
.sdot-o { background: var(--orange); box-shadow: 0 0 6px var(--orange); }
.sdot-x { background: var(--text3); }

/* Badges */
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: 20px;
  font-size: 11px; font-weight: 700; letter-spacing: .3px;
}
.badge-info   { background: rgba(33,136,255,.15); color: var(--blue);   border: 1px solid rgba(33,136,255,.2); }
.badge-ok     { background: rgba(63,185,80,.15);  color: var(--green);  border: 1px solid rgba(63,185,80,.2); }
.badge-fault  { background: rgba(248,81,73,.15);  color: var(--red);    border: 1px solid rgba(248,81,73,.2); }
.badge-outline{ background: transparent; color: var(--text2); border: 1px solid var(--border); cursor:pointer; }
.badge-outline:hover { background: var(--bg3); color: var(--text); }

/* Result boxes */
.fault-box {
  background: #2d1515; border: 2px solid var(--red);
  border-radius: 10px; padding: 16px 20px; margin: 10px 0;
}
.ok-box {
  background: #0e2a1a; border: 2px solid var(--green);
  border-radius: 10px; padding: 16px 20px; margin: 10px 0;
}
.detect-item {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 6px; padding: 10px 14px; margin: 5px 0;
  display: flex; align-items: center; justify-content: space-between;
}
.detect-type { font-size: 13px; font-weight: 600; }
.detect-conf { font-family: var(--mono); font-size: 13px; color: var(--text2); }

/* Page header */
.page-title { font-size: 22px; font-weight: 800; margin-bottom: 4px; }
.page-sub   { font-size: 13px; color: var(--text2); margin-bottom: 20px; }

/* Confidence bar */
.conf-track { height: 4px; background: var(--bg4); border-radius: 2px; margin-top: 6px; }
.conf-fill  { height: 100%; border-radius: 2px; }

/* Drop zone */
.drop-zone {
  border: 2px dashed var(--border2);
  border-radius: 10px; padding: 40px 20px;
  text-align: center; cursor: pointer; transition: all .2s;
}
.drop-zone:hover { border-color: var(--blue); background: rgba(33,136,255,.04); }

/* Empty state */
.empty-state {
  text-align: center; padding: 48px 20px;
  color: var(--text3); font-size: 13px;
}

/* Divider */
.divider { height: 1px; background: var(--border); margin: 14px 0; }

/* Mono helper */
.mono { font-family: var(--mono); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for k, v in {
    "history": [], "model": None, "page": "dashboard",
    "threshold": 0.40, "show_boxes": True,
    "autosave": True, "alert_on": True,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# DEFECT CONFIG
# ─────────────────────────────────────────────
DEFECT_MAP = {
    "flame":     {"label": "🔥 Flame (ไฟ)",       "hex": "#f85149", "rgb": (248,81,73)},
    "fire":      {"label": "🔥 Fire (ไฟ)",         "hex": "#f85149", "rgb": (248,81,73)},
    "smoke":     {"label": "💨 Smoke (ควัน)",       "hex": "#8b949e", "rgb": (139,148,158)},
    "rust":      {"label": "🟠 Rust (สนิม)",        "hex": "#e3b341", "rgb": (227,179,65)},
    "dent":      {"label": "🔵 Dent (รอยบุบ)",       "hex": "#2188ff", "rgb": (33,136,255)},
    "corrosion": {"label": "🟠 Corrosion",          "hex": "#e3b341", "rgb": (227,179,65)},
    "scratch":   {"label": "🔵 Scratch",            "hex": "#2188ff", "rgb": (33,136,255)},
}
def get_info(cls):
    return DEFECT_MAP.get(cls.lower(), {
        "label": f"⚠️ {cls.capitalize()}", "hex": "#f85149", "rgb": (248,81,73)
    })
def get_severity(c):
    return "🔴 High" if c >= .80 else ("🟡 Medium" if c >= .55 else "🟢 Low")

# ─────────────────────────────────────────────
# DRAW BOXES — PIL only
# ─────────────────────────────────────────────
def draw_boxes(img: Image.Image, dets: list) -> Image.Image:
    out  = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    for d in dets:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        r, g, b = d["rgb"]
        for t in range(3):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=(r,g,b))
        label = f"{d['cls'].capitalize()} {d['conf']:.0%}"
        lw = len(label)*8+12; lh = 22
        y_top = max(y1-lh, 0)
        draw.rectangle([x1, y_top, x1+lw, y1], fill=(r,g,b,210))
        draw.text((x1+6, y_top+4), label, fill=(255,255,255))
    return out

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_inference(img: Image.Image, thr: float):
    t0  = time.time()
    res = st.session_state.model(img, conf=thr, verbose=False)[0]
    el  = time.time()-t0
    dets = []
    for box in res.boxes:
        cid   = int(box.cls[0])
        cname = st.session_state.model.names[cid]
        conf  = float(box.conf[0])
        x1,y1,x2,y2 = [float(v) for v in box.xyxy[0]]
        info  = get_info(cname)
        dets.append({
            "cls":cname,"label":info["label"],"conf":conf,
            "severity":get_severity(conf),
            "hex":info["hex"],"rgb":info["rgb"],
            "x1":x1,"y1":y1,"x2":x2,"y2":y2,
        })
    dets.sort(key=lambda d:d["conf"], reverse=True)
    return dets, el

def add_history(src, status, dets, thumb=None):
    if not st.session_state.autosave: return
    st.session_state.history.insert(0, {
        "id": int(time.time()*1000),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":src,"status":status,"detections":dets,"thumb":thumb,
    })
    if len(st.session_state.history) > 200: st.session_state.history.pop()

def fault_count():
    return sum(1 for h in st.session_state.history if h["status"]=="FAULT")
def avg_conf():
    cs = [d["conf"] for h in st.session_state.history for d in h["detections"]]
    return f"{sum(cs)/len(cs):.0%}" if cs else "—"

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
MODEL_FILE = "Flame_Best_Model.pt"
@st.cache_resource
def load_model(path):
    from ultralytics import YOLO
    return YOLO(path)

if st.session_state.model is None and os.path.exists(MODEL_FILE):
    with st.spinner(f"⏳ Loading {MODEL_FILE}…"):
        st.session_state.model = load_model(MODEL_FILE)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo block
    model_ok = st.session_state.model is not None
    cls_str  = ', '.join(st.session_state.model.names.values()) if model_ok else "—"

    st.markdown(f"""
    <div class="sb-logo">
      <div class="sb-logo-icon">👁</div>
      <div>
        <div class="sb-logo-name">Littlelome<em>AI</em> Vision</div>
        <div class="sb-logo-sub">YOLOv11 · Flame Detection</div>
      </div>
    </div>
    <div class="model-status">
      {'<span style="color:#3fb950"><span class="sdot sdot-g"></span>Model loaded · '+cls_str+'</span>'
       if model_ok else
       '<span style="color:#f85149">⚠️ Model not found</span>'}
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        up = st.file_uploader("Upload Flame_Best_Model.pt", type=["pt"], label_visibility="collapsed")
        if up:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp.write(up.read())
            st.session_state.model = load_model(tmp.name)
            st.rerun()

    # Nav — MAIN
    st.markdown('<div class="nav-section">MAIN</div>', unsafe_allow_html=True)
    for pid, icon, label in [
        ("dashboard",  "▦", "Dashboard"),
        ("camera",     "▶", "Live Camera"),
        ("upload_img", "🖼", "Upload Image"),
        ("upload_vid", "🎬", "Upload Video"),
    ]:
        if st.session_state.page == pid:
            st.markdown(f'<div class="nav-active">{icon}&nbsp;&nbsp;{label}</div>', unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {label}", key=f"n_{pid}", use_container_width=True):
                st.session_state.page = pid; st.rerun()

    # Nav — DATA
    st.markdown('<div class="nav-section">DATA</div>', unsafe_allow_html=True)
    if st.session_state.page == "history":
        st.markdown('<div class="nav-active">⏱&nbsp;&nbsp;History</div>', unsafe_allow_html=True)
    else:
        if st.button("⏱  History", key="n_history", use_container_width=True):
            st.session_state.page = "history"; st.rerun()

    # Nav — SYSTEM
    st.markdown('<div class="nav-section">SYSTEM</div>', unsafe_allow_html=True)
    if st.session_state.page == "settings":
        st.markdown('<div class="nav-active">⚙&nbsp;&nbsp;Settings</div>', unsafe_allow_html=True)
    else:
        if st.button("⚙  Settings", key="n_settings", use_container_width=True):
            st.session_state.page = "settings"; st.rerun()

    # Bottom stats
    total  = len(st.session_state.history)
    faults = fault_count()
    st.markdown(f"""
    <div style="border-top:1px solid var(--border);margin-top:12px;padding-top:10px">
      <div class="sb-stats">
        <div class="sb-stat-box">
          <div class="sb-stat-val">{total}</div>
          <div class="sb-stat-label">Scanned</div>
        </div>
        <div class="sb-stat-box">
          <div class="sb-stat-val" style="color:var(--red)">{faults}</div>
          <div class="sb-stat-label">Faults</div>
        </div>
      </div>
      <div class="sb-footer">Littlelome AI Vision © 2025</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP HEADER BAR (sticky, spans content area)
# ─────────────────────────────────────────────
now_str = datetime.now().strftime("%H:%M:%S")
st.markdown(f"""
<div class="top-bar">
  <div class="logo-icon">👁</div>
  <div class="logo-text">Littlelome<em>AI</em> Vision</div>
  <div class="sep"></div>
  <span class="conn-dot"></span>
  <span class="conn-label">CONNECTED</span>
  <div class="hdr-spacer"></div>
  <span class="hdr-time" id="clock">{now_str}</span>
  <span style="font-size:12px;color:var(--text3);margin-left:12px">v1.0.0</span>
  <div class="avatar">OP</div>
</div>
""", unsafe_allow_html=True)

# Page titles / subtitles
PAGE_META = {
    "dashboard":  ("Dashboard",     "Real-time overview of inspection activity"),
    "camera":     ("Live Camera",   "Real-time defect detection via webcam"),
    "upload_img": ("Upload Image",  "Analyze a single image for defects"),
    "upload_vid": ("Upload Video",  "Process video frame-by-frame"),
    "history":    ("History",       "All past inspection records"),
    "settings":   ("Settings",      "Configure detection parameters"),
}
ptitle, psub = PAGE_META[st.session_state.page]
st.markdown(f'<div class="page-title">{ptitle}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-sub">{psub}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════
if st.session_state.page == "dashboard":
    total  = len(st.session_state.history)
    faults = fault_count()
    ok     = total - faults
    rate   = f"{faults/total:.0%}" if total else "0%"

    # ── KPI Grid — 4 columns, colored top border ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:var(--blue)">
          <div class="kpi-label">Total Inspected</div>
          <div class="kpi-val" style="color:var(--blue)">{total}</div>
          <div class="kpi-sub">All time</div>
          <div class="kpi-icon">🔍</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:var(--red)">
          <div class="kpi-label">Defects Found</div>
          <div class="kpi-val" style="color:var(--red)">{faults}</div>
          <div class="kpi-sub">Rate: {rate}</div>
          <div class="kpi-icon">⚠️</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:var(--green)">
          <div class="kpi-label">Passed (OK)</div>
          <div class="kpi-val" style="color:var(--green)">{ok}</div>
          <div class="kpi-sub">No defects detected</div>
          <div class="kpi-icon">✅</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:var(--orange)">
          <div class="kpi-label">Avg Confidence</div>
          <div class="kpi-val" style="color:var(--orange)">{avg_conf()}</div>
          <div class="kpi-sub">AI detection confidence</div>
          <div class="kpi-icon">📊</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)

    # ── Defect Trend ──
    with ca:
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <div class="card-title">📈 Defect Trend</div>
            <span class="badge badge-info">Live</span>
          </div>""", unsafe_allow_html=True)
        if len(st.session_state.history) >= 2:
            df = pd.DataFrame([{"Time": h["time"], "Fault": 1 if h["status"]=="FAULT" else 0}
                                for h in reversed(st.session_state.history)])
            df["Time"] = pd.to_datetime(df["Time"])
            st.line_chart(df.set_index("Time"), color="#f85149", height=150)
        else:
            st.markdown('<div class="empty-state">No data yet</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── System Status ──
    with cb:
        model_label = cls_str if model_ok else "Not loaded"
        model_color = "var(--green)" if model_ok else "var(--text3)"
        model_dot   = "sdot-g" if model_ok else "sdot-x"
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <div class="card-title">🖥 System Status</div>
          </div>
          <div class="sys-row">
            <span>AI Engine</span>
            <div style="display:flex;align-items:center">
              <span class="sdot sdot-g"></span>
              <span style="color:var(--green);font-family:var(--mono);font-size:12px">Online</span>
            </div>
          </div>
          <div class="sys-row">
            <span>YOLOv11 Model</span>
            <div style="display:flex;align-items:center">
              <span class="sdot {model_dot}"></span>
              <span style="color:{model_color};font-family:var(--mono);font-size:12px">{'Loaded' if model_ok else 'Not loaded'}</span>
            </div>
          </div>
          <div class="sys-row">
            <span>Classes</span>
            <span style="font-family:var(--mono);font-size:11px;color:var(--text2)">{cls_str}</span>
          </div>
          <div class="sys-row">
            <span>Alert System</span>
            <div style="display:flex;align-items:center">
              <span class="sdot sdot-o"></span>
              <span style="color:var(--orange);font-family:var(--mono);font-size:12px">Armed</span>
            </div>
          </div>
          <div class="sys-row">
            <span>Auto-Save</span>
            <span style="font-family:var(--mono);font-size:12px;color:var(--text2)">{'On' if st.session_state.autosave else 'Off'}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Recent Inspections ──
    rc1, rc2 = st.columns([5,1])
    with rc1:
        st.markdown('<div class="card-title" style="margin-bottom:12px">📋 Recent Inspections</div>', unsafe_allow_html=True)
    with rc2:
        if st.button("View All", key="view_all"):
            st.session_state.page = "history"; st.rerun()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.history:
        rows = [{"Status":  "⚠️ FAULT" if h["status"]=="FAULT" else "✅ OK",
                 "Defect":  ", ".join(d["label"] for d in h["detections"]) or "—",
                 "Conf":    f"{h['detections'][0]['conf']:.0%}" if h["detections"] else "—",
                 "Source":  h["source"],
                 "Time":    h["time"]}
                for h in st.session_state.history[:6]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="empty-state">No inspections yet. Start with Live Camera or Upload Image.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
# LIVE CAMERA
# ════════════════════════════════════════════
elif st.session_state.page == "camera":
    if not st.session_state.model:
        st.warning("⚠️ Load model first — upload .pt file in the sidebar"); st.stop()

    col_main, col_panel = st.columns([3,1])
    with col_panel:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:12px">⚙️ Detection Controls</div>', unsafe_allow_html=True)
        run_cam  = st.toggle("▶ Start Camera", key="cam_toggle")
        cam_src  = st.selectbox("Source", [0,1,2,"RTSP/HTTP URL"], key="cam_src", label_visibility="collapsed")
        if cam_src == "RTSP/HTTP URL":
            cam_src = st.text_input("URL", "rtsp://", key="cam_url")
        scan_s   = st.slider("Scan Interval (s)", 1, 10, 3, key="cam_scan")
        thr_cam  = st.slider("Confidence Threshold", 0.10, 0.95, st.session_state.threshold, 0.05,
                              format="%.0f%%", key="cam_thr")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:10px">🎯 Live Detections</div>', unsafe_allow_html=True)
        slot_dets  = st.empty()
        slot_dets.markdown('<div style="font-size:12px;color:var(--text3);text-align:center;padding:16px 0">Waiting for camera…</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:10px">📊 Session Stats</div>', unsafe_allow_html=True)
        slot_stats = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        slot_frame  = st.empty()
        slot_status = st.empty()

    if run_cam:
        try:
            import cv2 as _cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        if not has_cv2:
            slot_frame.markdown("""
            <div class="card" style="text-align:center;padding:60px 20px">
              <div style="font-size:40px;margin-bottom:10px">⚠️</div>
              <div style="font-weight:700;margin-bottom:6px">opencv-python-headless required</div>
              <div style="font-size:12px;color:var(--text2)">
                Add <code>opencv-python-headless</code> to requirements.txt<br>
                or use Upload Image / Upload Video instead
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            src_val = int(cam_src) if isinstance(cam_src, int) else cam_src
            cap = _cv2.VideoCapture(src_val)
            if not cap.isOpened():
                st.error("❌ Cannot open camera. Check index or URL.")
            else:
                slot_status.success("🟢 Camera running")
                last_t = 0; last_dets = []; ss = 0; sf = 0
                for _ in range(900):
                    if not st.session_state.get("cam_toggle", False): break
                    ret, frm = cap.read()
                    if not ret: break
                    now = time.time(); ann = frm.copy()

                    if now - last_t >= scan_s:
                        pil = Image.fromarray(_cv2.cvtColor(frm, _cv2.COLOR_BGR2RGB))
                        last_dets, el = run_inference(pil, thr_cam)
                        last_t = now; ss += 1
                        if last_dets:
                            sf += 1; add_history("Camera","FAULT",last_dets)
                            det_html = "".join(f"""
                            <div class="detect-item">
                              <div>
                                <div class="detect-type" style="color:{d['hex']}">{d['label']}</div>
                                <div style="font-size:11px;color:var(--text3)">{d['severity']}</div>
                              </div>
                              <div>
                                <div class="detect-conf">{d['conf']:.0%}</div>
                                <div class="conf-track"><div class="conf-fill" style="width:{d['conf']*100:.0f}%;background:{d['hex']}"></div></div>
                              </div>
                            </div>""" for d in last_dets)
                            slot_dets.markdown(det_html, unsafe_allow_html=True)
                        else:
                            add_history("Camera","OK",[])
                            slot_dets.markdown('<div style="font-size:12px;color:var(--green);text-align:center;padding:16px 0">✅ Clear</div>', unsafe_allow_html=True)

                        slot_stats.markdown(f"""
                        <div style="font-size:12px">
                          <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--bg4)">
                            <span style="color:var(--text2)">Scanned</span><span class="mono">{ss}</span>
                          </div>
                          <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--bg4)">
                            <span style="color:var(--text2)">Faults</span><span class="mono" style="color:var(--red)">{sf}</span>
                          </div>
                          <div style="display:flex;justify-content:space-between;padding:5px 0">
                            <span style="color:var(--text2)">Pass Rate</span><span class="mono" style="color:var(--green)">{(ss-sf)/max(ss,1):.0%}</span>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    for d in last_dets:
                        r,g,b = d["rgb"]
                        x1,y1,x2,y2 = int(d["x1"]),int(d["y1"]),int(d["x2"]),int(d["y2"])
                        _cv2.rectangle(ann,(x1,y1),(x2,y2),(b,g,r),3)
                        txt = f"{d['cls'].capitalize()} {d['conf']:.0%}"
                        _cv2.rectangle(ann,(x1,y1-26),(x1+len(txt)*10,y1),(b,g,r),-1)
                        _cv2.putText(ann,txt,(x1+4,y1-8),_cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,255),1)

                    slot_frame.image(_cv2.cvtColor(ann,_cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                cap.release(); slot_status.info("⏹ Camera stopped")
    else:
        slot_frame.markdown("""
        <div class="card" style="text-align:center;padding:80px 20px">
          <div style="font-size:52px;margin-bottom:12px;opacity:.3">📹</div>
          <div style="font-size:15px;font-weight:700;margin-bottom:6px">Camera Inactive</div>
          <div style="font-size:12px;color:var(--text2)">Toggle "Start Camera" to begin</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# UPLOAD IMAGE
# ════════════════════════════════════════════
elif st.session_state.page == "upload_img":
    if not st.session_state.model:
        st.warning("⚠️ Load model first (sidebar)"); st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:12px">📤 Upload</div>', unsafe_allow_html=True)
        uimg = st.file_uploader("Drop image here · JPG PNG WEBP",
                                 type=["jpg","jpeg","png","webp","bmp"], key="img_up")
        thr = st.slider("Confidence Threshold", 0.10, 0.95,
                         st.session_state.threshold, 0.05, format="%.0f%%", key="img_thr")
        abtn = st.button("🔍  Analyze Image", key="btn_img",
                          use_container_width=True, disabled=(uimg is None))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:12px">👁 Preview & Result</div>', unsafe_allow_html=True)
        if uimg:
            img   = Image.open(uimg).convert("RGB")
            rslot = st.empty()
            rslot.image(img, use_container_width=True)
        else:
            st.markdown('<div class="empty-state"><div style="font-size:36px;margin-bottom:8px">📷</div>No image loaded</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uimg and abtn:
        with st.spinner("Running YOLOv11…"):
            dets, elapsed = run_inference(img, thr)
        if dets:
            if st.session_state.show_boxes:
                rslot.image(draw_boxes(img, dets), use_container_width=True)
            add_history("Image","FAULT",dets,img.copy())
            st.markdown(f"""
            <div class="fault-box">
              <div style="font-size:20px;font-weight:800;color:var(--red)">⚠️ FAULT DETECTED</div>
              <div style="font-size:12px;color:var(--text2);margin-top:4px">{len(dets)} detection(s) · {elapsed*1000:.0f}ms</div>
            </div>""", unsafe_allow_html=True)
            for d in dets:
                st.markdown(f"""
                <div class="detect-item" style="flex-direction:column;align-items:flex-start;gap:4px">
                  <div style="display:flex;justify-content:space-between;width:100%">
                    <span class="detect-type" style="color:{d['hex']}">{d['label']}</span>
                    <span class="detect-conf">{d['conf']:.1%}</span>
                  </div>
                  <div style="font-size:11px;color:var(--text3)">{d['severity']}</div>
                  <div class="conf-track" style="width:100%">
                    <div class="conf-fill" style="width:{d['conf']*100:.0f}%;background:{d['hex']}"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            add_history("Image","OK",[],img.copy())
            st.markdown("""
            <div class="ok-box">
              <div style="font-size:20px;font-weight:800;color:var(--green)">✅ NO DEFECT DETECTED</div>
              <div style="font-size:12px;color:var(--text2);margin-top:4px">Image is clear</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# UPLOAD VIDEO
# ════════════════════════════════════════════
elif st.session_state.page == "upload_vid":
    if not st.session_state.model:
        st.warning("⚠️ Load model first (sidebar)"); st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:12px">📤 Upload Video</div>', unsafe_allow_html=True)
        uvid = st.file_uploader("Drop video here · MP4 MOV AVI",
                                 type=["mp4","mov","avi","mkv","webm"], key="vid_up")
        va, vb = st.columns(2)
        with va: fint = st.slider("Every (s)",  1,10,2, key="vid_int")
        with vb: mxf  = st.slider("Max frames", 5,30,10, key="vid_max")
        thr_v = st.slider("Threshold",0.10,0.95,st.session_state.threshold,0.05,format="%.0f%%",key="vid_thr")
        pbtn  = st.button("▶  Process Video", key="btn_vid",
                           use_container_width=True, disabled=(uvid is None))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:12px">🎞 Video Preview</div>', unsafe_allow_html=True)
        if uvid:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uvid.read()); vpath = tmp.name
            st.video(vpath)
        else:
            st.markdown('<div class="empty-state"><div style="font-size:36px;margin-bottom:8px">🎬</div>No video loaded</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uvid and pbtn:
        try:
            import cv2 as _cv2
            cap   = _cv2.VideoCapture(vpath)
            fps_v = cap.get(_cv2.CAP_PROP_FPS) or 25
            step  = max(1, int(fps_v*fint))
            prog  = st.progress(0, "Reading frames…")
            buf   = []; fi = 0
            while len(buf) < mxf:
                ret, frm = cap.read()
                if not ret: break
                if fi % step == 0:
                    buf.append((fi, fi/fps_v, Image.fromarray(_cv2.cvtColor(frm,_cv2.COLOR_BGR2RGB))))
                fi += 1
            cap.release()
        except ImportError:
            try:
                import imageio.v3 as iio
                prog = st.progress(0,"Reading frames…")
                fps_v = 25
                try:
                    meta = iio.immeta(vpath, plugin="pyav")
                    fps_v = float(meta.get("fps",25))
                except Exception: pass
                step = max(1,int(fps_v*fint)); buf=[]; fi=0
                for arr in iio.imiter(vpath, plugin="pyav"):
                    if len(buf)>=mxf: break
                    if fi%step==0: buf.append((fi,fi/fps_v,Image.fromarray(arr)))
                    fi+=1
            except ImportError:
                st.error("❌ Add `opencv-python-headless` or `imageio[pyav]` to requirements.txt"); st.stop()

        fres = []
        st.markdown('<div class="card-title" style="margin:16px 0 10px">🔍 Frame Results</div>', unsafe_allow_html=True)
        tcols = st.columns(min(4, max(len(buf),1)))
        for i,(fidx,tsec,pil) in enumerate(buf):
            prog.progress((i+1)/len(buf), f"Analyzing frame {i+1}/{len(buf)}…")
            dets,_ = run_inference(pil, thr_v)
            disp   = draw_boxes(pil.copy(),dets) if (st.session_state.show_boxes and dets) else pil
            with tcols[i%4]:
                st.image(disp, caption=f"{'🔥' if dets else '✅'} @{tsec:.1f}s", use_container_width=True)
            fres.append({"time":tsec,"status":"FAULT" if dets else "OK","dets":dets})
        prog.empty(); st.divider()

        ff = [r for r in fres if r["status"]=="FAULT"]
        ad = [d for r in fres for d in r["dets"]]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Frames",     len(fres))
        m2.metric("⚠️ Faults",  len(ff))
        m3.metric("✅ Clear",   len(fres)-len(ff))
        m4.metric("Clear Rate", f"{(len(fres)-len(ff))/max(len(fres),1):.0%}")
        if ff:
            st.error(f"🔥 Flame detected at: {', '.join(f'{r[chr(116)]:.1f}s' for r in ff)}")
        else:
            st.success("✅ No flame detected")
        add_history("Video","FAULT" if ff else "OK",ad[:5])

# ════════════════════════════════════════════
# HISTORY
# ════════════════════════════════════════════
elif st.session_state.page == "history":
    hc1, hc2, hc3 = st.columns([5,1,1])
    with hc2:
        if st.button("📥 Export", use_container_width=True):
            d = json.dumps([{k:v for k,v in h.items() if k!="thumb"}
                            for h in st.session_state.history], indent=2)
            st.download_button("Download JSON", d, "history.json", "application/json")
    with hc3:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history=[]; st.rerun()

    if not st.session_state.history:
        st.markdown("""
        <div class="card">
          <div class="empty-state">
            <div style="font-size:48px;margin-bottom:12px">📋</div>
            <div style="font-size:15px;font-weight:700;margin-bottom:6px">No history yet</div>
            <div>Results will appear here after inspections</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        total=len(st.session_state.history); faults=fault_count()
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total",total); k2.metric("Faults",faults)
        k3.metric("Passed",total-faults); k4.metric("Pass Rate",f"{(total-faults)/max(total,1):.0%}")
        st.markdown("<br>", unsafe_allow_html=True)

        for i, h in enumerate(st.session_state.history):
            isf  = h["status"]=="FAULT"
            icon = "⚠️ FAULT" if isf else "✅ OK"
            dstr = ", ".join(d["label"] for d in h["detections"]) or "No detections"
            with st.expander(f"{icon}  ·  {h['source']}  ·  {dstr}  ·  {h['time']}",
                             expanded=(i==0 and isf)):
                ec1, ec2 = st.columns([2,1])
                with ec1:
                    if h.get("thumb"):
                        try:
                            disp = draw_boxes(h["thumb"].copy(),h["detections"]) \
                                   if (st.session_state.show_boxes and h["detections"]) \
                                   else h["thumb"]
                            st.image(disp, use_container_width=True)
                        except Exception: pass
                with ec2:
                    c = "var(--red)" if isf else "var(--green)"
                    st.markdown(f"""
                    <div style="font-size:13px;line-height:2.2">
                      <b>Status:</b> <span style="color:{c};font-weight:700">{icon}</span><br>
                      <b>Source:</b> <span class="mono">{h['source']}</span><br>
                      <b>Time:</b>   <span class="mono" style="font-size:11px">{h['time']}</span>
                    </div>""", unsafe_allow_html=True)
                    for d in h["detections"]:
                        st.markdown(f"""
                        <div class="detect-item" style="flex-direction:column;gap:4px;align-items:flex-start">
                          <div style="display:flex;justify-content:space-between;width:100%">
                            <span style="color:{d['hex']};font-weight:600">{d['label']}</span>
                            <span class="mono">{d['conf']:.0%}</span>
                          </div>
                          <div style="font-size:11px;color:var(--text3)">{d['severity']}</div>
                        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════
elif st.session_state.page == "settings":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:14px">🤖 AI Configuration</div>', unsafe_allow_html=True)
        nthr = st.slider("Confidence Threshold",0.10,0.95,
                          st.session_state.threshold,0.05,format="%.0f%%",key="set_thr",
                          help="ต่ำ = sensitive | สูง = precise")
        if nthr != st.session_state.threshold:
            st.session_state.threshold = nthr
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**Active Detection Classes:**")
        if model_ok:
            for cls in st.session_state.model.names.values():
                info = get_info(cls)
                st.markdown(f'<div class="detect-item"><span class="detect-type" style="color:{info["hex"]}">{info["label"]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:14px">🔔 Alerts & Display</div>', unsafe_allow_html=True)
        st.session_state.show_boxes = st.toggle("Show Bounding Boxes",   value=st.session_state.show_boxes, key="set_bbox")
        st.session_state.autosave   = st.toggle("Auto-save to History",  value=st.session_state.autosave,   key="set_save")
        st.session_state.alert_on   = st.toggle("Fault Alert Highlight", value=st.session_state.alert_on,   key="set_alert")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**ℹ️ System Info**")
        if model_ok:
            st.markdown(f"""
            <div style="font-size:13px;line-height:2.2">
              <b>Model:</b>  <span class="mono">Flame_Best_Model.pt</span><br>
              <b>Architecture:</b> <span class="mono">YOLOv11n</span><br>
              <b>Classes:</b> <span class="mono">{cls_str}</span><br>
              <b>Inspected:</b> <span class="mono">{len(st.session_state.history)}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history=[]; st.success("✅ History cleared"); st.rerun()
        if st.session_state.history:
            d = json.dumps([{k:v for k,v in h.items() if k!="thumb"}
                            for h in st.session_state.history], indent=2)
            st.download_button("📥 Export History (JSON)", d, "history.json",
                                "application/json", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
