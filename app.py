import html
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_PATH, MODEL_PATH, METRICS_PATH

# ---------------------------------------------------------------------------
# Theme bootstrap: Streamlit reads theme colours only from .streamlit/config.toml.
# If the project has none, create a light-theme one (never overwrites an existing file).
# ---------------------------------------------------------------------------
_THEME_CREATED = False
try:
    _cfg = Path.cwd() / ".streamlit" / "config.toml"
    if not _cfg.exists():
        _cfg.parent.mkdir(parents=True, exist_ok=True)
        _cfg.write_text(
            "[theme]\n"
            'base = "light"\n'
            'primaryColor = "#0D9488"\n'
            'backgroundColor = "#F5F7FA"\n'
            'secondaryBackgroundColor = "#FFFFFF"\n'
            'textColor = "#334155"\n'
            'font = "sans serif"\n',
            encoding="utf-8",
        )
        _THEME_CREATED = True
except Exception:
    pass

st.set_page_config(page_title="Employee Performance | People Analytics", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Design system
#   Navy #14243A · Teal #0D9488 · Background #F5F7FA · Card #FFFFFF
#   Text #334155 · Muted #64748B · Accent blue #5B8DEF
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {
  --navy:#14243A; --teal:#0D9488; --teal-dark:#0B7A70; --teal-tint:#E6F5F3;
  --bg:#F5F7FA; --card:#FFFFFF; --text:#334155; --muted:#64748B;
  --blue:#5B8DEF; --blue-tint:#EAF0FD; --border:#E3E8EF;
  --coral:#E07A6B; --amber:#E5A93D;
}

html, body, [class*="css"] {font-family:'Inter',sans-serif;}
.stApp {background:var(--bg); color:var(--text);}
[data-testid="stHeader"] {background:rgba(245,247,250,.92); backdrop-filter:blur(6px);}
.block-container {max-width:1500px; padding-top:1.6rem; padding-bottom:2rem;}
footer {visibility:hidden;}

/* Typography */
h1,h2,h3,h4 {font-family:'Plus Jakarta Sans',sans-serif !important; color:var(--navy) !important; letter-spacing:-.02em;}
p, label, span, li {color:inherit;}

/* Sidebar */
[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid var(--border);}
[data-testid="stSidebar"] .block-container {padding-top:1rem;}
.sb-brand {background:var(--navy); border-radius:14px; padding:16px 18px; margin-bottom:1.1rem;}
.sb-brand .sb-name {color:#fff; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.02rem; letter-spacing:-.01em;}
.sb-brand .sb-sub {color:#9FB3C8; font-size:.76rem; margin-top:2px;}
.sb-section {color:var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; margin:.4rem 0 .5rem 0;}
.sb-stat {background:var(--bg); border:1px solid var(--border); border-radius:12px; padding:12px 14px; margin-bottom:.6rem;}
.sb-stat .k {color:var(--muted); font-size:.74rem; font-weight:600;}
.sb-stat .v {color:var(--navy); font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.25rem; line-height:1.3;}
.sb-bar {height:6px; background:#DDE4EC; border-radius:99px; overflow:hidden; margin-top:6px;}
.sb-bar > div {height:100%; background:var(--teal); border-radius:99px;}
.sb-chip {display:inline-block; background:var(--blue-tint); color:#3C65C4; font-size:.72rem; font-weight:600; padding:4px 10px; border-radius:99px;}

/* Header */
.app-header {background:var(--card); border:1px solid var(--border); border-radius:16px; padding:22px 28px; display:flex; justify-content:space-between; align-items:center; gap:24px; box-shadow:0 1px 3px rgba(20,36,58,.05); border-left:5px solid var(--teal);}
.app-header .brand {color:var(--teal); font-size:.74rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:6px;}
.app-header h1 {margin:0 !important; padding:0 !important; font-size:1.9rem !important; font-weight:800 !important; line-height:1.2;}
.app-header .sub {color:var(--muted); font-size:.92rem; margin-top:6px; max-width:760px;}
.status-pill {display:flex; align-items:center; gap:10px; background:var(--teal-tint); border:1px solid #BFE5E0; color:#0B6B63; padding:10px 16px; border-radius:99px; font-size:.82rem; font-weight:600; white-space:nowrap;}
.status-dot {width:9px; height:9px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 4px rgba(13,148,136,.18);}

/* Notice */
.notice {display:flex; gap:12px; align-items:flex-start; background:#FFF8EA; border:1px solid #F3DFB5; border-radius:12px; padding:12px 16px; margin:14px 0 18px 0; color:#6B4E12; font-size:.84rem; line-height:1.5;}
.notice b {color:#5A3F08;}
.notice svg {flex:0 0 auto; margin-top:2px;}

/* KPI cards */
.kpi {background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; height:118px; display:flex; align-items:center; gap:16px; box-shadow:0 1px 3px rgba(20,36,58,.05);}
.kpi .ico {flex:0 0 auto; width:46px; height:46px; border-radius:12px; display:flex; align-items:center; justify-content:center;}
.kpi .lbl {color:var(--muted); font-size:.78rem; font-weight:600;}
.kpi .val {color:var(--navy); font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.85rem; line-height:1.2; letter-spacing:-.02em;}
.kpi .hint {color:var(--muted); font-size:.72rem; margin-top:1px;}
.kpi.compact {height:104px;}
.kpi.compact .val {font-size:1.6rem;}

/* Share cards */
.share {background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; height:118px; box-shadow:0 1px 3px rgba(20,36,58,.05);}
.share .top {display:flex; justify-content:space-between; align-items:center;}
.share .name {display:flex; align-items:center; gap:8px; color:var(--text); font-weight:600; font-size:.86rem;}
.share .dot {width:10px; height:10px; border-radius:50%;}
.share .pct {color:var(--navy); font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.85rem; letter-spacing:-.02em; margin-top:6px;}
.share .track {height:7px; background:#EDF1F6; border-radius:99px; overflow:hidden; margin-top:6px;}
.share .track > div {height:100%; border-radius:99px;}

/* Section headings */
.sec {margin:1.4rem 0 .8rem 0;}
.sec .t {font-family:'Plus Jakarta Sans',sans-serif; color:var(--navy); font-weight:700; font-size:1.2rem; letter-spacing:-.015em;}
.sec .d {color:var(--muted); font-size:.86rem; margin-top:2px;}
.sec.first {margin-top:.4rem;}
.panel-note {color:var(--muted); font-size:.8rem; line-height:1.5; margin-top:.4rem;}

/* Containers used as chart / table cards */
[data-testid="stVerticalBlockBorderWrapper"] {background:var(--card); border-color:var(--border) !important; border-radius:14px; box-shadow:0 1px 3px rgba(20,36,58,.05);}

/* Tabs */
div[data-testid="stTabs"] [role="tablist"] {gap:6px; border-bottom:1px solid var(--border);}
div[data-testid="stTabs"] button {font-weight:600; font-size:.92rem; padding:12px 18px; color:var(--muted);}
div[data-testid="stTabs"] button:hover {color:var(--navy);}
div[data-testid="stTabs"] button[aria-selected="true"] {color:var(--teal-dark) !important;}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background-color:var(--teal) !important; height:3px;}

/* Inputs */
div[data-baseweb="select"] > div {background:#fff; border-color:var(--border); border-radius:10px; min-height:42px;}
div[data-baseweb="select"] > div:hover {border-color:var(--teal);}
.stSelectbox label, .stSlider label {color:var(--navy) !important; font-weight:600 !important; font-size:.84rem !important;}
.stSlider [data-testid="stThumbValue"] {color:var(--teal-dark); font-weight:700;}
.stSlider [role="slider"] {background-color:var(--teal) !important; box-shadow:0 0 0 4px rgba(13,148,136,.15);}
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {color:var(--muted);}

/* Form (prediction console) */
[data-testid="stForm"] {background:var(--card); border:1px solid var(--border); border-radius:16px; padding:22px 24px; box-shadow:0 1px 3px rgba(20,36,58,.05);}
.grp {display:flex; align-items:center; gap:10px; margin-bottom:.7rem; padding-bottom:.6rem; border-bottom:1px solid var(--border);}
.grp .gi {width:30px; height:30px; border-radius:8px; display:flex; align-items:center; justify-content:center;}
.grp .gt {color:var(--navy); font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:.98rem;}
.grp .gd {color:var(--muted); font-size:.74rem;}

/* Buttons */
div.stButton>button, div.stFormSubmitButton>button, div.stDownloadButton>button {background:var(--teal); color:#fff; border:1px solid var(--teal); border-radius:10px; font-weight:600; padding:.6rem 1.2rem; transition:background .15s ease;}
div.stButton>button:hover, div.stFormSubmitButton>button:hover, div.stDownloadButton>button:hover {background:var(--teal-dark); border-color:var(--teal-dark); color:#fff;}
div.stFormSubmitButton>button {height:48px; font-size:.98rem;}
div.stButton>button:focus-visible, div.stFormSubmitButton>button:focus-visible, div.stDownloadButton>button:focus-visible {outline:2px solid var(--blue); outline-offset:2px;}

/* Prediction result */
.result {border-radius:16px; padding:24px 26px; border:1px solid var(--border); background:var(--card); box-shadow:0 1px 3px rgba(20,36,58,.05); height:100%;}
.result .rl {color:var(--muted); font-size:.76rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;}
.result .rv {font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:2.6rem; letter-spacing:-.03em; line-height:1.15; margin-top:8px;}
.result .badge {display:inline-block; margin-top:10px; padding:5px 12px; border-radius:99px; font-size:.78rem; font-weight:600;}
.result .rs {color:var(--muted); font-size:.8rem; margin-top:14px; line-height:1.5;}

/* Empty state */
.empty {background:var(--blue-tint); border:1px dashed #B5C9F3; border-radius:14px; padding:22px; color:#33508F; font-size:.9rem; display:flex; gap:12px; align-items:center;}

/* Alerts and tables */
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:12px; overflow:hidden;}
[data-testid="stExpander"] {background:var(--card); border:1px solid var(--border); border-radius:12px;}
[data-testid="stExpander"] summary {font-weight:600; color:var(--navy);}
[data-testid="stCaptionContainer"] {color:var(--muted);}

/* ===== Force a consistent light theme, even when Streamlit runs a dark base theme ===== */
:root, html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"] {color-scheme:light !important;}
.stApp, [data-testid="stAppViewContainer"] {background:#F5F7FA !important; color:#334155 !important;}
[data-testid="stHeader"] {background:rgba(245,247,250,.92) !important;}
[data-testid="stHeader"] *, [data-testid="stToolbar"] * {color:#334155 !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {background:#FFFFFF !important;}
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li {color:#334155 !important;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {color:#14243A !important;}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, .stApp label p {color:#14243A !important; font-weight:600 !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stImageCaption"] {color:#64748B !important;}
.stApp hr {border-color:#E3E8EF !important;}

/* Tabs: readable labels, teal indicator */
div[data-testid="stTabs"] button[role="tab"] p {color:#64748B !important; font-weight:600 !important;}
div[data-testid="stTabs"] button[role="tab"]:hover p {color:#14243A !important;}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {color:#0B7A70 !important;}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {height:3px !important;}
div[data-testid="stTabs"] [data-baseweb="tab-border"] {background-color:#E3E8EF !important;}

/* Selectboxes and dropdown menus */
div[data-baseweb="select"] > div {background:#FFFFFF !important; border:1px solid #E3E8EF !important; border-radius:10px !important;}
div[data-baseweb="select"] div, div[data-baseweb="select"] span, div[data-baseweb="select"] input {color:#14243A !important;}
div[data-baseweb="select"] svg {fill:#64748B !important;}
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, ul[data-baseweb="menu"], ul[role="listbox"] {background:#FFFFFF !important;}
ul[role="listbox"] li, ul[data-baseweb="menu"] li {background:#FFFFFF !important; color:#14243A !important;}
ul[role="listbox"] li:hover, ul[data-baseweb="menu"] li:hover, ul[role="listbox"] li[aria-selected="true"] {background:#E6F5F3 !important;}

/* Sliders */
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {color:#64748B !important;}

/* Button labels always white on teal */
div.stFormSubmitButton button *, div.stDownloadButton button *, div.stButton button * {color:#FFFFFF !important;}

/* Cards, form, expander: white surface, light border */
[data-testid="stVerticalBlockBorderWrapper"] {background:#FFFFFF !important; border:1px solid #E3E8EF !important; border-radius:14px !important;}
[data-testid="stForm"] {background:#FFFFFF !important; border:1px solid #E3E8EF !important;}
[data-testid="stExpander"], [data-testid="stExpander"] details {background:#FFFFFF !important; border:1px solid #E3E8EF !important; border-radius:12px !important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {color:#14243A !important; font-weight:600;}
[data-testid="stExpander"] summary svg {fill:#64748B !important; color:#64748B !important;}

/* Custom callouts (replace dark st.info / st.warning) */
.callout {display:flex; gap:12px; align-items:flex-start; border-radius:12px; padding:13px 16px; margin:12px 0; font-size:.88rem; line-height:1.5; border:1px solid;}
.callout svg {flex:0 0 auto; margin-top:2px;}
.callout.info {background:#EAF0FD; border-color:#C3D5F7; color:#1E3A6E;}
.callout.warning {background:#FFF8EA; border-color:#F3DFB5; color:#6B4E12;}
.callout.error {background:#FDECEA; border-color:#F5C2BC; color:#7A1F14;}
.code {font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82em; background:#EEF2F7; color:#14243A; padding:1px 6px; border-radius:5px;}

/* HTML data tables (light, consistent with cards) */
.tbl-wrap {overflow:auto; border:1px solid #E3E8EF; border-radius:12px; background:#FFFFFF;}
.tbl {border-collapse:separate; border-spacing:0; width:100%; font-size:.86rem;}
.tbl th {position:sticky; top:0; z-index:1; background:#F1F5F9; color:#14243A; font-weight:700; text-align:left; padding:11px 14px; border-bottom:1px solid #E3E8EF; white-space:nowrap;}
.tbl td {padding:10px 14px; color:#334155; border-bottom:1px solid #EEF2F7; white-space:nowrap;}
.tbl tr:last-child td {border-bottom:none;}
.tbl tbody tr:nth-child(even) td {background:#FAFBFD;}
.tbl tbody tr:hover td {background:#F0F7F6;}
.tbl .num {text-align:right; font-variant-numeric:tabular-nums;}
.tbl th.num {text-align:right;}
.tbl td.best {background:#CCEBE7 !important; color:#0B4F49; font-weight:700;}
.tbl td.na {color:#94A3B8;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UI helpers (presentation only)
# ---------------------------------------------------------------------------
def _icon(name, color="currentColor", size=22):
    paths = {
        "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"/><circle cx="10" cy="7" r="4"/><path d="M21 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
        "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
        "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/><path d="m9 16 2 2 4-4"/>',
        "award": '<circle cx="12" cy="8" r="6"/><path d="M15.48 12.89 17 22l-5-3-5 3 1.52-9.11"/>',
        "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
        "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
        "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
        "rows": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18"/>',
        "columns": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18"/>',
        "alert": '<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>',
        "warn": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/>',
        "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
        "briefcase": '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
        "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
        "play": '<circle cx="12" cy="12" r="10"/><path d="m10 8 6 4-6 4z"/>',
    }
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths[name]}</svg>')


def kpi_card(icon, label, value, hint="", tone="teal", compact=False):
    tones = {
        "teal": ("#E6F5F3", "#0D9488"),
        "blue": ("#EAF0FD", "#5B8DEF"),
        "navy": ("#E8ECF2", "#14243A"),
        "amber": ("#FDF3E0", "#C98A12"),
    }
    bg, fg = tones[tone]
    cls = "kpi compact" if compact else "kpi"
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    return (f'<div class="{cls}"><div class="ico" style="background:{bg}">{_icon(icon, fg)}</div>'
            f'<div><div class="lbl">{label}</div><div class="val">{value}</div>{hint_html}</div></div>')


def share_card(label, share, color):
    pct = max(0.0, min(float(share), 1.0)) * 100
    return (f'<div class="share"><div class="top"><div class="name"><span class="dot" style="background:{color}"></span>{label}</div></div>'
            f'<div class="pct">{share:.1%}</div>'
            f'<div class="track"><div style="width:{pct:.1f}%;background:{color}"></div></div></div>')


def section(title, desc="", first=False):
    cls = "sec first" if first else "sec"
    d = f'<div class="d">{desc}</div>' if desc else ""
    st.markdown(f'<div class="{cls}"><div class="t">{title}</div>{d}</div>', unsafe_allow_html=True)


def group_header(icon, title, desc, tone="teal"):
    bg, fg = {"teal": ("#E6F5F3", "#0D9488"), "blue": ("#EAF0FD", "#5B8DEF"), "navy": ("#E8ECF2", "#14243A")}[tone]
    st.markdown(f'<div class="grp"><div class="gi" style="background:{bg}">{_icon(icon, fg, 17)}</div>'
                f'<div><div class="gt">{title}</div><div class="gd">{desc}</div></div></div>', unsafe_allow_html=True)


def callout(text, kind="info"):
    icon = {"info": "alert", "warning": "warn", "error": "alert"}[kind]
    col = {"info": "#1E3A6E", "warning": "#B7791F", "error": "#B3261E"}[kind]
    st.markdown(f'<div class="callout {kind}">{_icon(icon, col, 18)}<div>{text}</div></div>', unsafe_allow_html=True)


def render_table(frame, formats=None, best_cols=None, max_height=None):
    """Display-only HTML table. Values come straight from the DataFrame; formats only change how they are printed."""
    formats = formats or {}
    best_cols = best_cols or []
    best = {}
    for c in best_cols:
        vals = pd.to_numeric(frame[c], errors="coerce")
        if vals.notna().any():
            best[c] = vals.max()
    num_cols = {c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])}
    head = "".join(f'<th class="{"num" if c in num_cols else ""}">{html.escape(str(c))}</th>' for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = row[c]
            if pd.isna(v):
                cells.append('<td class="na num">–</td>' if c in num_cols else '<td class="na">–</td>')
                continue
            if c in formats:
                txt = formats[c].format(v)
            elif isinstance(v, float):
                txt = f"{v:.3f}".rstrip("0").rstrip(".")
            else:
                txt = str(v)
            cls = ["num"] if c in num_cols else []
            if c in best and pd.to_numeric(v, errors="coerce") == best[c]:
                cls.append("best")
            cells.append(f'<td class="{" ".join(cls)}">{html.escape(txt)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    style = f' style="max-height:{max_height}px"' if max_height else ""
    st.markdown(f'<div class="tbl-wrap"{style}><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)


ROOT = Path(__file__).resolve().parent
required = [DATA_PATH, MODEL_PATH, METRICS_PATH]
missing = [str(p) for p in required if not Path(p).exists()]
if missing:
    callout("Required project artifacts are missing: " + html.escape(", ".join(missing)), "error")
    callout("Run <span class=\"code\">python main.py</span> from the project root, then restart Streamlit.", "info")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    callout("Could not load project files: " + html.escape(str(exc)), "error")
    st.stop()

needed = {"department", "role", "training_hours", "attendance_rate", "skill_assessment", "performance_rating"}
if not needed.issubset(df.columns):
    callout("Dataset schema does not contain all expected dashboard columns.", "error")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-name">People Analytics</div>'
        '<div class="sb-sub">Workforce intelligence · ML demo</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-section">Filters</div>', unsafe_allow_html=True)
    departments = ["All departments"] + sorted(df.department.dropna().unique().tolist())
    selected_department = st.selectbox("Department", departments)
    filtered = df if selected_department == "All departments" else df[df.department == selected_department]
    st.caption("The filter updates the overview charts and the dataset preview.")
    st.divider()
    st.markdown('<div class="sb-section">Dataset scope</div>', unsafe_allow_html=True)
    scope_pct = (len(filtered) / len(df) * 100) if len(df) else 0
    st.markdown(
        f'<div class="sb-stat"><div class="k">Records in view</div>'
        f'<div class="v">{len(filtered):,} <span style="color:#64748B;font-weight:600;font-size:.82rem">of {len(df):,}</span></div>'
        f'<div class="sb-bar"><div style="width:{scope_pct:.1f}%"></div></div></div>'
        f'<div class="sb-stat"><div class="k">Active view</div>'
        f'<div class="v" style="font-size:1rem">{selected_department}</div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<span class="sb-chip">Academic demo · Synthetic data only</span>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><div>'
    '<div class="brand">People Analytics</div>'
    '<h1>Employee Performance Predictor</h1>'
    '<div class="sub">Explore workforce patterns, review model diagnostics, and test hypothetical prediction scenarios in one workspace.</div>'
    '</div>'
    '<div class="status-pill"><span class="status-dot"></span>Project demo · Ready</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="notice">{_icon("warn", "#B7791F", 18)}<div><b>Responsible-use notice.</b> '
    'This project uses synthetic data for educational demonstration. Predictions are not validated for real employees '
    'and must not guide hiring, promotion, compensation, or termination decisions.</div></div>',
    unsafe_allow_html=True,
)

if _THEME_CREATED:
    callout("Light theme config created at <span class=\"code\">.streamlit/config.toml</span>. Stop and restart Streamlit once to apply it to dropdowns, sliders and tabs.", "info")

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
counts = filtered.performance_rating.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi_card("users", "Total employee records", f"{len(filtered):,}", "Synthetic records in view", "navy"), unsafe_allow_html=True)
k2.markdown(kpi_card("book", "Average training hours", f"{filtered.training_hours.mean():.1f}", "Per employee", "teal"), unsafe_allow_html=True)
k3.markdown(kpi_card("calendar", "Average attendance", f"{filtered.attendance_rate.mean():.1%}", "Across selected records", "blue"), unsafe_allow_html=True)
k4.markdown(kpi_card("award", "Average skill assessment", f"{filtered.skill_assessment.mean():.1f}", "Assessment score", "amber"), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Shared chart styling
# ---------------------------------------------------------------------------
COLORS = {"Low": "#E07A6B", "Medium": "#E5A93D", "High": "#0D9488"}
LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    title=dict(font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=15, color="#14243A"), x=0.01, xanchor="left"),
    margin=dict(l=18, r=18, t=58, b=18),
    xaxis=dict(tickfont=dict(color="#475569", size=12), title=dict(font=dict(color="#334155", size=12)), linecolor="#CBD5E1"),
    yaxis=dict(tickfont=dict(color="#475569", size=12), title=dict(font=dict(color="#334155", size=12))),
    hoverlabel=dict(bgcolor="#14243A", font=dict(color="#FFFFFF", family="Inter, sans-serif", size=12), bordercolor="#14243A"),
)
CONFIG = {"displayModeBar": False, "responsive": True}
GRID = "#EDF1F6"

st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
tab_overview, tab_model, tab_prediction, tab_data = st.tabs(["Overview", "Model insights", "Prediction lab", "Dataset explorer"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    section("Workforce overview", "A high-level view of the selected synthetic dataset.", first=True)
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=COLORS,
                         labels={"x": "Synthetic performance category", "y": "Number of records"}, title="Performance category distribution", text=counts.values)
            fig.update_layout(**LAYOUT, height=360, showlegend=False, bargap=0.38)
            fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0,
                              hovertemplate="%{x}: %{y:,} records<extra></extra>")
            fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
            fig.update_yaxes(gridcolor=GRID, zeroline=False)
            st.plotly_chart(fig, use_container_width=True, config=CONFIG, theme=None)
    with b:
        with st.container(border=True):
            if selected_department == "All departments":
                summary = filtered.groupby("department", as_index=False).training_hours.mean().sort_values("training_hours", ascending=False)
                fig2 = px.bar(summary, x="department", y="training_hours", color="training_hours", color_continuous_scale=["#BFE5E0", "#0D9488"], title="Average training hours by department", labels={"training_hours": "Average training hours", "department": "Department"})
                fig2.update_traces(marker_line_width=0, hovertemplate="%{x}: %{y:.1f} hours<extra></extra>")
            else:
                fig2 = px.histogram(filtered, x="training_hours", nbins=18, title=f"Training-hours distribution · {selected_department}", labels={"training_hours": "Training hours"}, color_discrete_sequence=["#0D9488"])
                fig2.update_traces(marker_line_color="#FFFFFF", marker_line_width=1)
            fig2.update_layout(**LAYOUT, height=360, coloraxis_showscale=False, bargap=0.3)
            fig2.update_xaxes(showgrid=False, linecolor="#CBD5E1")
            fig2.update_yaxes(gridcolor=GRID, zeroline=False)
            st.plotly_chart(fig2, use_container_width=True, config=CONFIG, theme=None)

    section("Workforce snapshot", "Share of records in each synthetic performance category.")
    total = max(len(filtered), 1)
    x1, x2, x3 = st.columns(3)
    x1.markdown(share_card("Low category share", counts['Low'] / total, COLORS["Low"]), unsafe_allow_html=True)
    x2.markdown(share_card("Medium category share", counts['Medium'] / total, COLORS["Medium"]), unsafe_allow_html=True)
    x3.markdown(share_card("High category share", counts['High'] / total, COLORS["High"]), unsafe_allow_html=True)
    st.caption("Categories are generated by the synthetic label-generation process; they are not observed employee assessments.")

# ---------------------------------------------------------------------------
# Model insights
# ---------------------------------------------------------------------------
with tab_model:
    chosen = metrics.get("selected_model", "Not specified")
    chosen_result = metrics.get("models", {}).get(chosen, {})

    section("Model evaluation", "Held-out test-set metrics. Macro averages give each class equal weight.", first=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown(kpi_card("cpu", "Selected model", f"{chosen}", "Best on selection metric", "navy", compact=True), unsafe_allow_html=True)
    m2.markdown(kpi_card("target", "Test accuracy", f"{chosen_result.get('accuracy', 0):.1%}", "Held-out test split", "teal", compact=True), unsafe_allow_html=True)
    m3.markdown(kpi_card("layers", "Test macro F1", f"{chosen_result.get('macro_f1', 0):.3f}", "Equal weight per class", "blue", compact=True), unsafe_allow_html=True)
    st.markdown(f'<div class="panel-note">Selection criterion: <span class="code">{html.escape(str(metrics.get("selection_metric","not specified")))}</span> · Test rows: {html.escape(str(metrics.get("test_rows","not specified")))}</div>', unsafe_allow_html=True)

    section("Model comparison", "Best value in each metric is shaded. Values are shown exactly as evaluated.")
    rows = [{"Model": name, "Accuracy": r.get("accuracy"), "Macro precision": r.get("macro_precision"), "Macro recall": r.get("macro_recall"), "Macro F1": r.get("macro_f1")} for name, r in metrics.get("models", {}).items()]
    if rows:
        comp = pd.DataFrame(rows)
        metric_cols = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
        render_table(comp, formats={c: "{:.3f}" for c in metric_cols}, best_cols=metric_cols)

    section("Model diagnostics", "Confusion matrix and the inputs that most influence model performance.")
    dleft, dright = st.columns([1, 1.35], gap="large")
    cm_candidates = [ROOT / "images" / "confusion_matrix.png", ROOT / "outputs" / "confusion_matrix.png"]
    cm = next((p for p in cm_candidates if p.exists()), None)
    with dleft:
        with st.container(border=True):
            if cm:
                st.image(str(cm), caption="Confusion matrix from model evaluation", use_container_width=True)
            else:
                callout("Confusion matrix image not found in images/ or outputs/.", "info")
    with dright:
        with st.container(border=True):
            importance_path = ROOT / "outputs" / "permutation_importance.csv"
            if importance_path.exists():
                imp = pd.read_csv(importance_path)
                if {"feature", "importance_mean"}.issubset(imp.columns):
                    imp = imp.sort_values("importance_mean", ascending=True).tail(10)
                    fig_imp = px.bar(imp, x="importance_mean", y="feature", orientation="h",
                                     error_x="importance_std" if "importance_std" in imp.columns else None,
                                     color="importance_mean", color_continuous_scale=["#C5E3DF", "#0D9488"],
                                     title="Permutation importance · top inputs",
                                     labels={"importance_mean": "Mean macro-F1 decrease", "feature": "Input feature"})
                    fig_imp.update_layout(**LAYOUT, height=390, coloraxis_showscale=False)
                    fig_imp.update_traces(marker_line_width=0, error_x=dict(color="#64748B", thickness=1.2))
                    fig_imp.update_xaxes(gridcolor=GRID, zeroline=False)
                    fig_imp.update_yaxes(showgrid=False)
                    st.plotly_chart(fig_imp, use_container_width=True, config=CONFIG, theme=None)
                else:
                    callout("Permutation importance CSV is missing expected columns.", "warning")
            else:
                callout("Permutation importance file not found. Run <span class=\"code\">python main.py</span> to regenerate it.", "info")
    st.caption("Permutation importance measures how model performance changes when an input is shuffled. It is not causal evidence or an individual-level explanation.")

    section("Class-level performance", "Precision, recall and F1 for each class of the selected model.")
    report = []
    for label, values in chosen_result.get("classification_report", {}).items():
        if isinstance(values, dict) and "precision" in values:
            report.append({"Class": label, "Precision": values.get("precision"), "Recall": values.get("recall"), "F1-score": values.get("f1-score"), "Support": int(values.get("support", 0))})
    if report:
        rdf = pd.DataFrame(report)
        render_table(rdf, formats={"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Support": "{:,.0f}"})
    callout("Review recall separately for Low, Medium, and High. These results describe only this synthetic test split.", "info")

# ---------------------------------------------------------------------------
# Prediction lab
# ---------------------------------------------------------------------------
with tab_prediction:
    section("Interactive prediction lab", "Change hypothetical inputs and generate a model output. Do not enter real employee personal information.", first=True)
    with st.form("prediction_form", border=True):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            group_header("briefcase", "Role profile", "Where the employee sits", "navy")
            pred_department = st.selectbox("Department", sorted(df.department.dropna().unique()))
            roles = sorted(df.loc[df.department == pred_department, "role"].dropna().unique())
            pred_role = st.selectbox("Role", roles)
            tenure = st.slider("Tenure (years)", 0.2, 15.0, 3.0, 0.1)
            projects = st.slider("Prior projects completed", 0, 18, 6)
        with c2:
            group_header("activity", "Work and development", "Workload, presence and skills", "teal")
            workload = st.slider("Workload (hours/week)", 25, 65, 42)
            attendance = st.slider("Attendance rate", 0.70, 1.00, 0.95, 0.01)
            training = st.slider("Training hours", 0.0, 90.0, 30.0)
            skill = st.slider("Skill assessment", 30.0, 100.0, 75.0)
        with c3:
            group_header("sliders", "Feedback and engagement", "Qualitative signals", "blue")
            feedback = st.slider("Manager feedback score", 1.0, 5.0, 3.5, 0.1)
            engagement = st.slider("Engagement score", 1.0, 5.0, 3.5, 0.1)
            overtime = st.slider("Overtime hours/week", 0.0, 25.0, 3.0)
        submitted = st.form_submit_button("Run hypothetical prediction", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([{"department": pred_department, "role": pred_role, "tenure_years": tenure, "prior_projects_completed": projects, "workload_hours_week": workload, "attendance_rate": attendance, "training_hours": training, "skill_assessment": skill, "manager_feedback_score": feedback, "engagement_score": engagement, "overtime_hours": overtime}])
        try:
            prediction = model.predict(sample)[0]
            probabilities = model.predict_proba(sample)[0]
            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
            pleft, pright = st.columns([.8, 1.5], gap="large")
            with pleft:
                result_color = COLORS.get(str(prediction), "#0D9488")
                st.markdown(
                    f'<div class="result" style="border-top:5px solid {result_color}">'
                    f'<div class="rl">Model category output</div>'
                    f'<div class="rv" style="color:{result_color}">{prediction}</div>'
                    f'<span class="badge" style="background:{result_color}22;color:#14243A">Top class probability {max(probabilities):.1%}</span>'
                    f'<div class="rs">Hypothetical scenario only. This is a model output on synthetic data, not an assessment of a real person.</div></div>',
                    unsafe_allow_html=True,
                )
            with pright:
                with st.container(border=True):
                    prob = pd.DataFrame({"Category": model.classes_, "Model probability": probabilities}).sort_values("Model probability")
                    figp = px.bar(prob, x="Model probability", y="Category", orientation="h", color="Category", color_discrete_map=COLORS, text=prob["Model probability"].map(lambda v: f"{v:.1%}"), title="Predicted class probabilities", labels={"Model probability": "Model probability"})
                    figp.update_layout(**LAYOUT, height=270, showlegend=False, xaxis_range=[0, 1], bargap=0.35)
                    figp.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0,
                                       hovertemplate="%{y}: %{x:.1%}<extra></extra>")
                    figp.update_xaxes(gridcolor=GRID, zeroline=False, tickformat=".0%")
                    figp.update_yaxes(showgrid=False)
                    st.plotly_chart(figp, use_container_width=True, config=CONFIG, theme=None)
            callout("This output comes from a model trained on synthetic data. A class probability is not a measure of employee worth and is not a recommendation for an employment action.", "warning")
        except Exception as exc:
            callout("Prediction could not be generated: " + html.escape(str(exc)), "error")
    else:
        st.markdown(
            f'<div class="empty">{_icon("play", "#5B8DEF", 26)}<div>Set the hypothetical inputs above, then select '
            '<b>Run hypothetical prediction</b> to see the model output and class probabilities.</div></div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Dataset explorer
# ---------------------------------------------------------------------------
with tab_data:
    section("Dataset explorer", "Preview the synthetic records. Employee identifiers are hidden from this view.", first=True)
    preview = filtered.drop(columns=["employee_id"], errors="ignore")
    q1, q2, q3 = st.columns(3)
    q1.markdown(kpi_card("rows", "Rows in view", f"{len(preview):,}", "After department filter", "navy", compact=True), unsafe_allow_html=True)
    q2.markdown(kpi_card("columns", "Columns", f"{preview.shape[1]:,}", "Employee identifiers hidden", "teal", compact=True), unsafe_allow_html=True)
    q3.markdown(kpi_card("alert", "Missing values", f"{int(preview.isna().sum().sum()):,}", "Across all columns", "amber", compact=True), unsafe_allow_html=True)

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='panel-note' style='margin:0 0 .5rem 0'>Showing the first 100 rows of the current view.</div>", unsafe_allow_html=True)
        render_table(preview.head(100), max_height=420)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.download_button("Download filtered CSV", preview.to_csv(index=False).encode("utf-8"), "synthetic_employee_performance_filtered.csv", "text/csv")
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    with st.expander("View column types and missing-value counts"):
        profile = pd.DataFrame({"Column": preview.columns, "Data type": [str(t) for t in preview.dtypes], "Missing values": preview.isna().sum().values})
        render_table(profile)

st.markdown("<div style='border-top:1px solid #E3E8EF;margin-top:2rem;padding-top:1rem;color:#64748B;font-size:.78rem'>Employee Performance Predictor · Academic portfolio project · Synthetic data only</div>", unsafe_allow_html=True)