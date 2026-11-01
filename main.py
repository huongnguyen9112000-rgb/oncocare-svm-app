from pathlib import Path
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "metadata.json"

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Clinical Decision Support System"
)

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Exactly 30 named numeric features")

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str

def build_vector(payload: PredictionRequest) -> np.ndarray:
    expected = metadata["feature_names"]
    received = set(payload.features)
    missing = sorted(set(expected) - received)
    extra = sorted(received - set(expected))

    if missing or extra:
        raise HTTPException(status_code=422, detail={"missing": missing, "extra": extra})

    values = np.array([[payload.features[name] for name in expected]], dtype=np.float64)
    if not np.isfinite(values).all():
        raise HTTPException(status_code=422, detail="Features must be finite numbers")
    return values

HOME_HTML = r"""
<!DOCTYPE html>
<html lang="vi" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light dark">
<title>OncoCare AI - Hệ Thống Hỗ Trợ Chẩn Đoán Lâm Sàng</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
:root {
  --bg: #f4f7fb;
  --bg-grad: radial-gradient(900px 500px at 85% -10%, #d9f5ef 0%, transparent 60%), radial-gradient(700px 400px at -10% 10%, #e0ecff 0%, transparent 55%);
  --surface: #ffffff;
  --surface-2: #f8fafc;
  --border: #e5eaf1;
  --text: #0f172a;
  --muted: #64748b;
  --primary: #0d9488;
  --primary-600: #0f766e;
  --primary-soft: #ccfbf1;
  --primary-ink: #115e59;
  --ok: #16a34a; --ok-soft: #dcfce7;
  --bad: #e11d48; --bad-soft: #ffe4e6;
  --warn: #d97706; --warn-soft: #fef3c7;
  --shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 30px rgba(15,23,42,.05);
  --radius: 18px;
  --sidebar-w: 272px;
}
[data-theme="dark"] {
  --bg: #0b1220;
  --bg-grad: radial-gradient(900px 500px at 85% -10%, rgba(13,148,136,.22) 0%, transparent 60%), radial-gradient(700px 400px at -10% 10%, rgba(59,130,246,.14) 0%, transparent 55%);
  --surface: #111a2e;
  --surface-2: #0f1729;
  --border: #1f2b45;
  --text: #e8eefc;
  --muted: #8fa0c0;
  --primary: #2dd4bf;
  --primary-600: #14b8a6;
  --primary-soft: rgba(45,212,191,.14);
  --primary-ink: #5eead4;
  --ok: #4ade80; --ok-soft: rgba(74,222,128,.14);
  --bad: #fb7185; --bad-soft: rgba(251,113,133,.14);
  --warn: #fbbf24; --warn-soft: rgba(251,191,36,.14);
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 10px 40px rgba(0,0,0,.35);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; font-family: 'Plus Jakarta Sans', system-ui, sans-serif; color: var(--text);
  background: var(--bg) var(--bg-grad) no-repeat; background-attachment: fixed;
  -webkit-font-smoothing: antialiased; transition: background-color .3s, color .3s;
}
button, input { font-family: inherit; }

/* ---------- Sidebar ---------- */
.sidebar {
  position: fixed; inset: 0 auto 0 0; width: var(--sidebar-w); background: var(--surface);
  border-right: 1px solid var(--border); display: flex; flex-direction: column; z-index: 50;
  transition: transform .3s ease, background-color .3s;
}
.brand { display: flex; align-items: center; gap: 12px; padding: 26px 22px 22px; }
.brand-logo {
  width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center; color: #fff; font-size: 1.15rem;
  background: linear-gradient(135deg, #14b8a6, #0d9488 55%, #0f766e); box-shadow: 0 8px 20px rgba(13,148,136,.35);
}
.brand b { display: block; font-size: 1.05rem; font-weight: 800; letter-spacing: -.01em; }
.brand small { color: var(--muted); font-size: .72rem; font-weight: 500; }
.nav { list-style: none; margin: 6px 0 0; padding: 0 14px; display: grid; gap: 4px; }
.nav-label { padding: 14px 24px 6px; font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 700; }
.nav a {
  display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 12px; cursor: pointer; user-select: none;
  color: var(--muted); font-weight: 600; font-size: .9rem; transition: .18s; position: relative;
}
.nav a i { width: 20px; text-align: center; font-size: .95rem; }
.nav a:hover { background: var(--primary-soft); color: var(--primary-ink); }
.nav a.active { background: var(--primary-soft); color: var(--primary-ink); }
.nav a.active::before { content: ""; position: absolute; left: -14px; top: 10px; bottom: 10px; width: 4px; border-radius: 0 4px 4px 0; background: var(--primary); }
.nav .count { margin-left: auto; background: var(--primary); color: #fff; font-size: .7rem; padding: 2px 8px; border-radius: 99px; font-weight: 700; display: none; }
.side-foot { margin: auto 16px 18px; padding: 16px; border-radius: 16px; background: var(--surface-2); border: 1px solid var(--border); }
.status { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: .85rem; }
.pulse { width: 10px; height: 10px; border-radius: 50%; background: var(--ok); position: relative; }
.pulse::after { content: ""; position: absolute; inset: -4px; border-radius: 50%; border: 2px solid var(--ok); opacity: .5; animation: ping 1.8s infinite; }
@keyframes ping { 0% { transform: scale(.6); opacity: .8; } 100% { transform: scale(1.6); opacity: 0; } }
.side-foot small { display: block; margin-top: 6px; color: var(--muted); font-size: .72rem; }

/* ---------- Main ---------- */
.main { margin-left: var(--sidebar-w); padding: 28px 40px 60px; max-width: 1360px; }
.topbar { display: flex; align-items: center; gap: 14px; margin-bottom: 22px; }
.burger { display: none; width: 42px; height: 42px; border-radius: 12px; border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer; }
.crumb { color: var(--muted); font-size: .85rem; font-weight: 600; }
.crumb b { color: var(--text); }
.spacer { flex: 1; }
.icon-btn { width: 42px; height: 42px; border-radius: 12px; border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer; transition: .18s; box-shadow: var(--shadow); }
.icon-btn:hover { color: var(--primary); transform: translateY(-1px); }

.hero {
  position: relative; overflow: hidden; border-radius: 24px; padding: 30px 34px; color: #fff; margin-bottom: 28px;
  background: linear-gradient(120deg, #0f766e 0%, #0d9488 45%, #0891b2 100%); box-shadow: 0 20px 50px rgba(13,148,136,.28);
  display: flex; justify-content: space-between; align-items: center; gap: 24px; flex-wrap: wrap;
}
.hero::before, .hero::after { content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,.08); }
.hero::before { width: 320px; height: 320px; right: -80px; top: -140px; }
.hero::after { width: 200px; height: 200px; right: 180px; bottom: -120px; }
.hero > * { position: relative; z-index: 1; }
.hero h1 { margin: 0 0 6px; font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; }
.hero p { margin: 0; opacity: .88; font-size: .93rem; max-width: 520px; line-height: 1.55; }
.hero-kpis { display: flex; gap: 14px; }
.kpi { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22); backdrop-filter: blur(8px); border-radius: 16px; padding: 14px 20px; min-width: 130px; }
.kpi span { display: block; font-size: .68rem; text-transform: uppercase; letter-spacing: .1em; opacity: .85; font-weight: 700; }
.kpi strong { font-size: 1.35rem; font-weight: 800; }

.tab { display: none; }
.tab.active { display: block; animation: rise .35s ease; }
@keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

.page-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; }
.page-head h2 { margin: 0 0 4px; font-size: 1.35rem; font-weight: 800; letter-spacing: -.02em; }
.page-head p { margin: 0; color: var(--muted); font-size: .88rem; }

.btn { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--border); background: var(--surface); color: var(--text); padding: 10px 16px; border-radius: 12px; font-weight: 700; font-size: .85rem; cursor: pointer; transition: .18s; }
.btn:hover { transform: translateY(-1px); box-shadow: var(--shadow); }
.btn-ok { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 40%, transparent); background: var(--ok-soft); }
.btn-bad { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 40%, transparent); background: var(--bad-soft); }
.btn-primary { background: linear-gradient(135deg, #14b8a6, #0d9488); color: #fff; border: 0; padding: 15px 38px; font-size: .98rem; border-radius: 14px; box-shadow: 0 12px 28px rgba(13,148,136,.35); }
.btn-primary:hover { box-shadow: 0 16px 34px rgba(13,148,136,.45); }
.btn-primary:disabled { opacity: .7; cursor: wait; transform: none; }
.btn-ghost { background: transparent; }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); transition: background-color .3s; }

/* progress of form */
.fill-bar { display: flex; align-items: center; gap: 14px; padding: 14px 20px; margin-bottom: 16px; }
.fill-bar .track { flex: 1; height: 8px; border-radius: 99px; background: var(--surface-2); border: 1px solid var(--border); overflow: hidden; }
.fill-bar .track i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, #14b8a6, #22d3ee); border-radius: 99px; transition: width .4s; }
.fill-bar span { font-size: .82rem; font-weight: 700; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

.group { padding: 22px 24px 24px; margin-bottom: 16px; }
.group-head { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.group-ico { width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center; font-size: .95rem; }
.g-mean .group-ico { background: var(--primary-soft); color: var(--primary); }
.g-err .group-ico { background: var(--warn-soft); color: var(--warn); }
.g-worst .group-ico { background: var(--bad-soft); color: var(--bad); }
.group-head h3 { margin: 0; font-size: .98rem; font-weight: 800; }
.group-head small { color: var(--muted); font-size: .78rem; font-weight: 500; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
.field { background: var(--surface-2); border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px 12px; transition: .18s; }
.field:hover { border-color: var(--primary); }
.field:focus-within { border-color: var(--primary); box-shadow: 0 0 0 4px color-mix(in srgb, var(--primary) 16%, transparent); background: var(--surface); }
.field.invalid { border-color: var(--bad); box-shadow: 0 0 0 4px color-mix(in srgb, var(--bad) 14%, transparent); }
.field label { display: block; font-size: .82rem; font-weight: 700; }
.field .hint { color: var(--muted); font-size: .72rem; line-height: 1.35; min-height: 2.1em; margin: 2px 0 8px; }
.field input { width: 100%; border: 0; outline: 0; background: transparent; color: var(--text); font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: .95rem; padding: 2px 0; border-bottom: 1px dashed var(--border); }
.field input:focus { border-bottom-color: var(--primary); }

.submit-zone { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 20px 0 6px; }
.submit-zone small { color: var(--muted); font-size: .78rem; }

/* result */
.result { padding: 30px; margin-top: 20px; border-width: 2px; display: none; position: relative; overflow: hidden; }
.result.show { display: block; animation: rise .45s ease; }
.result.is-ok { border-color: color-mix(in srgb, var(--ok) 50%, transparent); }
.result.is-bad { border-color: color-mix(in srgb, var(--bad) 50%, transparent); }
.result-top { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 22px; flex-wrap: wrap; }
.result-top h3 { margin: 0; font-size: 1.05rem; font-weight: 800; display: flex; gap: 10px; align-items: center; }
.chip { font-family: 'JetBrains Mono', monospace; font-size: .75rem; font-weight: 600; padding: 6px 12px; border-radius: 99px; background: var(--surface-2); border: 1px solid var(--border); color: var(--muted); }
.banner { display: flex; gap: 12px; align-items: flex-start; padding: 14px 18px; border-radius: 14px; font-weight: 600; font-size: .9rem; line-height: 1.5; margin-bottom: 24px; }
.banner.ok { background: var(--ok-soft); color: var(--ok); }
.banner.bad { background: var(--bad-soft); color: var(--bad); }
.result-grid { display: grid; grid-template-columns: 260px 1fr; gap: 34px; align-items: center; }
.gauge { position: relative; width: 220px; height: 220px; margin: 0 auto; }
.gauge svg { transform: rotate(-90deg); }
.gauge .bg { stroke: var(--surface-2); }
.gauge .fg { transition: stroke-dashoffset 1.1s cubic-bezier(.4,0,.2,1); stroke-linecap: round; }
.gauge-center { position: absolute; inset: 0; display: grid; place-content: center; text-align: center; }
.gauge-center b { font-size: 2.3rem; font-weight: 800; letter-spacing: -.03em; line-height: 1; }
.gauge-center span { font-size: .7rem; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: .1em; margin-top: 6px; }
.verdict small { color: var(--muted); font-weight: 700; text-transform: uppercase; font-size: .72rem; letter-spacing: .1em; }
.verdict h2 { margin: 6px 0 18px; font-size: 2rem; font-weight: 800; letter-spacing: -.02em; }
.prob { margin-bottom: 14px; }
.prob-row { display: flex; justify-content: space-between; font-size: .88rem; font-weight: 700; margin-bottom: 8px; }
.prob-row b { font-family: 'JetBrains Mono', monospace; }
.bar { height: 12px; border-radius: 99px; background: var(--surface-2); border: 1px solid var(--border); overflow: hidden; }
.bar i { display: block; height: 100%; width: 0; border-radius: 99px; transition: width 1s cubic-bezier(.4,0,.2,1); }
.note { display: flex; gap: 10px; font-size: .8rem; color: var(--muted); padding: 14px 16px; border-radius: 12px; background: var(--surface-2); border: 1px dashed var(--border); line-height: 1.5; margin-top: 8px; }

/* table */
.table-wrap { overflow-x: auto; padding: 6px 10px 10px; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 14px 16px; font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 700; border-bottom: 1px solid var(--border); }
td { padding: 14px 16px; border-bottom: 1px solid var(--border); font-size: .88rem; font-weight: 600; }
tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: var(--surface-2); }
.badge { padding: 5px 12px; border-radius: 99px; font-size: .75rem; font-weight: 700; }
.badge.ok { background: var(--ok-soft); color: var(--ok); }
.badge.bad { background: var(--bad-soft); color: var(--bad); }
.empty { text-align: center; padding: 56px 16px !important; color: var(--muted); }
.empty i { display: block; font-size: 2rem; margin-bottom: 10px; opacity: .5; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* stats */
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }
.stat { padding: 22px 24px; position: relative; overflow: hidden; }
.stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--c); }
.stat span { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
.stat h3 { margin: 8px 0 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -.03em; color: var(--c); }
.stat i { position: absolute; right: 18px; top: 18px; font-size: 1.6rem; color: var(--c); opacity: .2; }
.donut-card { padding: 28px; display: flex; gap: 36px; align-items: center; flex-wrap: wrap; justify-content: center; }
.donut { width: 190px; height: 190px; border-radius: 50%; display: grid; place-items: center; background: conic-gradient(var(--border) 0 100%); transition: background .6s; }
.donut::after { content: attr(data-label); width: 130px; height: 130px; border-radius: 50%; background: var(--surface); display: grid; place-items: center; font-weight: 800; font-size: 1.1rem; text-align: center; padding: 10px; white-space: pre-line; }
.legend { display: grid; gap: 12px; font-weight: 700; font-size: .9rem; }
.legend div { display: flex; align-items: center; gap: 10px; }
.dot { width: 12px; height: 12px; border-radius: 4px; }

/* guide */
.search { position: relative; max-width: 340px; flex: 1; }
.search i { position: absolute; left: 14px; top: 50%; transform: translateY(-50%); color: var(--muted); }
.search input { width: 100%; padding: 12px 14px 12px 40px; border-radius: 12px; border: 1px solid var(--border); background: var(--surface); color: var(--text); font-weight: 500; outline: 0; }
.search input:focus { border-color: var(--primary); box-shadow: 0 0 0 4px color-mix(in srgb, var(--primary) 16%, transparent); }
.guide-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; padding: 20px; }
.guide-item { padding: 16px 18px; border: 1px solid var(--border); border-radius: 14px; background: var(--surface-2); transition: .18s; }
.guide-item:hover { border-color: var(--primary); transform: translateY(-2px); }
.guide-item h4 { margin: 0 0 6px; font-size: .92rem; font-weight: 800; color: var(--primary-ink); display: flex; gap: 8px; align-items: center; }
.guide-item h4 em { font-style: normal; font-size: .7rem; background: var(--primary-soft); padding: 2px 8px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; }
.guide-item code { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: var(--muted); margin-bottom: 6px; }
.guide-item p { margin: 0; font-size: .82rem; color: var(--muted); line-height: 1.5; }

/* toast */
.toasts { position: fixed; right: 22px; bottom: 22px; display: grid; gap: 10px; z-index: 200; }
.toast { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--bad); padding: 14px 18px; border-radius: 12px; box-shadow: var(--shadow); font-size: .86rem; font-weight: 600; max-width: 340px; animation: rise .3s; }
.toast.ok { border-left-color: var(--ok); }
.scrim { display: none; position: fixed; inset: 0; background: rgba(2,6,23,.5); z-index: 40; }

@media (max-width: 960px) {
  .sidebar { transform: translateX(-100%); }
  .sidebar.open { transform: none; box-shadow: 0 0 60px rgba(0,0,0,.4); }
  .sidebar.open ~ .scrim { display: block; }
  .main { margin-left: 0; padding: 18px 16px 50px; }
  .burger { display: block; }
  .result-grid { grid-template-columns: 1fr; }
  .stat-grid { grid-template-columns: 1fr; }
  .hero { padding: 24px 22px; }
  .hero-kpis { width: 100%; }
  .kpi { flex: 1; min-width: 0; }
}
</style>
</head>
<body>

<aside class="sidebar" id="sidebar">
  <div class="brand">
    <div class="brand-logo"><i class="fa-solid fa-notes-medical"></i></div>
    <div><b>OncoCare AI</b><small>Clinical Decision Support</small></div>
  </div>
  <div class="nav-label">Menu</div>
  <ul class="nav">
    <li><a class="active" data-tab="tab-predict"><i class="fa-solid fa-stethoscope"></i> Trạm Chẩn Đoán AI</a></li>
    <li><a data-tab="tab-history"><i class="fa-solid fa-clock-rotate-left"></i> Nhật Ký Khám Bệnh <span class="count" id="navCount">0</span></a></li>
    <li><a data-tab="tab-stats"><i class="fa-solid fa-chart-pie"></i> Thống Kê &amp; Báo Cáo</a></li>
    <li><a data-tab="tab-guide"><i class="fa-solid fa-book-medical"></i> Hướng Dẫn Chỉ Số</a></li>
  </ul>
  <div class="side-foot">
    <div class="status"><span class="pulse"></span> Hệ thống sẵn sàng</div>
    <small id="verText">SVM Engine</small>
  </div>
</aside>
<div class="scrim" id="scrim"></div>

<main class="main">
  <div class="topbar">
    <button class="burger" id="burger" aria-label="Menu"><i class="fa-solid fa-bars"></i></button>
    <div class="crumb">OncoCare / <b id="crumbText">Trạm Chẩn Đoán AI</b></div>
    <div class="spacer"></div>
    <button class="icon-btn" id="themeBtn" title="Đổi giao diện sáng/tối"><i class="fa-solid fa-moon"></i></button>
  </div>

  <section class="hero">
    <div>
      <h1>Hệ Thống Phân Tích Khối U Vú (SVM)</h1>
      <p>Hỗ trợ phân loại lành tính / ác tính dựa trên 30 đặc trưng tế bào học chuẩn từ bộ dữ liệu Wisconsin Breast Cancer.</p>
    </div>
    <div class="hero-kpis">
      <div class="kpi"><span>Độ chính xác</span><strong>98.24%</strong></div>
      <div class="kpi"><span>Đặc trưng</span><strong>30 chỉ số</strong></div>
    </div>
  </section>

  <!-- TAB: PREDICT -->
  <div id="tab-predict" class="tab active">
    <div class="page-head">
      <div>
        <h2>Nhập Dữ Liệu Hình Thái Học Tế Bào</h2>
        <p>Điền các thông số hoặc nạp nhanh bộ mẫu chuẩn để kiểm tra.</p>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button type="button" class="btn btn-ok" onclick="loadSampleData('benign')"><i class="fa-solid fa-shield-halved"></i> Mẫu Lành Tính</button>
        <button type="button" class="btn btn-bad" onclick="loadSampleData('malignant')"><i class="fa-solid fa-biohazard"></i> Mẫu Ác Tính</button>
        <button type="button" class="btn btn-ghost" onclick="clearForm()"><i class="fa-solid fa-eraser"></i> Xóa</button>
      </div>
    </div>

    <div class="card fill-bar">
      <i class="fa-solid fa-list-check" style="color:var(--primary)"></i>
      <div class="track"><i id="fillTrack"></i></div>
      <span id="fillText">0/30</span>
    </div>

    <form id="predictionForm" novalidate>
      <div class="card group g-mean">
        <div class="group-head"><div class="group-ico"><i class="fa-solid fa-circle-dot"></i></div>
          <div><h3>1. Nhóm Chỉ Số Trung Bình</h3><small>Mean Features · 10 chiều</small></div></div>
        <div class="grid" id="groupMean"></div>
      </div>
      <div class="card group g-err">
        <div class="group-head"><div class="group-ico"><i class="fa-solid fa-chart-line"></i></div>
          <div><h3>2. Nhóm Sai Số Tiêu Chuẩn</h3><small>Standard Error · 10 chiều</small></div></div>
        <div class="grid" id="groupError"></div>
      </div>
      <div class="card group g-worst">
        <div class="group-head"><div class="group-ico"><i class="fa-solid fa-triangle-exclamation"></i></div>
          <div><h3>3. Nhóm Giá Trị Cực Đoan</h3><small>Worst Features · 10 chiều</small></div></div>
        <div class="grid" id="groupWorst"></div>
      </div>

      <div class="submit-zone">
        <button type="submit" id="btnSubmit" class="btn btn-primary"><i class="fa-solid fa-wand-magic-sparkles"></i> THỰC HIỆN CHẨN ĐOÁN AI</button>
        <small>Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán giải phẫu bệnh.</small>
      </div>
    </form>

    <div class="card result" id="resultCard">
      <div class="result-top">
        <h3><i class="fa-solid fa-file-medical" style="color:var(--primary)"></i> Báo Cáo Kết Quả Chẩn Đoán</h3>
        <span class="chip" id="resId">#CA-0000</span>
      </div>
      <div class="banner" id="resultAlert"></div>
      <div class="result-grid">
        <div class="gauge">
          <svg width="220" height="220" viewBox="0 0 140 140">
            <circle class="bg" cx="70" cy="70" r="58" fill="none" stroke-width="13"/>
            <circle class="fg" id="gaugeFg" cx="70" cy="70" r="58" fill="none" stroke-width="13" stroke-dasharray="364.4" stroke-dashoffset="364.4"/>
          </svg>
          <div class="gauge-center"><b id="gaugeVal">0%</b><span>Nguy cơ ác tính</span></div>
        </div>
        <div>
          <div class="verdict"><small>Kết luận phân loại mô hình</small><h2 id="resPrediction"></h2></div>
          <div class="prob">
            <div class="prob-row"><span>Xác suất Lành tính (Benign)</span><b id="resBenignText" style="color:var(--ok)">0%</b></div>
            <div class="bar"><i id="barBenign" style="background:var(--ok)"></i></div>
          </div>
          <div class="prob">
            <div class="prob-row"><span>Xác suất Ác tính (Malignant)</span><b id="resMalignantText" style="color:var(--bad)">0%</b></div>
            <div class="bar"><i id="barMalignant" style="background:var(--bad)"></i></div>
          </div>
          <div class="note"><i class="fa-solid fa-circle-info" style="color:var(--primary);margin-top:2px"></i>
            <span id="warnText">Kết quả từ thuật toán SVM chỉ hỗ trợ tham khảo lâm sàng.</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB: HISTORY -->
  <div id="tab-history" class="tab">
    <div class="page-head"><div><h2>Nhật Ký Khám Bệnh</h2><p>Các ca phân tích trong phiên làm việc hiện tại.</p></div>
      <button class="btn" onclick="clearHistory()"><i class="fa-solid fa-trash-can"></i> Xóa nhật ký</button></div>
    <div class="card table-wrap">
      <table>
        <thead><tr><th>Mã ca</th><th>Thời gian</th><th>Kết luận</th><th>Xác suất ác tính</th><th></th></tr></thead>
        <tbody id="historyTableBody"></tbody>
      </table>
    </div>
  </div>

  <!-- TAB: STATS -->
  <div id="tab-stats" class="tab">
    <div class="page-head"><div><h2>Thống Kê Hệ Thống</h2><p>Tổng quan phân phối kết quả trong phiên.</p></div></div>
    <div class="stat-grid">
      <div class="card stat" style="--c:var(--primary)"><i class="fa-solid fa-vial"></i><span>Tổng số ca</span><h3 id="statTotal">0</h3></div>
      <div class="card stat" style="--c:var(--ok)"><i class="fa-solid fa-shield-heart"></i><span>Ca lành tính</span><h3 id="statBenign">0</h3></div>
      <div class="card stat" style="--c:var(--bad)"><i class="fa-solid fa-biohazard"></i><span>Ca ác tính</span><h3 id="statMalignant">0</h3></div>
    </div>
    <div class="card donut-card">
      <div class="donut" id="donut" data-label="Chưa có&#10;dữ liệu"></div>
      <div class="legend">
        <div><span class="dot" style="background:var(--ok)"></span> Lành tính: <span id="lgB">0%</span></div>
        <div><span class="dot" style="background:var(--bad)"></span> Ác tính: <span id="lgM">0%</span></div>
      </div>
    </div>
  </div>

  <!-- TAB: GUIDE -->
  <div id="tab-guide" class="tab">
    <div class="page-head"><div><h2>Tra Cứu 30 Chỉ Số Lâm Sàng</h2><p>Ý nghĩa kỹ thuật của các đặc trưng trích xuất tế bào học.</p></div>
      <div class="search"><i class="fa-solid fa-magnifying-glass"></i><input id="guideSearch" placeholder="Tìm chỉ số..."></div></div>
    <div class="card"><div class="guide-grid" id="guideContainer"></div></div>
  </div>
</main>

<div class="toasts" id="toasts"></div>

<script>
const featureMap = {
  "mean radius": { name: "Bán kính trung bình", desc: "Kích thước trung bình nhân tế bào" },
  "mean texture": { name: "Độ mịn/Kết cấu bề mặt", desc: "Độ lệch chuẩn giá trị xám bề mặt" },
  "mean perimeter": { name: "Chu vi trung bình", desc: "Chu vi trung bình của nhân" },
  "mean area": { name: "Diện tích trung bình", desc: "Diện tích trung bình của nhân tế bào" },
  "mean smoothness": { name: "Độ trơn nhẵn TB", desc: "Độ biến thiên cục bộ của bán kính" },
  "mean compactness": { name: "Độ nén trung bình", desc: "Tính theo (chu vi^2 / diện tích - 1.0)" },
  "mean concavity": { name: "Độ lõm trung bình", desc: "Mức độ sâu các phần lõm trên bờ viền" },
  "mean concave points": { name: "Số điểm lõm TB", desc: "Số lượng các phần lõm trên đường viền" },
  "mean symmetry": { name: "Độ đối xứng TB", desc: "Mức độ cân đối hình dáng tế bào" },
  "mean fractal dimension": { name: "Độ chiều Fractal", desc: "Độ phức tạp bờ viền (Xấp xỉ fractal)" },
  "radius error": { name: "Sai số bán kính", desc: "Độ lệch chuẩn kích thước bán kính" },
  "texture error": { name: "Sai số kết cấu", desc: "Độ lệch chuẩn độ mịn bề mặt" },
  "perimeter error": { name: "Sai số chu vi", desc: "Độ lệch chuẩn chu vi nhân" },
  "area error": { name: "Sai số diện tích", desc: "Độ lệch chuẩn diện tích tế bào" },
  "smoothness error": { name: "Sai số độ nhẵn", desc: "Độ biến thiên sai số độ mịn" },
  "compactness error": { name: "Sai số độ nén", desc: "Sai số mức độ nén tế bào" },
  "concavity error": { name: "Sai số độ lõm", desc: "Sai số độ sâu phần lõm bờ viền" },
  "concave points error": { name: "Sai số điểm lõm", desc: "Sai số số lượng điểm lõm" },
  "symmetry error": { name: "Sai số độ đối xứng", desc: "Sai số mức độ cân đối" },
  "fractal dimension error": { name: "Sai số chiều Fractal", desc: "Sai số độ phức tạp bờ viền" },
  "worst radius": { name: "Bán kính lớn nhất", desc: "Giá trị lớn nhất của bán kính" },
  "worst texture": { name: "Kết cấu lớn nhất", desc: "Giá trị lớn nhất kết cấu bề mặt" },
  "worst perimeter": { name: "Chu vi lớn nhất", desc: "Chu vi lớn nhất ghi nhận được" },
  "worst area": { name: "Diện tích lớn nhất", desc: "Diện tích lớn nhất ghi nhận được" },
  "worst smoothness": { name: "Độ nhẵn lớn nhất", desc: "Độ trơn nhẵn cao nhất trong nhóm" },
  "worst compactness": { name: "Độ nén lớn nhất", desc: "Độ nén lớn nhất ghi nhận" },
  "worst concavity": { name: "Độ lõm lớn nhất", desc: "Độ lõm sâu nhất ghi nhận" },
  "worst concave points": { name: "Điểm lõm lớn nhất", desc: "Số điểm lõm lớn nhất ghi nhận" },
  "worst symmetry": { name: "Đối xứng lớn nhất", desc: "Độ lệch đối xứng lớn nhất" },
  "worst fractal dimension": { name: "Chiều Fractal lớn nhất", desc: "Độ phức tạp bờ viền lớn nhất" }
};

const sampleMalignant = {"mean radius":17.99,"mean texture":10.38,"mean perimeter":122.8,"mean area":1001.0,"mean smoothness":0.1184,"mean compactness":0.2776,"mean concavity":0.3001,"mean concave points":0.1471,"mean symmetry":0.2419,"mean fractal dimension":0.07871,"radius error":1.095,"texture error":0.9053,"perimeter error":8.589,"area error":153.4,"smoothness error":0.006399,"compactness error":0.04904,"concavity error":0.05373,"concave points error":0.01587,"symmetry error":0.03003,"fractal dimension error":0.006193,"worst radius":25.38,"worst texture":17.33,"worst perimeter":184.6,"worst area":2019.0,"worst smoothness":0.1622,"worst compactness":0.6656,"worst concavity":0.7119,"worst concave points":0.2654,"worst symmetry":0.4601,"worst fractal dimension":0.1189};
const sampleBenign = {"mean radius":13.54,"mean texture":14.36,"mean perimeter":87.46,"mean area":566.3,"mean smoothness":0.09779,"mean compactness":0.08129,"mean concavity":0.06664,"mean concave points":0.04781,"mean symmetry":0.1885,"mean fractal dimension":0.05766,"radius error":0.2699,"texture error":0.7886,"perimeter error":2.058,"area error":23.56,"smoothness error":0.008462,"compactness error":0.0146,"concavity error":0.02387,"concave points error":0.01315,"symmetry error":0.0198,"fractal dimension error":0.0023,"worst radius":15.11,"worst texture":19.26,"worst perimeter":99.7,"worst area":711.2,"worst smoothness":0.144,"worst compactness":0.1773,"worst concavity":0.239,"worst concave points":0.1288,"worst symmetry":0.2977,"worst fractal dimension":0.07259};

const $ = id => document.getElementById(id);
const idOf = f => 'inp_' + f.split(' ').join('_');
let featureNames = [], historyList = [], stats = { total: 0, benign: 0, malignant: 0 }, caseSeq = 0;

/* ---------- Theme ---------- */
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  $('themeBtn').innerHTML = t === 'dark' ? '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
  try { localStorage.setItem('onco-theme', t); } catch (e) {}
}
(function () {
  let t = 'light';
  try { t = localStorage.getItem('onco-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch (e) {}
  applyTheme(t);
})();
$('themeBtn').onclick = () => applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');

/* ---------- Navigation ---------- */
const titles = { 'tab-predict': 'Trạm Chẩn Đoán AI', 'tab-history': 'Nhật Ký Khám Bệnh', 'tab-stats': 'Thống Kê & Báo Cáo', 'tab-guide': 'Hướng Dẫn Chỉ Số' };
function switchTab(id) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav a').forEach(el => el.classList.toggle('active', el.dataset.tab === id));
  $(id).classList.add('active');
  $('crumbText').textContent = titles[id];
  closeSidebar();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
document.querySelectorAll('.nav a').forEach(a => a.onclick = () => switchTab(a.dataset.tab));
const closeSidebar = () => $('sidebar').classList.remove('open');
$('burger').onclick = () => $('sidebar').classList.toggle('open');
$('scrim').onclick = closeSidebar;

/* ---------- Toast ---------- */
function toast(msg, ok) {
  const t = document.createElement('div');
  t.className = 'toast' + (ok ? ' ok' : '');
  t.textContent = msg;
  $('toasts').appendChild(t);
  setTimeout(() => t.remove(), 4200);
}

/* ---------- Init & render ---------- */
async function init() {
  try {
    const res = await fetch('/metadata');
    const data = await res.json();
    featureNames = data.feature_names;
    if (data.model_version) $('verText').textContent = 'SVM Engine v' + data.model_version;
    if (data.warning) $('warnText').textContent = data.warning;
  } catch (e) {
    toast('Không tải được metadata từ máy chủ.');
    return;
  }
  renderInputs();
  renderGuide();
  renderHistory();
  updateFill();
}

function renderInputs() {
  const g = { mean: '', error: '', worst: '' };
  featureNames.forEach(f => {
    const info = featureMap[f] || { name: f, desc: 'Chỉ số tế bào' };
    const html = `<div class="field" id="wrap_${idOf(f)}"><label for="${idOf(f)}">${info.name}</label><div class="hint">${info.desc}</div><input type="text" inputmode="decimal" autocomplete="off" name="${f}" id="${idOf(f)}" value="${sampleBenign[f] ?? ''}"></div>`;
    if (f.startsWith('mean')) g.mean += html;
    else if (f.includes('error')) g.error += html;
    else if (f.startsWith('worst')) g.worst += html;
  });
  $('groupMean').innerHTML = g.mean; $('groupError').innerHTML = g.error; $('groupWorst').innerHTML = g.worst;
  document.querySelectorAll('#predictionForm input').forEach(i => i.addEventListener('input', () => { i.parentElement.classList.remove('invalid'); updateFill(); }));
}

function renderGuide(filter) {
  const q = (filter || '').toLowerCase().trim();
  let html = '', i = 0;
  Object.keys(featureMap).forEach(k => {
    i++;
    const info = featureMap[k];
    if (q && !(k.toLowerCase().includes(q) || info.name.toLowerCase().includes(q) || info.desc.toLowerCase().includes(q))) return;
    html += `<div class="guide-item"><h4><em>${String(i).padStart(2, '0')}</em>${info.name}</h4><code>${k}</code><p>${info.desc}</p></div>`;
  });
  $('guideContainer').innerHTML = html || '<div class="empty" style="grid-column:1/-1;padding:30px">Không tìm thấy chỉ số phù hợp.</div>';
}
$('guideSearch').addEventListener('input', e => renderGuide(e.target.value));

function parseNum(v) { return parseFloat(String(v).trim().replace(',', '.')); }
function updateFill() {
  let n = 0;
  featureNames.forEach(f => { const el = $(idOf(f)); if (el && el.value.trim() !== '' && !isNaN(parseNum(el.value))) n++; });
  $('fillText').textContent = n + '/' + featureNames.length;
  $('fillTrack').style.width = (featureNames.length ? n / featureNames.length * 100 : 0) + '%';
}

function setValues(data) {
  featureNames.forEach(f => { const el = $(idOf(f)); if (el) { el.value = data[f] ?? ''; el.parentElement.classList.remove('invalid'); } });
  updateFill();
}
function loadSampleData(type) { setValues(type === 'malignant' ? sampleMalignant : sampleBenign); toast(type === 'malignant' ? 'Đã nạp mẫu ác tính' : 'Đã nạp mẫu lành tính', true); }
function clearForm() { setValues({}); $('resultCard').classList.remove('show'); }

/* ---------- History & stats ---------- */
function renderHistory() {
  const tb = $('historyTableBody');
  const nc = $('navCount');
  nc.style.display = historyList.length ? 'inline-block' : 'none';
  nc.textContent = historyList.length;
  if (!historyList.length) {
    tb.innerHTML = '<tr><td colspan="5" class="empty"><i class="fa-regular fa-folder-open"></i>Chưa có lịch sử chẩn đoán nào được ghi nhận.</td></tr>';
    return;
  }
  tb.innerHTML = historyList.map((h, idx) => `<tr><td class="mono">${h.id}</td><td style="color:var(--muted)">${h.time}</td><td><span class="badge ${h.malignant ? 'bad' : 'ok'}">${h.malignant ? 'Ác tính' : 'Lành tính'}</span></td><td class="mono">${h.prob}</td><td style="text-align:right"><button class="btn" style="padding:7px 14px" onclick="viewCase(${idx})"><i class="fa-regular fa-eye"></i> Xem</button></td></tr>`).join('');
}
function viewCase(idx) {
  const h = historyList[idx];
  setValues(h.features);
  switchTab('tab-predict');
  showResult(h.res, h.id);
}
function clearHistory() {
  historyList = []; stats = { total: 0, benign: 0, malignant: 0 }; renderHistory(); renderStats();
}
function renderStats() {
  $('statTotal').textContent = stats.total; $('statBenign').textContent = stats.benign; $('statMalignant').textContent = stats.malignant;
  const d = $('donut');
  if (!stats.total) { d.style.background = 'conic-gradient(var(--border) 0 100%)'; d.dataset.label = 'Chưa có\ndữ liệu'; $('lgB').textContent = '0%'; $('lgM').textContent = '0%'; return; }
  const b = stats.benign / stats.total * 100, m = 100 - b;
  d.style.background = `conic-gradient(var(--ok) 0 ${b}%, var(--bad) ${b}% 100%)`;
  d.dataset.label = stats.total + '\nca';
  $('lgB').textContent = b.toFixed(0) + '%'; $('lgM').textContent = m.toFixed(0) + '%';
}

/* ---------- Result ---------- */
const CIRC = 2 * Math.PI * 58;
function showResult(data, id) {
  const pB = data.probability_benign * 100, pM = data.probability_malignant * 100;
  const bad = data.predicted_label !== 'benign';
  const card = $('resultCard');
  card.classList.add('show');
  card.classList.toggle('is-bad', bad); card.classList.toggle('is-ok', !bad);
  $('resId').textContent = '#' + id;
  $('resBenignText').textContent = pB.toFixed(2) + '%';
  $('resMalignantText').textContent = pM.toFixed(2) + '%';
  $('gaugeVal').textContent = pM.toFixed(1) + '%';
  const fg = $('gaugeFg');
  fg.style.stroke = bad ? 'var(--bad)' : 'var(--ok)';
  fg.setAttribute('stroke-dasharray', CIRC);
  fg.style.strokeDashoffset = CIRC;
  $('barBenign').style.width = '0%'; $('barMalignant').style.width = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    fg.style.strokeDashoffset = CIRC * (1 - pM / 100);
    $('barBenign').style.width = pB + '%'; $('barMalignant').style.width = pM + '%';
  }));
  const alertEl = $('resultAlert');
  if (!bad) {
    $('resPrediction').innerHTML = '<span style="color:var(--ok)">LÀNH TÍNH</span> <small style="font-size:.9rem;color:var(--muted)">(Benign)</small>';
    alertEl.className = 'banner ok';
    alertEl.innerHTML = '<i class="fa-solid fa-circle-check"></i><span>Mô hình xác định khối u có đặc trưng lành tính. Vẫn nên theo dõi định kỳ theo chỉ định của bác sĩ.</span>';
  } else {
    $('resPrediction').innerHTML = '<span style="color:var(--bad)">ÁC TÍNH</span> <small style="font-size:.9rem;color:var(--muted)">(Malignant)</small>';
    alertEl.className = 'banner bad';
    alertEl.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i><span>Khối u mang đặc trưng ác tính. Cần sinh thiết và hội chẩn chuyên khoa để xác nhận.</span>';
  }
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ---------- Submit ---------- */
$('predictionForm').addEventListener('submit', async function (e) {
  e.preventDefault();
  const obj = {}; let bad = [];
  featureNames.forEach(f => {
    const el = $(idOf(f)); const v = parseNum(el.value);
    if (el.value.trim() === '' || !isFinite(v)) { bad.push(el); } else obj[f] = v;
  });
  if (bad.length) {
    bad.forEach(el => el.parentElement.classList.add('invalid'));
    bad[0].scrollIntoView({ behavior: 'smooth', block: 'center' }); bad[0].focus();
    toast('Có ' + bad.length + ' ô trống hoặc không phải số hợp lệ.');
    return;
  }
  const btn = $('btnSubmit');
  btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang phân tích...';
  try {
    const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features: obj }) });
    if (!res.ok) throw new Error('Máy chủ trả về lỗi ' + res.status);
    const data = await res.json();
    caseSeq++;
    const id = 'CA-' + String(caseSeq).padStart(4, '0');
    const isMal = data.predicted_label !== 'benign';
    stats.total++; isMal ? stats.malignant++ : stats.benign++;
    historyList.unshift({ id, time: new Date().toLocaleString('vi-VN'), malignant: isMal, prob: (data.probability_malignant * 100).toFixed(2) + '%', features: obj, res: data });
    showResult(data, id); renderHistory(); renderStats();
  } catch (err) {
    toast(err.message || 'Không thể kết nối máy chủ.');
  } finally {
    btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> THỰC HIỆN CHẨN ĐOÁN AI';
  }
});

init();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home_page():
    return HOME_HTML

@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata["model_version"]}

@app.get("/metadata")
def get_metadata():
    return metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    x = build_vector(payload)
    pred = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    classes = list(model.named_steps["svc"].classes_)
    p = {int(c): float(v) for c, v in zip(classes, proba)}
    return PredictionResponse(
        predicted_class=pred,
        predicted_label=metadata["class_mapping"][str(pred)],
        probability_malignant=p.get(0, 0.0),
        probability_benign=p.get(1, 0.0),
        model_version=metadata["model_version"],
        warning=metadata["warning"]
    )
