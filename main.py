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
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="only light">
<title>OncoCare - Hệ Thống Hỗ Trợ Chẩn Đoán Lâm Sàng</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<style>
:root {
  color-scheme: only light;
  --paper: #fbfcfc;
  --white: #ffffff;
  --mint: #eaf6f4;
  --sky: #eaf2fb;
  --ink: #12302f;
  --ink-2: #3d5a59;
  --muted: #7c9291;
  --line: #e1ebea;
  --line-2: #d2e0df;
  --teal: #0f7c78;
  --teal-2: #14a39d;
  --blue: #3b78b8;
  --ok: #2b9a6a;  --ok-bg: #e6f6ee;
  --warn: #d58a1c; --warn-bg: #fdf3e1;
  --bad: #cf4545; --bad-bg: #fcebeb;
  --serif: 'Fraunces', Georgia, serif;
  --sans: 'DM Sans', system-ui, sans-serif;
  --mono: 'DM Mono', ui-monospace, monospace;
  --shadow: 0 1px 2px rgba(18,48,47,.04), 0 10px 30px -12px rgba(18,48,47,.12);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; font-family: var(--sans); color: var(--ink); background-color: var(--paper);
  background-image: radial-gradient(var(--line) 1px, transparent 1px); background-size: 22px 22px;
  -webkit-font-smoothing: antialiased;
}
button, input { font-family: inherit; }

/* ---------- Header ---------- */
.header { position: sticky; top: 0; z-index: 50; background: rgba(255,255,255,.88); backdrop-filter: blur(12px); border-bottom: 1px solid var(--line); }
.header-in { max-width: 1280px; margin: 0 auto; padding: 0 32px; height: 70px; display: flex; align-items: center; gap: 36px; }
.brand { display: flex; align-items: center; gap: 12px; }
.logo { width: 38px; height: 38px; border-radius: 11px; background: linear-gradient(145deg, var(--teal-2), var(--teal)); display: grid; place-items: center; box-shadow: 0 6px 14px -4px rgba(15,124,120,.5); }
.logo svg { width: 20px; height: 20px; }
.brand b { font-family: var(--serif); font-size: 1.28rem; font-weight: 600; letter-spacing: -.01em; }
.brand b i { font-style: normal; color: var(--teal); }
.brand small { display: block; font-size: .64rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-top: -2px; }
.tabs { display: flex; gap: 4px; height: 100%; }
.tabs a { display: flex; align-items: center; gap: 9px; padding: 0 16px; height: 100%; font-size: .88rem; font-weight: 600; color: var(--ink-2); cursor: pointer; position: relative; user-select: none; white-space: nowrap; transition: color .2s; }
.tabs a i { color: var(--muted); font-size: .85rem; transition: color .2s; }
.tabs a:hover, .tabs a.active { color: var(--teal); }
.tabs a:hover i, .tabs a.active i { color: var(--teal); }
.tabs a.active::after { content: ""; position: absolute; left: 14px; right: 14px; bottom: -1px; height: 3px; border-radius: 3px 3px 0 0; background: var(--teal); }
.tabs .count { background: var(--teal); color: #fff; font-size: .66rem; padding: 1px 7px; border-radius: 99px; display: none; }
.spacer { flex: 1; }
.status { display: flex; align-items: center; gap: 9px; font-size: .78rem; font-weight: 600; color: var(--ink-2); background: var(--ok-bg); padding: 7px 14px; border-radius: 99px; white-space: nowrap; }
.pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); position: relative; }
.pulse::after { content: ""; position: absolute; inset: -4px; border-radius: 50%; border: 2px solid var(--ok); animation: ping 1.8s infinite; }
@keyframes ping { 0% { transform: scale(.5); opacity: .8; } 100% { transform: scale(1.5); opacity: 0; } }

.wrap { max-width: 1280px; margin: 0 auto; padding: 30px 32px 70px; }

/* ---------- Hero ---------- */
.hero { position: relative; overflow: hidden; border-radius: 26px; border: 1px solid var(--line); padding: 44px 48px; margin-bottom: 30px;
  background: linear-gradient(115deg, #ffffff 0%, var(--mint) 55%, var(--sky) 100%); box-shadow: var(--shadow); display: flex; justify-content: space-between; align-items: flex-end; gap: 30px; flex-wrap: wrap; }
.hero .ecg { position: absolute; left: 0; right: 0; bottom: 18px; width: 100%; height: 90px; opacity: .55; pointer-events: none; }
.ecg path { fill: none; stroke: var(--teal-2); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 700; stroke-dashoffset: 700; animation: beat 4.5s linear infinite; }
@keyframes beat { to { stroke-dashoffset: -700; } }
.hero-l { position: relative; z-index: 1; max-width: 600px; }
.eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: .7rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 700; color: var(--teal); background: rgba(255,255,255,.75); border: 1px solid var(--line-2); padding: 6px 12px; border-radius: 99px; margin-bottom: 18px; }
.hero h1 { font-family: var(--serif); font-weight: 600; font-size: 2.6rem; line-height: 1.12; letter-spacing: -.02em; margin: 0 0 14px; }
.hero h1 em { font-style: italic; color: var(--teal); font-weight: 500; }
.hero p { margin: 0; color: var(--ink-2); line-height: 1.65; font-size: .98rem; max-width: 520px; }
.kpis { position: relative; z-index: 1; display: flex; gap: 14px; }
.kpi { background: rgba(255,255,255,.85); border: 1px solid var(--line); border-radius: 18px; padding: 16px 22px; min-width: 140px; box-shadow: 0 6px 18px -10px rgba(18,48,47,.2); }
.kpi strong { display: block; font-family: var(--serif); font-size: 1.9rem; font-weight: 600; color: var(--teal); letter-spacing: -.02em; line-height: 1.1; }
.kpi span { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 700; }

.tab { display: none; }
.tab.active { display: block; animation: rise .4s ease; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

.page-head { margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
.page-head h2 { font-family: var(--serif); font-weight: 600; font-size: 1.7rem; letter-spacing: -.015em; margin: 0 0 4px; }
.page-head p { margin: 0; color: var(--muted); font-size: .9rem; }

.card { background: var(--white); border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow); }
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px; border: 1px solid var(--line-2); background: var(--white); color: var(--ink-2); padding: 10px 16px; border-radius: 12px; font-weight: 600; font-size: .83rem; cursor: pointer; transition: .18s; }
.btn:hover { border-color: var(--teal); color: var(--teal); transform: translateY(-1px); }
.btn-ok { color: var(--ok); background: var(--ok-bg); border-color: transparent; }
.btn-bad { color: var(--bad); background: var(--bad-bg); border-color: transparent; }
.btn-ok:hover { color: var(--ok); border-color: var(--ok); } .btn-bad:hover { color: var(--bad); border-color: var(--bad); }
.btn-primary { width: 100%; border: 0; color: #fff; padding: 16px; font-size: .95rem; border-radius: 14px; background: linear-gradient(135deg, var(--teal-2), var(--teal)); box-shadow: 0 14px 26px -12px rgba(15,124,120,.65); letter-spacing: .01em; }
.btn-primary:hover { color: #fff; transform: translateY(-2px); box-shadow: 0 18px 30px -12px rgba(15,124,120,.75); }
.btn-primary:disabled { opacity: .75; cursor: wait; transform: none; }

/* ---------- Predict layout ---------- */
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 24px; align-items: start; }
.aside { position: sticky; top: 94px; display: grid; gap: 16px; }
.aside .card { padding: 22px; }
.aside h4 { margin: 0 0 4px; font-family: var(--serif); font-size: 1.08rem; font-weight: 600; }
.aside .sub { color: var(--muted); font-size: .76rem; margin: 0 0 14px; line-height: 1.5; }

.cell-box { position: relative; border-radius: 16px; overflow: hidden; background: radial-gradient(circle at 50% 50%, #fff7fb 0%, #fdeef6 55%, #f8e1ee 100%); border: 1px solid #f3d3e3; }
.cell-box svg { display: block; width: 100%; height: auto; }
#cellShape { transition: d .35s ease; }
.cell-meta { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.cell-meta div { background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; }
.cell-meta small { display: block; font-size: .64rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 700; }
.cell-meta b { font-family: var(--mono); font-size: .86rem; font-weight: 500; }
.legend-note { display: flex; align-items: center; gap: 8px; font-size: .7rem; color: var(--muted); margin-top: 10px; }
.legend-note i { width: 18px; border-top: 2px dashed var(--blue); }

.fill-top { display: flex; justify-content: space-between; font-size: .78rem; font-weight: 700; margin-bottom: 8px; }
.fill-top span:last-child { font-family: var(--mono); color: var(--teal); }
.track { height: 8px; border-radius: 99px; background: var(--mint); overflow: hidden; }
.track i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--teal-2), #5dd3c8); border-radius: 99px; transition: width .4s; }
.sample-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 16px 0 8px; }
.aside small.disc { display: block; text-align: center; color: var(--muted); font-size: .72rem; line-height: 1.5; margin-top: 12px; }

.group { padding: 26px 28px 28px; margin-bottom: 20px; position: relative; overflow: hidden; }
.group::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c); }
.g-mean { --c: var(--teal); } .g-err { --c: var(--blue); } .g-worst { --c: var(--warn); }
.group-head { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.group-num { font-family: var(--serif); font-size: 2.3rem; font-weight: 500; font-style: italic; color: var(--c); opacity: .8; line-height: 1; }
.group-head h3 { margin: 0; font-family: var(--serif); font-size: 1.15rem; font-weight: 600; }
.group-head small { color: var(--muted); font-size: .78rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }
.field { border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; background: var(--paper); transition: .18s; }
.field:hover { border-color: var(--line-2); background: #fff; }
.field:focus-within { border-color: var(--c); background: #fff; box-shadow: 0 0 0 4px color-mix(in srgb, var(--c) 14%, transparent); }
.field.invalid { border-color: var(--bad); box-shadow: 0 0 0 4px rgba(207,69,69,.12); }
.field label { display: block; font-size: .8rem; font-weight: 700; color: var(--ink); }
.field .hint { color: var(--muted); font-size: .7rem; line-height: 1.4; min-height: 2.1em; margin: 2px 0 6px; }
.field input { width: 100%; border: 0; outline: 0; background: transparent; color: var(--ink); font-family: var(--mono); font-weight: 500; font-size: .98rem; padding: 2px 0; }

/* ---------- Report ---------- */
.report { display: none; margin-top: 4px; padding: 0; overflow: hidden; }
.report.show { display: block; animation: rise .5s ease; }
.report-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 22px 30px; background: linear-gradient(100deg, var(--mint), #fff); border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.report-head .l { display: flex; align-items: center; gap: 14px; }
.report-head h3 { margin: 0; font-family: var(--serif); font-size: 1.2rem; font-weight: 600; }
.report-head small { color: var(--muted); font-size: .76rem; }
.chip { font-family: var(--mono); font-size: .76rem; padding: 6px 12px; border-radius: 99px; background: #fff; border: 1px solid var(--line-2); color: var(--ink-2); }
.report-body { padding: 28px 30px 30px; }
.banner { display: flex; gap: 12px; padding: 14px 18px; border-radius: 14px; font-weight: 600; font-size: .9rem; line-height: 1.55; margin-bottom: 26px; }
.banner.ok { background: var(--ok-bg); color: #1f7a52; } .banner.bad { background: var(--bad-bg); color: #a93232; }
.report-grid { display: grid; grid-template-columns: 300px 1fr; gap: 36px; align-items: center; }
.meter { text-align: center; }
.meter svg { width: 100%; max-width: 300px; overflow: visible; }
.needle { transform-origin: 100px 100px; transform-box: view-box; transform: rotate(-90deg); transition: transform 1.3s cubic-bezier(.25,1.25,.45,1); }
.meter-val { font-family: var(--serif); font-size: 2.3rem; font-weight: 600; letter-spacing: -.02em; line-height: 1; margin-top: -6px; }
.meter-lbl { font-size: .7rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; color: var(--muted); margin-top: 6px; }
.verdict small { font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; color: var(--muted); }
.verdict h2 { font-family: var(--serif); font-weight: 600; font-size: 2.3rem; letter-spacing: -.02em; margin: 6px 0 20px; display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.verdict h2 span { font-size: .95rem; font-family: var(--sans); font-weight: 500; color: var(--muted); }
.prob { margin-bottom: 16px; }
.prob-row { display: flex; justify-content: space-between; font-size: .86rem; font-weight: 600; margin-bottom: 7px; color: var(--ink-2); }
.prob-row b { font-family: var(--mono); font-weight: 500; font-size: .95rem; }
.bar { height: 10px; border-radius: 99px; background: var(--paper); border: 1px solid var(--line); overflow: hidden; }
.bar i { display: block; height: 100%; width: 0; border-radius: 99px; transition: width 1.1s cubic-bezier(.4,0,.2,1); }
.note { display: flex; gap: 10px; font-size: .79rem; color: var(--ink-2); padding: 14px 16px; border-radius: 12px; background: var(--paper); border: 1px dashed var(--line-2); line-height: 1.55; margin-top: 6px; }

/* ---------- Tables / stats / guide ---------- */
.table-wrap { overflow-x: auto; padding: 8px 12px 12px; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 14px 16px; font-size: .68rem; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); font-weight: 700; border-bottom: 1px solid var(--line); }
td { padding: 14px 16px; border-bottom: 1px solid var(--line); font-size: .88rem; font-weight: 500; }
tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: var(--paper); }
.mono { font-family: var(--mono); }
.badge { padding: 5px 12px; border-radius: 99px; font-size: .74rem; font-weight: 700; }
.badge.ok { background: var(--ok-bg); color: var(--ok); } .badge.bad { background: var(--bad-bg); color: var(--bad); }
.empty { text-align: center; padding: 60px 16px !important; color: var(--muted); }
.empty i { display: block; font-size: 2.2rem; margin-bottom: 12px; opacity: .45; }

.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }
.stat { padding: 24px 26px; position: relative; overflow: hidden; }
.stat span { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); }
.stat h3 { margin: 8px 0 0; font-family: var(--serif); font-size: 3rem; font-weight: 600; color: var(--c); line-height: 1; }
.stat i { position: absolute; right: 20px; top: 20px; font-size: 1.7rem; color: var(--c); opacity: .22; }
.donut-card { padding: 30px; display: flex; gap: 40px; align-items: center; justify-content: center; flex-wrap: wrap; }
.donut { width: 200px; height: 200px; border-radius: 50%; display: grid; place-items: center; background: conic-gradient(var(--line) 0 100%); transition: background .6s; }
.donut::after { content: attr(data-label); width: 136px; height: 136px; border-radius: 50%; background: #fff; display: grid; place-items: center; font-family: var(--serif); font-weight: 600; font-size: 1.15rem; text-align: center; white-space: pre-line; color: var(--ink-2); box-shadow: inset 0 0 0 1px var(--line); }
.legend { display: grid; gap: 12px; font-weight: 600; font-size: .92rem; }
.legend div { display: flex; align-items: center; gap: 10px; }
.dot { width: 12px; height: 12px; border-radius: 4px; }

.search { position: relative; width: 320px; max-width: 100%; }
.search i { position: absolute; left: 14px; top: 50%; transform: translateY(-50%); color: var(--muted); font-size: .85rem; }
.search input { width: 100%; padding: 12px 14px 12px 38px; border-radius: 12px; border: 1px solid var(--line-2); background: #fff; color: var(--ink); outline: 0; font-size: .88rem; }
.search input:focus { border-color: var(--teal); box-shadow: 0 0 0 4px rgba(15,124,120,.12); }
.guide-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 12px; padding: 22px; }
.guide-item { padding: 16px 18px; border: 1px solid var(--line); border-radius: 14px; background: var(--paper); transition: .18s; }
.guide-item:hover { border-color: var(--teal-2); background: #fff; transform: translateY(-2px); }
.guide-item h4 { margin: 0 0 6px; font-size: .92rem; font-weight: 700; display: flex; gap: 10px; align-items: center; }
.guide-item h4 em { font-style: normal; font-family: var(--mono); font-size: .68rem; background: var(--mint); color: var(--teal); padding: 2px 8px; border-radius: 8px; }
.guide-item code { display: inline-block; font-family: var(--mono); font-size: .72rem; color: var(--muted); margin-bottom: 6px; }
.guide-item p { margin: 0; font-size: .82rem; color: var(--ink-2); line-height: 1.5; }

.toasts { position: fixed; right: 22px; bottom: 22px; display: grid; gap: 10px; z-index: 200; }
.toast { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--bad); padding: 14px 18px; border-radius: 12px; box-shadow: 0 14px 34px -10px rgba(18,48,47,.25); font-size: .86rem; font-weight: 600; max-width: 340px; animation: rise .3s; }
.toast.ok { border-left-color: var(--ok); }

@media (max-width: 1080px) {
  .layout { grid-template-columns: 1fr; }
  .aside { position: static; order: -1; grid-template-columns: 1fr 1fr; }
}
@media (max-width: 820px) {
  .header-in { padding: 0 16px; gap: 14px; height: 64px; }
  .tabs { overflow-x: auto; scrollbar-width: none; } .tabs a { padding: 0 12px; } .tabs a span.t { display: none; }
  .status { display: none; }
  .wrap { padding: 20px 16px 60px; }
  .hero { padding: 30px 24px 44px; } .hero h1 { font-size: 1.9rem; }
  .aside { grid-template-columns: 1fr; }
  .report-grid { grid-template-columns: 1fr; gap: 20px; }
  .stat-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<header class="header">
  <div class="header-in">
    <div class="brand">
      <div class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round"><path d="M12 4v16M4 12h16"/></svg></div>
      <div><b>Onco<i>Care</i></b><small>Clinical Decision Support</small></div>
    </div>
    <nav class="tabs">
      <a class="active" data-tab="tab-predict"><i class="fa-solid fa-stethoscope"></i><span class="t">Chẩn Đoán AI</span></a>
      <a data-tab="tab-history"><i class="fa-solid fa-clock-rotate-left"></i><span class="t">Nhật Ký</span><span class="count" id="navCount">0</span></a>
      <a data-tab="tab-stats"><i class="fa-solid fa-chart-pie"></i><span class="t">Thống Kê</span></a>
      <a data-tab="tab-guide"><i class="fa-solid fa-book-medical"></i><span class="t">Hướng Dẫn Chỉ Số</span></a>
    </nav>
    <div class="spacer"></div>
    <div class="status"><span class="pulse"></span><span id="verText">Hệ thống sẵn sàng</span></div>
  </div>
</header>

<main class="wrap">
  <section class="hero">
    <svg class="ecg" viewBox="0 0 1200 90" preserveAspectRatio="none"><path d="M0 50 H180 L200 50 L215 20 L232 80 L248 34 L262 50 H520 L540 50 L555 14 L574 84 L592 30 L606 50 H900 L920 50 L935 20 L952 80 L968 34 L982 50 H1200"/></svg>
    <div class="hero-l">
      <span class="eyebrow"><i class="fa-solid fa-microscope"></i> Hỗ trợ chẩn đoán lâm sàng</span>
      <h1>Phân tích khối u vú<br>bằng <em>học máy</em> SVM</h1>
      <p>Phân loại lành tính / ác tính từ 30 đặc trưng tế bào học của bộ dữ liệu Wisconsin Breast Cancer, trình bày rõ ràng như một bản báo cáo xét nghiệm.</p>
    </div>
    <div class="kpis">
      <div class="kpi"><strong>98.24%</strong><span>Độ chính xác</span></div>
      <div class="kpi"><strong>30</strong><span>Đặc trưng</span></div>
    </div>
  </section>

  <!-- TAB: PREDICT -->
  <div id="tab-predict" class="tab active">
    <div class="page-head">
      <div><h2>Phiếu nhập dữ liệu tế bào học</h2><p>Điền các thông số, hoặc nạp nhanh một mẫu chuẩn ở khung bên phải.</p></div>
    </div>

    <div class="layout">
      <form id="predictionForm" novalidate>
        <div class="card group g-mean">
          <div class="group-head"><span class="group-num">01</span><div><h3>Nhóm chỉ số trung bình</h3><small>Mean features · 10 chiều</small></div></div>
          <div class="grid" id="groupMean"></div>
        </div>
        <div class="card group g-err">
          <div class="group-head"><span class="group-num">02</span><div><h3>Nhóm sai số tiêu chuẩn</h3><small>Standard error · 10 chiều</small></div></div>
          <div class="grid" id="groupError"></div>
        </div>
        <div class="card group g-worst">
          <div class="group-head"><span class="group-num">03</span><div><h3>Nhóm giá trị cực đoan</h3><small>Worst features · 10 chiều</small></div></div>
          <div class="grid" id="groupWorst"></div>
        </div>
      </form>

      <aside class="aside">
        <div class="card">
          <h4>Mô phỏng nhân tế bào</h4>
          <p class="sub">Hình minh họa thay đổi theo bán kính, độ lõm và độ nhẵn bạn nhập. Không phải ảnh thực.</p>
          <div class="cell-box">
            <svg viewBox="0 0 260 260">
              <defs>
                <radialGradient id="nucFill" cx="50%" cy="45%" r="60%"><stop offset="0" stop-color="#b9a0e0" stop-opacity=".75"/><stop offset="1" stop-color="#7a58b8" stop-opacity=".85"/></radialGradient>
              </defs>
              <circle id="refCircle" cx="130" cy="130" r="41" fill="none" stroke="#3b78b8" stroke-width="1.5" stroke-dasharray="5 5"/>
              <path id="cellShape" d="" fill="url(#nucFill)" stroke="#5b3e96" stroke-width="2.2" stroke-linejoin="round"/>
              <circle id="nuc1" cx="122" cy="122" r="5" fill="#4a2f80" opacity=".55"/>
              <circle id="nuc2" cx="138" cy="136" r="3.5" fill="#4a2f80" opacity=".45"/>
            </svg>
          </div>
          <div class="cell-meta">
            <div><small>Bán kính TB</small><b id="mR">—</b></div>
            <div><small>Độ lõm TB</small><b id="mC">—</b></div>
          </div>
          <div class="legend-note"><i></i> Tham chiếu: bán kính TB nhóm lành tính (≈ 12.1)</div>
        </div>

        <div class="card">
          <div class="fill-top"><span>Tiến độ nhập liệu</span><span id="fillText">0/30</span></div>
          <div class="track"><i id="fillTrack"></i></div>
          <div class="sample-row">
            <button type="button" class="btn btn-ok" onclick="loadSampleData('benign')"><i class="fa-solid fa-shield-halved"></i> Mẫu lành tính</button>
            <button type="button" class="btn btn-bad" onclick="loadSampleData('malignant')"><i class="fa-solid fa-biohazard"></i> Mẫu ác tính</button>
          </div>
          <button type="button" class="btn" style="width:100%;margin-bottom:14px" onclick="clearForm()"><i class="fa-solid fa-eraser"></i> Xóa toàn bộ</button>
          <button type="submit" form="predictionForm" id="btnSubmit" class="btn btn-primary"><i class="fa-solid fa-wand-magic-sparkles"></i> Thực hiện chẩn đoán AI</button>
          <small class="disc">Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán giải phẫu bệnh.</small>
        </div>
      </aside>
    </div>

    <div class="card report" id="resultCard" style="margin-top:24px">
      <div class="report-head">
        <div class="l">
          <div class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round"><path d="M12 4v16M4 12h16"/></svg></div>
          <div><h3>Báo cáo phân tích tế bào học</h3><small id="resDate">—</small></div>
        </div>
        <span class="chip" id="resId">#CA-0000</span>
      </div>
      <div class="report-body">
        <div class="banner" id="resultAlert"></div>
        <div class="report-grid">
          <div class="meter">
            <svg viewBox="0 0 200 118">
              <defs><linearGradient id="zone" x1="20" y1="0" x2="180" y2="0" gradientUnits="userSpaceOnUse">
                <stop offset="0" stop-color="#2b9a6a"/><stop offset=".5" stop-color="#e8b23c"/><stop offset="1" stop-color="#cf4545"/></linearGradient></defs>
              <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#eef3f3" stroke-width="18" stroke-linecap="round"/>
              <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#zone)" stroke-width="12" stroke-linecap="round"/>
              <g class="needle" id="needle"><path d="M100 100 L95.5 100 L100 34 L104.5 100 Z" fill="#12302f"/></g>
              <circle cx="100" cy="100" r="8" fill="#12302f"/><circle cx="100" cy="100" r="3" fill="#fff"/>
              <text x="14" y="116" font-size="8" fill="#7c9291" font-family="DM Sans" font-weight="600">THẤP</text>
              <text x="186" y="116" font-size="8" fill="#7c9291" font-family="DM Sans" font-weight="600" text-anchor="end">CAO</text>
            </svg>
            <div class="meter-val" id="gaugeVal">0%</div>
            <div class="meter-lbl">Xác suất ác tính</div>
          </div>
          <div>
            <div class="verdict"><small>Kết luận phân loại của mô hình</small><h2 id="resPrediction"></h2></div>
            <div class="prob">
              <div class="prob-row"><span>Lành tính (Benign)</span><b id="resBenignText" style="color:var(--ok)">0%</b></div>
              <div class="bar"><i id="barBenign" style="background:var(--ok)"></i></div>
            </div>
            <div class="prob">
              <div class="prob-row"><span>Ác tính (Malignant)</span><b id="resMalignantText" style="color:var(--bad)">0%</b></div>
              <div class="bar"><i id="barMalignant" style="background:var(--bad)"></i></div>
            </div>
            <div class="note"><i class="fa-solid fa-circle-info" style="color:var(--teal);margin-top:3px"></i><span id="warnText">Kết quả từ thuật toán SVM chỉ hỗ trợ tham khảo lâm sàng.</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB: HISTORY -->
  <div id="tab-history" class="tab">
    <div class="page-head"><div><h2>Nhật ký khám bệnh</h2><p>Các ca phân tích trong phiên làm việc hiện tại.</p></div>
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
    <div class="page-head"><div><h2>Thống kê phiên làm việc</h2><p>Tổng quan phân phối kết quả chẩn đoán.</p></div></div>
    <div class="stat-grid">
      <div class="card stat" style="--c:var(--teal)"><i class="fa-solid fa-vial"></i><span>Tổng số ca</span><h3 id="statTotal">0</h3></div>
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
    <div class="page-head"><div><h2>Tra cứu 30 chỉ số lâm sàng</h2><p>Ý nghĩa kỹ thuật của các đặc trưng trích xuất tế bào học.</p></div>
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

/* ---------- Tabs ---------- */
function switchTab(id) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tabs a').forEach(el => el.classList.toggle('active', el.dataset.tab === id));
  $(id).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
document.querySelectorAll('.tabs a').forEach(a => a.onclick = () => switchTab(a.dataset.tab));

function toast(msg, ok) {
  const t = document.createElement('div');
  t.className = 'toast' + (ok ? ' ok' : '');
  t.textContent = msg;
  $('toasts').appendChild(t);
  setTimeout(() => t.remove(), 4200);
}

/* ---------- Init ---------- */
async function init() {
  try {
    const res = await fetch('/metadata');
    const data = await res.json();
    featureNames = data.feature_names;
    if (data.model_version) $('verText').textContent = 'Sẵn sàng · SVM v' + data.model_version;
    if (data.warning) $('warnText').textContent = data.warning;
  } catch (e) {
    toast('Không tải được metadata từ máy chủ.');
    return;
  }
  renderInputs(); renderGuide(); renderHistory(); renderStats(); updateFill(); drawCell();
}

function renderInputs() {
  const g = { mean: '', error: '', worst: '' };
  featureNames.forEach(f => {
    const info = featureMap[f] || { name: f, desc: 'Chỉ số tế bào' };
    const html = `<div class="field"><label for="${idOf(f)}">${info.name}</label><div class="hint">${info.desc}</div><input type="text" inputmode="decimal" autocomplete="off" name="${f}" id="${idOf(f)}" value="${sampleBenign[f] ?? ''}"></div>`;
    if (f.startsWith('mean')) g.mean += html;
    else if (f.includes('error')) g.error += html;
    else if (f.startsWith('worst')) g.worst += html;
  });
  $('groupMean').innerHTML = g.mean; $('groupError').innerHTML = g.error; $('groupWorst').innerHTML = g.worst;
  document.querySelectorAll('#predictionForm input').forEach(i => i.addEventListener('input', () => {
    i.parentElement.classList.remove('invalid'); updateFill(); drawCell();
  }));
}

function renderGuide(filter) {
  const q = (filter || '').toLowerCase().trim();
  let html = '', i = 0;
  Object.keys(featureMap).forEach(k => {
    i++;
    const info = featureMap[k];
    if (q && !(k.includes(q) || info.name.toLowerCase().includes(q) || info.desc.toLowerCase().includes(q))) return;
    html += `<div class="guide-item"><h4><em>${String(i).padStart(2, '0')}</em>${info.name}</h4><code>${k}</code><p>${info.desc}</p></div>`;
  });
  $('guideContainer').innerHTML = html || '<div class="empty" style="grid-column:1/-1">Không tìm thấy chỉ số phù hợp.</div>';
}
$('guideSearch').addEventListener('input', e => renderGuide(e.target.value));

function parseNum(v) { return parseFloat(String(v).trim().replace(',', '.')); }
function getNum(f) { const el = $(idOf(f)); if (!el) return NaN; return parseNum(el.value); }

function updateFill() {
  let n = 0;
  featureNames.forEach(f => { if (isFinite(getNum(f))) n++; });
  $('fillText').textContent = n + '/' + featureNames.length;
  $('fillTrack').style.width = (featureNames.length ? n / featureNames.length * 100 : 0) + '%';
}

/* ---------- Cell simulation ---------- */
function drawCell() {
  let R = getNum('mean radius'), C = getNum('mean concavity'), S = getNum('mean smoothness');
  const hasR = isFinite(R), hasC = isFinite(C);
  $('mR').textContent = hasR ? R.toFixed(2) : '—';
  $('mC').textContent = hasC ? C.toFixed(3) : '—';
  if (!hasR) R = 12; if (!hasC) C = 0.05; if (!isFinite(S)) S = 0.09;
  const scale = 3.4;
  const base = Math.max(5, Math.min(34, R)) * scale;
  const irr = Math.max(0, Math.min(C / 0.35, 1.3));
  const rough = Math.max(0, Math.min((S - 0.06) / 0.1, 1.2));
  const N = 140; let d = '';
  for (let i = 0; i <= N; i++) {
    const t = i / N * Math.PI * 2;
    const k = 1 + irr * 0.14 * Math.sin(5 * t + 0.8) + irr * 0.09 * Math.sin(9 * t + 2.1) + irr * 0.05 * Math.sin(14 * t) + rough * 0.025 * Math.sin(27 * t);
    const r = Math.min(base * k, 126);
    d += (i ? 'L' : 'M') + (130 + r * Math.cos(t)).toFixed(1) + ' ' + (130 + r * Math.sin(t)).toFixed(1);
  }
  $('cellShape').setAttribute('d', d + 'Z');
  const s = base / 41;
  $('nuc1').setAttribute('r', (4 + irr * 3).toFixed(1));
  $('nuc1').setAttribute('cx', (130 - base * 0.12).toFixed(1)); $('nuc1').setAttribute('cy', (130 - base * 0.1).toFixed(1));
  $('nuc2').setAttribute('cx', (130 + base * 0.14).toFixed(1)); $('nuc2').setAttribute('cy', (130 + base * 0.12).toFixed(1));
}

function setValues(data) {
  featureNames.forEach(f => { const el = $(idOf(f)); if (el) { el.value = data[f] ?? ''; el.parentElement.classList.remove('invalid'); } });
  updateFill(); drawCell();
}
function loadSampleData(type) { setValues(type === 'malignant' ? sampleMalignant : sampleBenign); toast(type === 'malignant' ? 'Đã nạp mẫu ác tính' : 'Đã nạp mẫu lành tính', true); }
function clearForm() { setValues({}); $('resultCard').classList.remove('show'); }

/* ---------- History & stats ---------- */
function renderHistory() {
  const tb = $('historyTableBody'), nc = $('navCount');
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
  setValues(h.features); switchTab('tab-predict'); showResult(h.res, h.id, h.time);
}
function clearHistory() { historyList = []; stats = { total: 0, benign: 0, malignant: 0 }; renderHistory(); renderStats(); }
function renderStats() {
  $('statTotal').textContent = stats.total; $('statBenign').textContent = stats.benign; $('statMalignant').textContent = stats.malignant;
  const d = $('donut');
  if (!stats.total) { d.style.background = 'conic-gradient(var(--line) 0 100%)'; d.dataset.label = 'Chưa có\ndữ liệu'; $('lgB').textContent = '0%'; $('lgM').textContent = '0%'; return; }
  const b = stats.benign / stats.total * 100;
  d.style.background = `conic-gradient(var(--ok) 0 ${b}%, var(--bad) ${b}% 100%)`;
  d.dataset.label = stats.total + '\nca';
  $('lgB').textContent = b.toFixed(0) + '%'; $('lgM').textContent = (100 - b).toFixed(0) + '%';
}

/* ---------- Result ---------- */
function showResult(data, id, time) {
  const pB = data.probability_benign * 100, pM = data.probability_malignant * 100;
  const bad = data.predicted_label !== 'benign';
  $('resultCard').classList.add('show');
  $('resId').textContent = '#' + id;
  $('resDate').textContent = time || new Date().toLocaleString('vi-VN');
  $('resBenignText').textContent = pB.toFixed(2) + '%';
  $('resMalignantText').textContent = pM.toFixed(2) + '%';
  $('gaugeVal').textContent = pM.toFixed(1) + '%';
  $('gaugeVal').style.color = bad ? 'var(--bad)' : 'var(--ok)';
  $('needle').style.transform = 'rotate(-90deg)';
  $('barBenign').style.width = '0%'; $('barMalignant').style.width = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $('needle').style.transform = `rotate(${-90 + pM / 100 * 180}deg)`;
    $('barBenign').style.width = pB + '%'; $('barMalignant').style.width = pM + '%';
  }));
  const a = $('resultAlert');
  if (!bad) {
    $('resPrediction').innerHTML = '<b style="color:var(--ok)">Lành tính</b><span>Benign</span>';
    a.className = 'banner ok';
    a.innerHTML = '<i class="fa-solid fa-circle-check" style="margin-top:3px"></i><span>Mô hình xác định khối u có đặc trưng lành tính. Vẫn nên theo dõi định kỳ theo chỉ định của bác sĩ.</span>';
  } else {
    $('resPrediction').innerHTML = '<b style="color:var(--bad)">Ác tính</b><span>Malignant</span>';
    a.className = 'banner bad';
    a.innerHTML = '<i class="fa-solid fa-triangle-exclamation" style="margin-top:3px"></i><span>Khối u mang đặc trưng ác tính. Cần sinh thiết và hội chẩn chuyên khoa để xác nhận.</span>';
  }
  $('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ---------- Submit ---------- */
$('predictionForm').addEventListener('submit', async function (e) {
  e.preventDefault();
  const obj = {}, bad = [];
  featureNames.forEach(f => {
    const el = $(idOf(f)), v = parseNum(el.value);
    if (el.value.trim() === '' || !isFinite(v)) bad.push(el); else obj[f] = v;
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
    const time = new Date().toLocaleString('vi-VN');
    stats.total++; isMal ? stats.malignant++ : stats.benign++;
    historyList.unshift({ id, time, malignant: isMal, prob: (data.probability_malignant * 100).toFixed(2) + '%', features: obj, res: data });
    showResult(data, id, time); renderHistory(); renderStats();
  } catch (err) {
    toast(err.message || 'Không thể kết nối máy chủ.');
  } finally {
    btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Thực hiện chẩn đoán AI';
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
