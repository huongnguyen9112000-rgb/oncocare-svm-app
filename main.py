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

class BatchRequest(BaseModel):
    rows: list[dict[str, float]] = Field(..., description="List of feature dicts (max 1000)")

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

def _build_reference() -> dict:
    """Min/max and per-class means of the public Wisconsin dataset bundled with scikit-learn.
    Used only for UI hints (range warnings, radar chart). Never used for prediction."""
    try:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        X, y = d.data, d.target  # target: 0 = malignant, 1 = benign
        ref = {}
        for j, name in enumerate(d.feature_names):
            col = X[:, j]
            ref[str(name)] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "mean_malignant": float(col[y == 0].mean()),
                "mean_benign": float(col[y == 1].mean()),
            }
        return ref
    except Exception:
        return {}

reference = _build_reference()

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
  --paper: #fbfcfc; --white: #ffffff; --mint: #eaf6f4; --sky: #eaf2fb;
  --ink: #12302f; --ink-2: #3d5a59; --muted: #7c9291; --line: #e1ebea; --line-2: #d2e0df;
  --teal: #0f7c78; --teal-2: #14a39d; --blue: #3b78b8; --violet: #6a5acd;
  --ok: #2b9a6a; --ok-bg: #e6f6ee; --warn: #d58a1c; --warn-bg: #fdf3e1; --bad: #cf4545; --bad-bg: #fcebeb;
  --serif: 'Fraunces', Georgia, serif; --sans: 'DM Sans', system-ui, sans-serif; --mono: 'DM Mono', ui-monospace, monospace;
  --shadow: 0 1px 2px rgba(18,48,47,.04), 0 10px 30px -12px rgba(18,48,47,.12);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; font-family: var(--sans); color: var(--ink); background-color: var(--paper);
  background-image: radial-gradient(var(--line) 1px, transparent 1px); background-size: 22px 22px; -webkit-font-smoothing: antialiased; }
button, input, textarea { font-family: inherit; }

/* Header */
.header { position: sticky; top: 0; z-index: 50; background: rgba(255,255,255,.9); backdrop-filter: blur(12px); border-bottom: 1px solid var(--line); }
.header-in { max-width: 1280px; margin: 0 auto; padding: 0 32px; height: 70px; display: flex; align-items: center; gap: 30px; }
.brand { display: flex; align-items: center; gap: 12px; }
.logo { width: 38px; height: 38px; border-radius: 11px; background: linear-gradient(145deg, var(--teal-2), var(--teal)); display: grid; place-items: center; box-shadow: 0 6px 14px -4px rgba(15,124,120,.5); flex: none; }
.logo svg { width: 20px; height: 20px; }
.brand b { font-family: var(--serif); font-size: 1.28rem; font-weight: 600; letter-spacing: -.01em; }
.brand b i { font-style: normal; color: var(--teal); }
.brand small { display: block; font-size: .64rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-top: -2px; }
.tabs { display: flex; gap: 2px; height: 100%; }
.tabs a { display: flex; align-items: center; gap: 9px; padding: 0 14px; height: 100%; font-size: .87rem; font-weight: 600; color: var(--ink-2); cursor: pointer; position: relative; user-select: none; white-space: nowrap; }
.tabs a i { color: var(--muted); font-size: .85rem; }
.tabs a:hover, .tabs a.active, .tabs a:hover i, .tabs a.active i { color: var(--teal); }
.tabs a.active::after { content: ""; position: absolute; left: 12px; right: 12px; bottom: -1px; height: 3px; border-radius: 3px 3px 0 0; background: var(--teal); }
.tabs .count { background: var(--teal); color: #fff; font-size: .66rem; padding: 1px 7px; border-radius: 99px; display: none; }
.spacer { flex: 1; }
.lang { display: flex; background: var(--paper); border: 1px solid var(--line-2); border-radius: 10px; padding: 3px; }
.lang button { border: 0; background: transparent; padding: 5px 10px; border-radius: 7px; font-weight: 700; font-size: .74rem; color: var(--ink-2); cursor: pointer; }
.lang button.on { background: var(--teal); color: #fff; }
.status { display: flex; align-items: center; gap: 9px; font-size: .78rem; font-weight: 600; color: var(--ink-2); background: var(--ok-bg); padding: 7px 14px; border-radius: 99px; white-space: nowrap; }
.pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); position: relative; }
.pulse::after { content: ""; position: absolute; inset: -4px; border-radius: 50%; border: 2px solid var(--ok); animation: ping 1.8s infinite; }
@keyframes ping { 0% { transform: scale(.5); opacity: .8; } 100% { transform: scale(1.5); opacity: 0; } }
.wrap { max-width: 1280px; margin: 0 auto; padding: 30px 32px 70px; }

/* Hero */
.hero { position: relative; overflow: hidden; border-radius: 26px; border: 1px solid var(--line); padding: 44px 48px; margin-bottom: 30px;
  background: linear-gradient(115deg, #fff 0%, var(--mint) 55%, var(--sky) 100%); box-shadow: var(--shadow); display: flex; justify-content: space-between; align-items: flex-end; gap: 30px; flex-wrap: wrap; }
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
.page-head p { margin: 0; color: var(--muted); font-size: .9rem; max-width: 640px; line-height: 1.5; }
.toolbar { display: flex; gap: 10px; flex-wrap: wrap; }

.card { background: var(--white); border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow); }
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px; border: 1px solid var(--line-2); background: var(--white); color: var(--ink-2); padding: 10px 16px; border-radius: 12px; font-weight: 600; font-size: .83rem; cursor: pointer; transition: .18s; white-space: nowrap; }
.btn:hover:not(:disabled) { border-color: var(--teal); color: var(--teal); transform: translateY(-1px); }
.btn:disabled { opacity: .45; cursor: not-allowed; }
.btn-ok { color: var(--ok); background: var(--ok-bg); border-color: transparent; } .btn-bad { color: var(--bad); background: var(--bad-bg); border-color: transparent; }
.btn-ok:hover:not(:disabled) { color: var(--ok); border-color: var(--ok); } .btn-bad:hover:not(:disabled) { color: var(--bad); border-color: var(--bad); }
.btn-solid { background: var(--teal); color: #fff; border-color: var(--teal); } .btn-solid:hover:not(:disabled) { color: #fff; background: #0c6a67; }
.btn-primary { width: 100%; border: 0; color: #fff; padding: 16px; font-size: .95rem; border-radius: 14px; background: linear-gradient(135deg, var(--teal-2), var(--teal)); box-shadow: 0 14px 26px -12px rgba(15,124,120,.65); }
.btn-primary:hover:not(:disabled) { color: #fff; transform: translateY(-2px); box-shadow: 0 18px 30px -12px rgba(15,124,120,.75); }
.btn-primary:disabled { opacity: .75; cursor: wait; }
.seg { display: inline-flex; background: var(--paper); border: 1px solid var(--line-2); border-radius: 10px; padding: 3px; }
.seg button { border: 0; background: transparent; padding: 6px 12px; border-radius: 8px; font-weight: 600; font-size: .78rem; color: var(--ink-2); cursor: pointer; }
.seg button.on { background: var(--teal); color: #fff; }

/* Predict */
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 24px; align-items: start; }
.aside { position: sticky; top: 94px; display: grid; gap: 16px; }
.aside .card { padding: 22px; }
.aside h4 { margin: 0 0 4px; font-family: var(--serif); font-size: 1.08rem; font-weight: 600; }
.aside .sub { color: var(--muted); font-size: .76rem; margin: 0 0 14px; line-height: 1.5; }
.cell-box { border-radius: 16px; overflow: hidden; background: radial-gradient(circle at 50% 50%, #fff7fb 0%, #fdeef6 55%, #f8e1ee 100%); border: 1px solid #f3d3e3; }
.cell-box svg { display: block; width: 100%; height: auto; }
.cell-meta { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.cell-meta div { background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; }
.cell-meta small { display: block; font-size: .64rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 700; }
.cell-meta b { font-family: var(--mono); font-size: .86rem; font-weight: 500; }
.legend-note { display: flex; align-items: center; gap: 8px; font-size: .7rem; color: var(--muted); margin-top: 10px; }
.legend-note i { width: 18px; border-top: 2px dashed var(--blue); flex: none; }
.fill-top { display: flex; justify-content: space-between; font-size: .78rem; font-weight: 700; margin-bottom: 8px; }
.fill-top span:last-child { font-family: var(--mono); color: var(--teal); }
.track { height: 8px; border-radius: 99px; background: var(--mint); overflow: hidden; }
.track i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--teal-2), #5dd3c8); border-radius: 99px; transition: width .4s; }
.warnline { display: none; align-items: center; gap: 8px; margin-top: 12px; font-size: .76rem; font-weight: 600; color: #a96a0c; background: var(--warn-bg); padding: 8px 12px; border-radius: 10px; }
.sample-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 16px 0 8px; }
.aside small.disc { display: block; text-align: center; color: var(--muted); font-size: .72rem; line-height: 1.5; margin-top: 12px; }

.group { padding: 26px 28px 28px; margin-bottom: 20px; position: relative; overflow: hidden; }
.group::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c); }
.g-pat { --c: var(--violet); } .g-mean { --c: var(--teal); } .g-err { --c: var(--blue); } .g-worst { --c: var(--warn); }
.group-head { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.group-num { font-family: var(--serif); font-size: 2.3rem; font-weight: 500; font-style: italic; color: var(--c); opacity: .8; line-height: 1; min-width: 48px; }
.group-head h3 { margin: 0; font-family: var(--serif); font-size: 1.15rem; font-weight: 600; }
.group-head small { color: var(--muted); font-size: .78rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }
.field { border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; background: var(--paper); transition: .18s; }
.field:hover { border-color: var(--line-2); background: #fff; }
.field:focus-within { border-color: var(--c); background: #fff; box-shadow: 0 0 0 4px color-mix(in srgb, var(--c) 14%, transparent); }
.field.invalid { border-color: var(--bad); box-shadow: 0 0 0 4px rgba(207,69,69,.12); }
.field.warn { border-color: var(--warn); background: var(--warn-bg); }
.field label { display: block; font-size: .8rem; font-weight: 700; }
.field .hint { color: var(--muted); font-size: .7rem; line-height: 1.4; min-height: 2.1em; margin: 2px 0 6px; }
.field input { width: 100%; border: 0; outline: 0; background: transparent; color: var(--ink); font-family: var(--mono); font-weight: 500; font-size: .98rem; padding: 2px 0; }
.field .warnmsg { display: none; font-size: .68rem; color: #a96a0c; margin-top: 4px; line-height: 1.35; font-weight: 600; }
.field.warn .warnmsg { display: block; }
.pgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.pgrid .wide { grid-column: 1 / -1; }
.pgrid input { font-family: var(--sans); font-size: .92rem; }

/* Report */
.report { display: none; padding: 0; overflow: hidden; margin-top: 24px; }
.report.show { display: block; animation: rise .5s ease; }
.report-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 22px 30px; background: linear-gradient(100deg, var(--mint), #fff); border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.report-head .l { display: flex; align-items: center; gap: 14px; }
.report-head h3 { margin: 0; font-family: var(--serif); font-size: 1.2rem; font-weight: 600; }
.report-head small { color: var(--muted); font-size: .76rem; }
.report-head .r { display: flex; align-items: center; gap: 10px; }
.chip { font-family: var(--mono); font-size: .76rem; padding: 6px 12px; border-radius: 99px; background: #fff; border: 1px solid var(--line-2); color: var(--ink-2); }
.report-body { padding: 28px 30px 30px; }
.pblock { display: none; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 22px; padding: 16px 18px; border: 1px solid var(--line); border-radius: 14px; background: var(--paper); }
.pblock small { display: block; font-size: .64rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 700; margin-bottom: 2px; }
.pblock b { font-size: .92rem; font-weight: 600; word-break: break-word; }
.banner { display: flex; gap: 12px; padding: 14px 18px; border-radius: 14px; font-weight: 600; font-size: .9rem; line-height: 1.55; margin-bottom: 14px; }
.banner.ok { background: var(--ok-bg); color: #1f7a52; } .banner.bad { background: var(--bad-bg); color: #a93232; } .banner.warn { background: var(--warn-bg); color: #8a5608; }
.report-grid { display: grid; grid-template-columns: 300px 1fr; gap: 36px; align-items: center; margin-top: 12px; }
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
.radar-sec { margin-top: 28px; padding-top: 24px; border-top: 1px solid var(--line); }
.radar-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; flex-wrap: wrap; margin-bottom: 8px; }
.radar-top h4 { margin: 0 0 4px; font-family: var(--serif); font-size: 1.1rem; font-weight: 600; }
.radar-top p { margin: 0; font-size: .78rem; color: var(--muted); max-width: 560px; line-height: 1.5; }
.radar-box { display: flex; justify-content: center; }
.radar { width: 100%; max-width: 460px; height: auto; }
.rlegend { display: flex; gap: 18px; justify-content: center; flex-wrap: wrap; font-size: .78rem; font-weight: 600; color: var(--ink-2); }
.rlegend span { display: inline-flex; align-items: center; gap: 7px; }
.rlegend i { width: 12px; height: 12px; border-radius: 4px; display: inline-block; }
.rep-foot { margin-top: 22px; font-size: .72rem; color: var(--muted); text-align: center; }
.sign { display: none; grid-template-columns: 1fr 1fr; gap: 60px; margin-top: 50px; text-align: center; font-size: .8rem; color: var(--ink-2); }
.sign div { border-top: 1px solid #999; padding-top: 8px; }

/* Tables */
.table-wrap { overflow-x: auto; padding: 8px 12px 12px; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 14px 14px; font-size: .68rem; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); font-weight: 700; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 13px 14px; border-bottom: 1px solid var(--line); font-size: .87rem; font-weight: 500; }
tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: var(--paper); }
td small { display: block; color: var(--muted); font-size: .72rem; }
.mono { font-family: var(--mono); }
.badge { padding: 5px 12px; border-radius: 99px; font-size: .74rem; font-weight: 700; white-space: nowrap; }
.badge.ok { background: var(--ok-bg); color: var(--ok); } .badge.bad { background: var(--bad-bg); color: var(--bad); }
.empty { text-align: center; padding: 60px 16px !important; color: var(--muted); }
.empty i { display: block; font-size: 2.2rem; margin-bottom: 12px; opacity: .45; }
input[type=checkbox] { width: 17px; height: 17px; accent-color: var(--teal); cursor: pointer; }
.row-btn { border: 0; background: transparent; color: var(--muted); cursor: pointer; padding: 6px 8px; border-radius: 8px; }
.row-btn:hover { background: var(--bad-bg); color: var(--bad); }

/* Batch */
.drop { margin: 22px; border: 2px dashed var(--line-2); border-radius: 18px; padding: 38px 20px; text-align: center; background: var(--paper); transition: .2s; }
.drop.drag { border-color: var(--teal); background: var(--mint); }
.drop i.big { font-size: 2.2rem; color: var(--teal-2); margin-bottom: 12px; display: block; }
.drop p { margin: 0 0 14px; font-weight: 600; color: var(--ink-2); }
.drop .row { display: flex; gap: 10px; justify-content: center; align-items: center; flex-wrap: wrap; color: var(--muted); font-size: .85rem; }
.bstatus { padding: 0 22px 8px; font-size: .88rem; font-weight: 600; }
.bstatus .errs { margin-top: 8px; font-size: .8rem; color: #a93232; font-weight: 500; line-height: 1.6; }
.bactions { display: flex; gap: 10px; flex-wrap: wrap; padding: 6px 22px 22px; }
.chips { display: flex; gap: 10px; flex-wrap: wrap; padding: 0 22px 14px; }

/* Stats */
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }
.stat { padding: 24px 26px; position: relative; overflow: hidden; }
.stat span { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); }
.stat h3 { margin: 8px 0 0; font-family: var(--serif); font-size: 3rem; font-weight: 600; color: var(--c); line-height: 1; }
.stat i { position: absolute; right: 20px; top: 20px; font-size: 1.7rem; color: var(--c); opacity: .22; }
.donut-card { padding: 30px; display: flex; gap: 40px; align-items: center; justify-content: center; flex-wrap: wrap; margin-bottom: 16px; }
.donut { width: 200px; height: 200px; border-radius: 50%; display: grid; place-items: center; background: conic-gradient(var(--line) 0 100%); transition: background .6s; }
.donut::after { content: attr(data-label); width: 136px; height: 136px; border-radius: 50%; background: #fff; display: grid; place-items: center; font-family: var(--serif); font-weight: 600; font-size: 1.15rem; text-align: center; white-space: pre-line; color: var(--ink-2); box-shadow: inset 0 0 0 1px var(--line); }
.legend { display: grid; gap: 12px; font-weight: 600; font-size: .92rem; }
.legend div { display: flex; align-items: center; gap: 10px; }
.dot { width: 12px; height: 12px; border-radius: 4px; }
.trend-card { padding: 26px 28px; }
.trend-card h3 { margin: 0 0 4px; font-family: var(--serif); font-size: 1.15rem; font-weight: 600; }
.trend-card p { margin: 0 0 14px; font-size: .8rem; color: var(--muted); }
.trend-card svg { width: 100%; height: auto; display: block; }

/* Guide */
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

/* Modal & toast */
.modal { position: fixed; inset: 0; background: rgba(18,48,47,.45); display: none; place-items: center; z-index: 300; padding: 20px; }
.modal.show { display: grid; }
.modal-card { width: min(1000px, 100%); max-height: 92vh; overflow: auto; background: #fff; border-radius: 22px; padding: 26px 28px; box-shadow: 0 30px 80px -20px rgba(0,0,0,.4); }
.modal-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; gap: 12px; }
.modal-top h3 { margin: 0; font-family: var(--serif); font-size: 1.4rem; font-weight: 600; }
.cmp-heads { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 18px; }
.cmp-head { border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; border-top: 4px solid var(--c); }
.cmp-head b { font-family: var(--mono); font-size: .9rem; }
.cmp-head div { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.cmp-head small { color: var(--muted); font-size: .76rem; }
.cmp-tbl { max-height: 320px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; margin-top: 18px; }
.cmp-tbl th { position: sticky; top: 0; background: #fff; }
.diff-hi { color: #a96a0c; font-weight: 700; }
.toasts { position: fixed; right: 22px; bottom: 22px; display: grid; gap: 10px; z-index: 400; }
.toast { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--bad); padding: 14px 18px; border-radius: 12px; box-shadow: 0 14px 34px -10px rgba(18,48,47,.25); font-size: .86rem; font-weight: 600; max-width: 340px; animation: rise .3s; }
.toast.ok { border-left-color: var(--ok); }

@media (max-width: 1080px) { .layout { grid-template-columns: 1fr; } .aside { position: static; order: -1; grid-template-columns: 1fr 1fr; } }
@media (max-width: 860px) {
  .header-in { padding: 0 14px; gap: 10px; height: 64px; }
  .brand small, .brand b { display: none; }
  .tabs { overflow-x: auto; scrollbar-width: none; } .tabs a { padding: 0 11px; } .tabs a span.t { display: none; }
  .status { display: none; }
  .wrap { padding: 20px 16px 60px; }
  .hero { padding: 30px 24px 44px; } .hero h1 { font-size: 1.9rem; }
  .aside { grid-template-columns: 1fr; }
  .report-grid { grid-template-columns: 1fr; gap: 20px; }
  .stat-grid, .cmp-heads { grid-template-columns: 1fr; }
}

/* Lab-sheet style print layout (hidden on screen) */
#printSheet { display: none; }
.ps { color: #111; font-size: 10pt; line-height: 1.4; }
.ps-head { display: flex; justify-content: space-between; align-items: center; border-bottom: 2.5px solid #0f7c78; padding-bottom: 10px; }
.ps-brand { display: flex; align-items: center; gap: 10px; }
.ps-brand b { font-family: var(--serif); font-size: 17pt; font-weight: 600; }
.ps-brand b i { font-style: normal; color: #0f7c78; }
.ps-brand small { display: block; font-size: 7.5pt; letter-spacing: .14em; text-transform: uppercase; color: #666; }
.ps-meta { text-align: right; font-size: 9pt; color: #333; line-height: 1.6; }
.ps-title { text-align: center; margin: 14px 0 12px; }
.ps-title h1 { margin: 0; font-family: var(--serif); font-size: 15pt; letter-spacing: .04em; font-weight: 700; }
.ps-title p { margin: 3px 0 0; font-size: 9pt; color: #555; }
.ps table { width: 100%; border-collapse: collapse; }
.ps th, .ps td { border: 1px solid #b8c4c4; padding: 5px 8px; font-size: 9.5pt; text-align: left; letter-spacing: 0; text-transform: none; font-weight: 400; color: #111; white-space: normal; }
.ps tr:last-child td { border-bottom: 1px solid #b8c4c4; }
.ps .lab { background: #eef5f4; font-weight: 700; width: 17%; }
.ps-blank { display: block; border-bottom: 1px dotted #777; height: 1.1em; }
.ps-sec { margin: 14px 0 5px; font-weight: 700; font-size: 9.5pt; text-transform: uppercase; letter-spacing: .06em; color: #0f7c78; }
.ps-res th { background: #0f7c78; color: #fff; font-weight: 700; font-size: 9pt; text-align: center; }
.ps-res td.n { text-align: right; font-family: var(--mono); }
.ps-res td.c { text-align: center; width: 6%; }
.ps-note { font-size: 8pt; color: #555; margin: 5px 0 0; }
.ps-concl { margin-top: 14px; border: 2px solid var(--c); border-radius: 8px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; gap: 16px; }
.ps-concl.ok { --c: #2b9a6a; } .ps-concl.bad { --c: #cf4545; }
.ps-concl small { display: block; font-size: 8pt; letter-spacing: .1em; text-transform: uppercase; color: #555; font-weight: 700; }
.ps-concl b { font-family: var(--serif); font-size: 18pt; color: var(--c); }
.ps-concl .pr { text-align: right; font-size: 9.5pt; line-height: 1.7; }
.ps-advice { margin: 8px 0 0; font-size: 9.5pt; }
.ps-warn { margin: 6px 0 0; font-size: 9pt; color: #8a5608; }
.ps-sign { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 18px; text-align: center; font-size: 9.5pt; }
.ps-sign div { height: 85px; }
.ps-sign small { display: block; color: #666; font-size: 8pt; }
.ps-foot { margin-top: 10px; border-top: 1px solid #ccc; font-size: 8pt; color: #666; padding-top: 6px; text-align: center; }
@media print {
  @page { size: A4; margin: 12mm 14mm; }
  body { background: #fff !important; }
  .header, .hero, .page-head, .layout, .toasts, .modal, .no-print, .report, #tab-history, #tab-batch, #tab-stats, #tab-guide { display: none !important; }
  #tab-predict { display: block !important; animation: none !important; }
  .wrap { padding: 0; max-width: none; }
  #printSheet { display: block !important; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
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
      <a class="active" data-tab="tab-predict"><i class="fa-solid fa-stethoscope"></i><span class="t" data-i18n="nav_predict"></span></a>
      <a data-tab="tab-history"><i class="fa-solid fa-clock-rotate-left"></i><span class="t" data-i18n="nav_history"></span><span class="count" id="navCount">0</span></a>
      <a data-tab="tab-batch"><i class="fa-solid fa-file-csv"></i><span class="t" data-i18n="nav_batch"></span></a>
      <a data-tab="tab-stats"><i class="fa-solid fa-chart-pie"></i><span class="t" data-i18n="nav_stats"></span></a>
      <a data-tab="tab-guide"><i class="fa-solid fa-book-medical"></i><span class="t" data-i18n="nav_guide"></span></a>
    </nav>
    <div class="spacer"></div>
    <div class="lang"><button data-lang="vi">VI</button><button data-lang="en">EN</button></div>
    <div class="status"><span class="pulse"></span><span id="verText">…</span></div>
  </div>
</header>

<main class="wrap">
  <section class="hero">
    <svg class="ecg" viewBox="0 0 1200 90" preserveAspectRatio="none"><path d="M0 50 H180 L200 50 L215 20 L232 80 L248 34 L262 50 H520 L540 50 L555 14 L574 84 L592 30 L606 50 H900 L920 50 L935 20 L952 80 L968 34 L982 50 H1200"/></svg>
    <div class="hero-l">
      <span class="eyebrow"><i class="fa-solid fa-microscope"></i> <span data-i18n="eyebrow"></span></span>
      <h1 data-i18n-html="hero_h1"></h1>
      <p data-i18n="hero_p"></p>
    </div>
    <div class="kpis">
      <div class="kpi"><strong>98.24%</strong><span data-i18n="kpi_acc"></span></div>
      <div class="kpi"><strong>30</strong><span data-i18n="kpi_feat"></span></div>
    </div>
  </section>

  <!-- PREDICT -->
  <div id="tab-predict" class="tab active">
    <div class="page-head"><div><h2 data-i18n="pred_title"></h2><p data-i18n="pred_sub"></p></div></div>

    <div class="layout">
      <form id="predictionForm" novalidate>
        <div class="card group g-pat">
          <div class="group-head"><span class="group-num"><i class="fa-regular fa-id-card" style="font-size:1.6rem;font-style:normal"></i></span>
            <div><h3 data-i18n="pat_title"></h3><small data-i18n="pat_sub"></small></div></div>
          <div class="pgrid">
            <div class="field"><label for="pName" data-i18n="pat_name"></label><input id="pName" type="text" autocomplete="off"></div>
            <div class="field"><label for="pAge" data-i18n="pat_age"></label><input id="pAge" type="number" min="0" max="120" inputmode="numeric"></div>
            <div class="field"><label for="pId" data-i18n="pat_id"></label><input id="pId" type="text" autocomplete="off"></div>
            <div class="field"><label for="pDoc" data-i18n="pat_doc"></label><input id="pDoc" type="text" autocomplete="off"></div>
            <div class="field wide"><label for="pNote" data-i18n="pat_note"></label><input id="pNote" type="text" autocomplete="off"></div>
          </div>
        </div>
        <div class="card group g-mean">
          <div class="group-head"><span class="group-num">01</span><div><h3 data-i18n="g1_t"></h3><small data-i18n="g1_s"></small></div></div>
          <div class="grid" id="groupMean"></div>
        </div>
        <div class="card group g-err">
          <div class="group-head"><span class="group-num">02</span><div><h3 data-i18n="g2_t"></h3><small data-i18n="g2_s"></small></div></div>
          <div class="grid" id="groupError"></div>
        </div>
        <div class="card group g-worst">
          <div class="group-head"><span class="group-num">03</span><div><h3 data-i18n="g3_t"></h3><small data-i18n="g3_s"></small></div></div>
          <div class="grid" id="groupWorst"></div>
        </div>
      </form>

      <aside class="aside">
        <div class="card">
          <h4 data-i18n="sim_title"></h4>
          <p class="sub" data-i18n="sim_sub"></p>
          <div class="cell-box">
            <svg viewBox="0 0 260 260">
              <defs><radialGradient id="nucFill" cx="50%" cy="45%" r="60%"><stop offset="0" stop-color="#b9a0e0" stop-opacity=".75"/><stop offset="1" stop-color="#7a58b8" stop-opacity=".85"/></radialGradient></defs>
              <circle cx="130" cy="130" r="41" fill="none" stroke="#3b78b8" stroke-width="1.5" stroke-dasharray="5 5"/>
              <path id="cellShape" d="" fill="url(#nucFill)" stroke="#5b3e96" stroke-width="2.2" stroke-linejoin="round"/>
              <circle id="nuc1" cx="122" cy="122" r="5" fill="#4a2f80" opacity=".55"/>
              <circle id="nuc2" cx="138" cy="136" r="3.5" fill="#4a2f80" opacity=".45"/>
            </svg>
          </div>
          <div class="cell-meta">
            <div><small data-i18n="sim_r"></small><b id="mR">—</b></div>
            <div><small data-i18n="sim_c"></small><b id="mC">—</b></div>
          </div>
          <div class="legend-note"><i></i> <span data-i18n="sim_ref"></span></div>
        </div>

        <div class="card">
          <div class="fill-top"><span data-i18n="fill_label"></span><span id="fillText">0/30</span></div>
          <div class="track"><i id="fillTrack"></i></div>
          <div class="warnline" id="warnLine"><i class="fa-solid fa-triangle-exclamation"></i><span id="warnLineT"></span></div>
          <div class="sample-row">
            <button type="button" class="btn btn-ok" onclick="loadSampleData('benign')"><i class="fa-solid fa-shield-halved"></i> <span data-i18n="sample_b"></span></button>
            <button type="button" class="btn btn-bad" onclick="loadSampleData('malignant')"><i class="fa-solid fa-biohazard"></i> <span data-i18n="sample_m"></span></button>
          </div>
          <button type="button" class="btn" style="width:100%;margin-bottom:14px" onclick="clearForm()"><i class="fa-solid fa-eraser"></i> <span data-i18n="clear_all"></span></button>
          <button type="submit" form="predictionForm" id="btnSubmit" class="btn btn-primary"></button>
          <small class="disc" data-i18n="disc"></small>
        </div>
      </aside>
    </div>

    <div class="card report" id="resultCard">
      <div class="report-head">
        <div class="l">
          <div class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round"><path d="M12 4v16M4 12h16"/></svg></div>
          <div><h3 data-i18n="rep_title"></h3><small id="resDate">—</small></div>
        </div>
        <div class="r">
          <span class="chip" id="resId">#CA-0000</span>
          <button type="button" class="btn btn-solid no-print" onclick="printReport()"><i class="fa-solid fa-print"></i> <span data-i18n="rep_print"></span></button>
        </div>
      </div>
      <div class="report-body">
        <div class="pblock" id="pblock">
          <div><small data-i18n="pat_name"></small><b id="rpName">—</b></div>
          <div><small data-i18n="pat_age"></small><b id="rpAge">—</b></div>
          <div><small data-i18n="pat_id"></small><b id="rpId">—</b></div>
          <div><small data-i18n="pat_doc"></small><b id="rpDoc">—</b></div>
          <div style="grid-column:1/-1"><small data-i18n="pat_note"></small><b id="rpNote">—</b></div>
        </div>
        <div class="banner" id="resultAlert"></div>
        <div class="banner warn" id="rangeAlert" style="display:none"></div>
        <div class="report-grid">
          <div class="meter">
            <svg viewBox="0 0 200 118">
              <defs><linearGradient id="zone" x1="20" y1="0" x2="180" y2="0" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#2b9a6a"/><stop offset=".5" stop-color="#e8b23c"/><stop offset="1" stop-color="#cf4545"/></linearGradient></defs>
              <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#eef3f3" stroke-width="18" stroke-linecap="round"/>
              <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#zone)" stroke-width="12" stroke-linecap="round"/>
              <g class="needle" id="needle"><path d="M100 100 L95.5 100 L100 34 L104.5 100 Z" fill="#12302f"/></g>
              <circle cx="100" cy="100" r="8" fill="#12302f"/><circle cx="100" cy="100" r="3" fill="#fff"/>
              <text id="lowTxt" x="14" y="116" font-size="8" fill="#7c9291" font-family="DM Sans" font-weight="600"></text>
              <text id="highTxt" x="186" y="116" font-size="8" fill="#7c9291" font-family="DM Sans" font-weight="600" text-anchor="end"></text>
            </svg>
            <div class="meter-val" id="gaugeVal">0%</div>
            <div class="meter-lbl" data-i18n="meter_lbl"></div>
          </div>
          <div>
            <div class="verdict"><small data-i18n="verdict_lbl"></small><h2 id="resPrediction"></h2></div>
            <div class="prob"><div class="prob-row"><span data-i18n="p_benign"></span><b id="resBenignText" style="color:var(--ok)">0%</b></div><div class="bar"><i id="barBenign" style="background:var(--ok)"></i></div></div>
            <div class="prob"><div class="prob-row"><span data-i18n="p_malignant"></span><b id="resMalignantText" style="color:var(--bad)">0%</b></div><div class="bar"><i id="barMalignant" style="background:var(--bad)"></i></div></div>
            <div class="note"><i class="fa-solid fa-circle-info" style="color:var(--teal);margin-top:3px"></i><span id="warnText"></span></div>
          </div>
        </div>

        <div class="radar-sec">
          <div class="radar-top">
            <div><h4 data-i18n="radar_title"></h4><p data-i18n="radar_sub"></p></div>
            <div class="seg no-print" id="segRadar">
              <button data-g="mean" class="on" data-i18n="radar_mean"></button><button data-g="error" data-i18n="radar_err"></button><button data-g="worst" data-i18n="radar_worst"></button>
            </div>
          </div>
          <div class="radar-box" id="radarBox"></div>
          <div class="rlegend" id="radarLegend"></div>
        </div>

                <div class="rep-foot" data-i18n="rep_foot"></div>
      </div>
    </div>
  </div>

  <!-- HISTORY -->
  <div id="tab-history" class="tab">
    <div class="page-head"><div><h2 data-i18n="his_title"></h2><p data-i18n="his_sub"></p></div>
      <div class="toolbar">
        <button class="btn" id="btnCmp" onclick="openCompare()" disabled><i class="fa-solid fa-code-compare"></i> <span data-i18n="his_cmp"></span></button>
        <button class="btn" onclick="exportHistoryCSV()"><i class="fa-solid fa-file-arrow-down"></i> <span data-i18n="his_csv"></span></button>
        <button class="btn" onclick="clearHistory()"><i class="fa-solid fa-trash-can"></i> <span data-i18n="his_clear"></span></button>
      </div>
    </div>
    <div class="card table-wrap"><table>
      <thead><tr><th></th><th data-i18n="th_id"></th><th data-i18n="th_time"></th><th data-i18n="th_patient"></th><th data-i18n="th_result"></th><th data-i18n="th_prob"></th><th></th></tr></thead>
      <tbody id="historyTableBody"></tbody>
    </table></div>
  </div>

  <!-- BATCH -->
  <div id="tab-batch" class="tab">
    <div class="page-head"><div><h2 data-i18n="batch_title"></h2><p data-i18n="batch_sub"></p></div>
      <div class="toolbar"><button class="btn" onclick="downloadTemplate()"><i class="fa-solid fa-download"></i> <span data-i18n="batch_tpl"></span></button></div></div>
    <div class="card">
      <div class="drop" id="drop">
        <i class="fa-solid fa-cloud-arrow-up big"></i>
        <p data-i18n="batch_drop"></p>
        <div class="row"><span data-i18n="batch_or"></span>
          <button class="btn btn-solid" onclick="$('fileIn').click()"><i class="fa-solid fa-folder-open"></i> <span data-i18n="batch_pick"></span></button></div>
        <input type="file" id="fileIn" accept=".csv,.txt,text/csv" hidden>
      </div>
      <div class="bstatus" id="batchStatus"></div>
      <div class="bactions">
        <button class="btn btn-primary" style="width:auto;padding:12px 26px" id="btnBatchRun" onclick="runBatch()" disabled><i class="fa-solid fa-play"></i> <span data-i18n="batch_run"></span></button>
        <button class="btn" id="btnBatchCsv" onclick="exportBatchCSV()" disabled><i class="fa-solid fa-file-arrow-down"></i> <span data-i18n="batch_export"></span></button>
        <button class="btn" id="btnBatchSave" onclick="saveBatchToHistory()" disabled><i class="fa-solid fa-bookmark"></i> <span data-i18n="batch_save"></span></button>
      </div>
      <div class="chips" id="batchChips"></div>
      <div class="table-wrap" id="batchTableWrap" style="display:none"><table>
        <thead><tr><th>#</th><th data-i18n="th_case"></th><th data-i18n="th_result"></th><th data-i18n="th_prob"></th></tr></thead>
        <tbody id="batchBody"></tbody></table></div>
    </div>
  </div>

  <!-- STATS -->
  <div id="tab-stats" class="tab">
    <div class="page-head"><div><h2 data-i18n="st_title"></h2><p data-i18n="st_sub"></p></div></div>
    <div class="stat-grid">
      <div class="card stat" style="--c:var(--teal)"><i class="fa-solid fa-vial"></i><span data-i18n="st_total"></span><h3 id="statTotal">0</h3></div>
      <div class="card stat" style="--c:var(--ok)"><i class="fa-solid fa-shield-heart"></i><span data-i18n="st_benign"></span><h3 id="statBenign">0</h3></div>
      <div class="card stat" style="--c:var(--bad)"><i class="fa-solid fa-biohazard"></i><span data-i18n="st_malig"></span><h3 id="statMalignant">0</h3></div>
    </div>
    <div class="card donut-card">
      <div class="donut" id="donut" data-label=""></div>
      <div class="legend">
        <div><span class="dot" style="background:var(--ok)"></span> <span data-i18n="benign"></span>: <span id="lgB">0%</span></div>
        <div><span class="dot" style="background:var(--bad)"></span> <span data-i18n="malignant"></span>: <span id="lgM">0%</span></div>
      </div>
    </div>
    <div class="card trend-card"><h3 data-i18n="trend_title"></h3><p data-i18n="trend_sub"></p><div id="trendBox"></div></div>
  </div>

  <!-- GUIDE -->
  <div id="tab-guide" class="tab">
    <div class="page-head"><div><h2 data-i18n="guide_title"></h2><p data-i18n="guide_sub"></p></div>
      <div class="search"><i class="fa-solid fa-magnifying-glass"></i><input id="guideSearch" data-i18n-ph="guide_search"></div></div>
    <div class="card"><div class="guide-grid" id="guideContainer"></div></div>
  </div>
</main>

<div class="modal" id="modal"><div class="modal-card" id="modalCard"></div></div>
<div class="ps" id="printSheet"></div>
<div class="toasts" id="toasts"></div>

<script>
/* ================= i18n ================= */
const I18N = {
vi: {
doc_title:'OncoCare - Hệ Thống Hỗ Trợ Chẩn Đoán Lâm Sàng',
nav_predict:'Chẩn Đoán AI', nav_history:'Nhật Ký', nav_batch:'Hàng Loạt', nav_stats:'Thống Kê', nav_guide:'Hướng Dẫn Chỉ Số',
eyebrow:'Hỗ trợ chẩn đoán lâm sàng',
hero_h1:'Phân tích khối u vú<br>bằng <em>học máy</em> SVM',
hero_p:'Phân loại lành tính / ác tính từ 30 đặc trưng tế bào học của bộ dữ liệu Wisconsin Breast Cancer, trình bày rõ ràng như một bản báo cáo xét nghiệm.',
kpi_acc:'Độ chính xác', kpi_feat:'Đặc trưng',
pred_title:'Phiếu nhập dữ liệu tế bào học', pred_sub:'Điền các thông số, hoặc nạp nhanh một mẫu chuẩn ở khung bên phải.',
pat_title:'Thông tin bệnh nhân', pat_sub:'Tùy chọn · chỉ lưu trên trình duyệt của bạn, không gửi lên máy chủ',
pat_name:'Họ và tên', pat_age:'Tuổi', pat_id:'Mã bệnh nhân', pat_doc:'Bác sĩ phụ trách', pat_note:'Ghi chú',
g1_t:'Nhóm chỉ số trung bình', g1_s:'Mean features · 10 chiều', g2_t:'Nhóm sai số tiêu chuẩn', g2_s:'Standard error · 10 chiều', g3_t:'Nhóm giá trị cực đoan', g3_s:'Worst features · 10 chiều',
sim_title:'Mô phỏng nhân tế bào', sim_sub:'Hình minh họa thay đổi theo bán kính, độ lõm và độ nhẵn bạn nhập. Không phải ảnh thực.',
sim_r:'Bán kính TB', sim_c:'Độ lõm TB', sim_ref:'Tham chiếu: bán kính TB nhóm lành tính (≈ 12.1)',
fill_label:'Tiến độ nhập liệu', warn_n:'{0} ô ngoài khoảng thường gặp', out_of_range:'Ngoài khoảng dữ liệu huấn luyện, hãy kiểm tra lại.',
sample_b:'Mẫu lành tính', sample_m:'Mẫu ác tính', clear_all:'Xóa toàn bộ', run:'Thực hiện chẩn đoán AI', running:'Đang phân tích...',
disc:'Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán giải phẫu bệnh.',
rep_title:'Báo cáo phân tích tế bào học', rep_print:'In / Lưu PDF',
rep_ok:'Mô hình xác định khối u có đặc trưng lành tính. Vẫn nên theo dõi định kỳ theo chỉ định của bác sĩ.',
rep_bad:'Khối u mang đặc trưng ác tính. Cần sinh thiết và hội chẩn chuyên khoa để xác nhận.',
rep_warn:'Có {0} chỉ số nằm ngoài khoảng dữ liệu huấn luyện, kết quả có thể kém tin cậy.',
meter_lbl:'Xác suất ác tính', meter_low:'THẤP', meter_high:'CAO', verdict_lbl:'Kết luận phân loại của mô hình',
benign:'Lành tính', malignant:'Ác tính', p_benign:'Lành tính (Benign)', p_malignant:'Ác tính (Malignant)',
radar_title:'So sánh với trung bình bộ dữ liệu', radar_sub:'Mỗi trục là một chỉ số, chuẩn hóa theo khoảng min–max của bộ dữ liệu. Chỉ mang tính trực quan, không phải giải thích cách mô hình quyết định.',
radar_mean:'Trung bình', radar_err:'Sai số', radar_worst:'Cực đoan', radar_case:'Ca này', radar_ben:'TB nhóm lành tính', radar_mal:'TB nhóm ác tính',
sign_doc:'Bác sĩ phụ trách', sign_maker:'Người lập phiếu', rep_foot:'Phiếu được tạo tự động bởi OncoCare AI, chỉ mang tính tham khảo lâm sàng.',
his_title:'Nhật ký khám bệnh', his_sub:'Lưu ngay trên trình duyệt này. Tích chọn 2 ca để so sánh.',
his_cmp:'So sánh 2 ca', his_csv:'Xuất CSV', his_clear:'Xóa nhật ký', confirm_clear:'Xóa toàn bộ nhật ký trên trình duyệt này?',
th_id:'Mã ca', th_time:'Thời gian', th_patient:'Bệnh nhân', th_result:'Kết luận', th_prob:'Xác suất ác tính', th_case:'Ca',
his_empty:'Chưa có lịch sử chẩn đoán nào được ghi nhận.', view:'Xem',
cmp_title:'So sánh hai ca', cmp_feature:'Chỉ số', cmp_diff:'Chênh lệch', close:'Đóng', cmp_pick:'Hãy chọn đúng 2 ca để so sánh.',
batch_title:'Dự đoán hàng loạt từ CSV', batch_sub:'Tải lên file chứa nhiều ca để dự đoán cùng lúc. Hỗ trợ cả tên cột kiểu bộ dữ liệu Kaggle (radius_mean, radius_se, radius_worst...).',
batch_drop:'Kéo thả file CSV vào đây', batch_or:'hoặc', batch_pick:'Chọn file', batch_tpl:'Tải file mẫu',
batch_parse:'Đọc được {0} dòng hợp lệ.', batch_bad:'{0} dòng bị bỏ qua:', batch_run:'Chạy dự đoán', batch_export:'Xuất kết quả CSV', batch_save:'Thêm vào nhật ký',
batch_saved:'Đã thêm {0} ca vào nhật ký', batch_limit:'File quá 1000 dòng, chỉ đọc 1000 dòng đầu.', batch_missing:'Thiếu cột: {0}', batch_row_err:'Dòng {0}: giá trị trống, âm hoặc không phải số',
batch_empty:'File không có dữ liệu.', batch_sum:'{0} ca', 
st_title:'Thống kê & xu hướng', st_sub:'Tổng quan kết quả trong nhật ký.', st_total:'Tổng số ca', st_benign:'Ca lành tính', st_malig:'Ca ác tính',
no_data:'Chưa có\ndữ liệu', ca_unit:'ca', trend_title:'Xu hướng xác suất ác tính', trend_sub:'Theo thứ tự thời gian của các ca gần nhất (tối đa 40).', trend_empty:'Cần ít nhất 2 ca để vẽ biểu đồ xu hướng.',
guide_title:'Tra cứu 30 chỉ số lâm sàng', guide_sub:'Ý nghĩa kỹ thuật của các đặc trưng trích xuất tế bào học.', guide_search:'Tìm chỉ số...', guide_none:'Không tìm thấy chỉ số phù hợp.',
ps_title:'PHIẾU KẾT QUẢ PHÂN TÍCH TẾ BÀO HỌC', ps_sub:'Hỗ trợ chẩn đoán bằng học máy (SVM) · Bộ dữ liệu Wisconsin Breast Cancer',
ps_code:'Mã phiếu', ps_date:'Ngày giờ', ps_info:'I. Thông tin bệnh nhân', ps_res:'II. Kết quả các chỉ số tế bào học', ps_concl:'III. Kết luận của mô hình',
ps_idx:'Chỉ số', ps_mean:'Trung bình', ps_se:'Sai số chuẩn', ps_worst:'Lớn nhất', ps_ref:'Tham chiếu lành tính (TB)',
ps_ref_note:'Cột tham chiếu là giá trị trung bình của nhóm lành tính trong bộ dữ liệu Wisconsin, không phải khoảng bình thường lâm sàng.',
ps_prob:'Xác suất ác tính', ps_sign_hint:'(Ký, ghi rõ họ tên)', ps_model:'Mô hình',
t_meta:'Không tải được metadata từ máy chủ.', t_fill:'Có {0} ô trống hoặc không hợp lệ.', t_server:'Máy chủ trả về lỗi {0}', t_net:'Không thể kết nối máy chủ.',
t_lb:'Đã nạp mẫu lành tính', t_lm:'Đã nạp mẫu ác tính', t_age:'Tuổi phải từ 0 đến 120.', status:'Sẵn sàng · SVM v{0}'
},
en: {
doc_title:'OncoCare - Clinical Decision Support',
nav_predict:'AI Diagnosis', nav_history:'History', nav_batch:'Batch', nav_stats:'Statistics', nav_guide:'Feature Guide',
eyebrow:'Clinical decision support',
hero_h1:'Breast mass analysis<br>with <em>machine learning</em> SVM',
hero_p:'Benign / malignant classification from 30 cytological features of the Wisconsin Breast Cancer dataset, presented like a lab report.',
kpi_acc:'Accuracy', kpi_feat:'Features',
pred_title:'Cytology data entry form', pred_sub:'Fill in the values, or quickly load a standard sample from the panel on the right.',
pat_title:'Patient information', pat_sub:'Optional · stored only in your browser, never sent to the server',
pat_name:'Full name', pat_age:'Age', pat_id:'Patient ID', pat_doc:'Physician', pat_note:'Notes',
g1_t:'Mean values', g1_s:'Mean features · 10 dimensions', g2_t:'Standard errors', g2_s:'Standard error · 10 dimensions', g3_t:'Extreme values', g3_s:'Worst features · 10 dimensions',
sim_title:'Nucleus simulation', sim_sub:'An illustration that changes with the radius, concavity and smoothness you enter. Not a real image.',
sim_r:'Mean radius', sim_c:'Mean concavity', sim_ref:'Reference: mean radius of the benign group (≈ 12.1)',
fill_label:'Entry progress', warn_n:'{0} field(s) outside the usual range', out_of_range:'Outside the training data range, please double-check.',
sample_b:'Benign sample', sample_m:'Malignant sample', clear_all:'Clear all', run:'Run AI diagnosis', running:'Analyzing...',
disc:'Results are for reference only and do not replace a pathology diagnosis.',
rep_title:'Cytology analysis report', rep_print:'Print / Save PDF',
rep_ok:'The model classifies this mass as having benign characteristics. Routine follow-up as directed by a physician is still advised.',
rep_bad:'This mass has malignant characteristics. A biopsy and specialist consultation are needed to confirm.',
rep_warn:'{0} value(s) fall outside the training data range, so the result may be less reliable.',
meter_lbl:'Malignancy probability', meter_low:'LOW', meter_high:'HIGH', verdict_lbl:'Model classification',
benign:'Benign', malignant:'Malignant', p_benign:'Benign probability', p_malignant:'Malignant probability',
radar_title:'Comparison with dataset averages', radar_sub:'Each axis is one feature, normalized to the dataset min–max range. Visual aid only, not an explanation of how the model decides.',
radar_mean:'Mean', radar_err:'Error', radar_worst:'Worst', radar_case:'This case', radar_ben:'Benign group avg', radar_mal:'Malignant group avg',
sign_doc:'Physician', sign_maker:'Prepared by', rep_foot:'Generated automatically by OncoCare AI, for clinical reference only.',
his_title:'Case history', his_sub:'Stored in this browser. Tick 2 cases to compare them.',
his_cmp:'Compare 2 cases', his_csv:'Export CSV', his_clear:'Clear history', confirm_clear:'Delete all history stored in this browser?',
th_id:'Case ID', th_time:'Time', th_patient:'Patient', th_result:'Result', th_prob:'Malignant prob.', th_case:'Case',
his_empty:'No diagnoses recorded yet.', view:'View',
cmp_title:'Compare two cases', cmp_feature:'Feature', cmp_diff:'Difference', close:'Close', cmp_pick:'Please select exactly 2 cases to compare.',
batch_title:'Batch prediction from CSV', batch_sub:'Upload a file with many cases to predict them all at once. Kaggle-style column names (radius_mean, radius_se, radius_worst...) are supported too.',
batch_drop:'Drag and drop a CSV file here', batch_or:'or', batch_pick:'Choose file', batch_tpl:'Download template',
batch_parse:'Read {0} valid row(s).', batch_bad:'{0} row(s) skipped:', batch_run:'Run predictions', batch_export:'Export results CSV', batch_save:'Add to history',
batch_saved:'Added {0} case(s) to history', batch_limit:'File exceeds 1000 rows, only the first 1000 were read.', batch_missing:'Missing columns: {0}', batch_row_err:'Row {0}: empty, negative or non-numeric value',
batch_empty:'The file has no data.', batch_sum:'{0} cases',
st_title:'Statistics & trends', st_sub:'Overview of the results in your history.', st_total:'Total cases', st_benign:'Benign cases', st_malig:'Malignant cases',
no_data:'No\ndata', ca_unit:'cases', trend_title:'Malignancy probability trend', trend_sub:'In chronological order of the most recent cases (max 40).', trend_empty:'At least 2 cases are needed to draw the trend.',
guide_title:'Reference for the 30 clinical features', guide_sub:'Technical meaning of the extracted cytological features.', guide_search:'Search features...', guide_none:'No matching feature found.',
ps_title:'CYTOLOGY ANALYSIS RESULT SHEET', ps_sub:'Machine-learning (SVM) decision support · Wisconsin Breast Cancer dataset',
ps_code:'Report no.', ps_date:'Date / time', ps_info:'I. Patient information', ps_res:'II. Cytological feature results', ps_concl:'III. Model conclusion',
ps_idx:'Feature', ps_mean:'Mean', ps_se:'Std. error', ps_worst:'Worst', ps_ref:'Benign ref. (avg)',
ps_ref_note:'The reference column is the benign-group average in the Wisconsin dataset, not a clinical normal range.',
ps_prob:'Malignant probability', ps_sign_hint:'(Signature, full name)', ps_model:'Model',
t_meta:'Could not load metadata from the server.', t_fill:'{0} field(s) are empty or invalid.', t_server:'Server returned error {0}', t_net:'Cannot reach the server.',
t_lb:'Benign sample loaded', t_lm:'Malignant sample loaded', t_age:'Age must be between 0 and 120.', status:'Ready · SVM v{0}'
}};
let lang = 'vi';
try { lang = localStorage.getItem('onco-lang') || 'vi'; } catch (e) {}
const t = (k, ...a) => { let s = (I18N[lang][k] ?? I18N.vi[k] ?? k); a.forEach((v, i) => { s = s.split('{' + i + '}').join(v); }); return s; };

const featureVi = {
  "mean radius": ["Bán kính trung bình","Kích thước trung bình nhân tế bào"],
  "mean texture": ["Độ mịn/Kết cấu bề mặt","Độ lệch chuẩn giá trị xám bề mặt"],
  "mean perimeter": ["Chu vi trung bình","Chu vi trung bình của nhân"],
  "mean area": ["Diện tích trung bình","Diện tích trung bình của nhân tế bào"],
  "mean smoothness": ["Độ trơn nhẵn TB","Độ biến thiên cục bộ của bán kính"],
  "mean compactness": ["Độ nén trung bình","Tính theo (chu vi^2 / diện tích - 1.0)"],
  "mean concavity": ["Độ lõm trung bình","Mức độ sâu các phần lõm trên bờ viền"],
  "mean concave points": ["Số điểm lõm TB","Số lượng các phần lõm trên đường viền"],
  "mean symmetry": ["Độ đối xứng TB","Mức độ cân đối hình dáng tế bào"],
  "mean fractal dimension": ["Độ chiều Fractal","Độ phức tạp bờ viền (Xấp xỉ fractal)"],
  "radius error": ["Sai số bán kính","Độ lệch chuẩn kích thước bán kính"],
  "texture error": ["Sai số kết cấu","Độ lệch chuẩn độ mịn bề mặt"],
  "perimeter error": ["Sai số chu vi","Độ lệch chuẩn chu vi nhân"],
  "area error": ["Sai số diện tích","Độ lệch chuẩn diện tích tế bào"],
  "smoothness error": ["Sai số độ nhẵn","Độ biến thiên sai số độ mịn"],
  "compactness error": ["Sai số độ nén","Sai số mức độ nén tế bào"],
  "concavity error": ["Sai số độ lõm","Sai số độ sâu phần lõm bờ viền"],
  "concave points error": ["Sai số điểm lõm","Sai số số lượng điểm lõm"],
  "symmetry error": ["Sai số độ đối xứng","Sai số mức độ cân đối"],
  "fractal dimension error": ["Sai số chiều Fractal","Sai số độ phức tạp bờ viền"],
  "worst radius": ["Bán kính lớn nhất","Giá trị lớn nhất của bán kính"],
  "worst texture": ["Kết cấu lớn nhất","Giá trị lớn nhất kết cấu bề mặt"],
  "worst perimeter": ["Chu vi lớn nhất","Chu vi lớn nhất ghi nhận được"],
  "worst area": ["Diện tích lớn nhất","Diện tích lớn nhất ghi nhận được"],
  "worst smoothness": ["Độ nhẵn lớn nhất","Độ trơn nhẵn cao nhất trong nhóm"],
  "worst compactness": ["Độ nén lớn nhất","Độ nén lớn nhất ghi nhận"],
  "worst concavity": ["Độ lõm lớn nhất","Độ lõm sâu nhất ghi nhận"],
  "worst concave points": ["Điểm lõm lớn nhất","Số điểm lõm lớn nhất ghi nhận"],
  "worst symmetry": ["Đối xứng lớn nhất","Độ lệch đối xứng lớn nhất"],
  "worst fractal dimension": ["Chiều Fractal lớn nhất","Độ phức tạp bờ viền lớn nhất"]
};
const BASES = ['radius','texture','perimeter','area','smoothness','compactness','concavity','concave points','symmetry','fractal dimension'];
const BASE_DESC_EN = ['Mean distance from center to points on the perimeter','Standard deviation of gray-scale values','Nuclear perimeter','Nuclear area','Local variation in radius lengths','perimeter² / area − 1.0','Severity of concave portions of the contour','Number of concave portions of the contour','Symmetry of the cell shape','Coastline approximation (contour complexity)'];
const BASE_LBL = { vi: ['Bán kính','Kết cấu','Chu vi','Diện tích','Độ nhẵn','Độ nén','Độ lõm','Điểm lõm','Đối xứng','Fractal'], en: ['Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave pts','Symmetry','Fractal'] };
function parseKey(f) {
  if (f.startsWith('mean ')) return ['mean', f.slice(5)];
  if (f.startsWith('worst ')) return ['worst', f.slice(6)];
  if (f.endsWith(' error')) return ['error', f.slice(0, -6)];
  return ['mean', f];
}
const featKey = (g, b) => g === 'mean' ? 'mean ' + b : g === 'error' ? b + ' error' : 'worst ' + b;
function fName(f) { if (lang === 'vi' && featureVi[f]) return featureVi[f][0]; return f.charAt(0).toUpperCase() + f.slice(1); }
function fDesc(f) {
  if (lang === 'vi' && featureVi[f]) return featureVi[f][1];
  const [g, b] = parseKey(f); const i = BASES.indexOf(b);
  const suf = g === 'mean' ? ' (mean)' : g === 'error' ? ' (standard error)' : ' (worst / largest)';
  return (i >= 0 ? BASE_DESC_EN[i] : 'Cell feature') + suf;
}

const sampleMalignant = {"mean radius":17.99,"mean texture":10.38,"mean perimeter":122.8,"mean area":1001.0,"mean smoothness":0.1184,"mean compactness":0.2776,"mean concavity":0.3001,"mean concave points":0.1471,"mean symmetry":0.2419,"mean fractal dimension":0.07871,"radius error":1.095,"texture error":0.9053,"perimeter error":8.589,"area error":153.4,"smoothness error":0.006399,"compactness error":0.04904,"concavity error":0.05373,"concave points error":0.01587,"symmetry error":0.03003,"fractal dimension error":0.006193,"worst radius":25.38,"worst texture":17.33,"worst perimeter":184.6,"worst area":2019.0,"worst smoothness":0.1622,"worst compactness":0.6656,"worst concavity":0.7119,"worst concave points":0.2654,"worst symmetry":0.4601,"worst fractal dimension":0.1189};
const sampleBenign = {"mean radius":13.54,"mean texture":14.36,"mean perimeter":87.46,"mean area":566.3,"mean smoothness":0.09779,"mean compactness":0.08129,"mean concavity":0.06664,"mean concave points":0.04781,"mean symmetry":0.1885,"mean fractal dimension":0.05766,"radius error":0.2699,"texture error":0.7886,"perimeter error":2.058,"area error":23.56,"smoothness error":0.008462,"compactness error":0.0146,"concavity error":0.02387,"concave points error":0.01315,"symmetry error":0.0198,"fractal dimension error":0.0023,"worst radius":15.11,"worst texture":19.26,"worst perimeter":99.7,"worst area":711.2,"worst smoothness":0.144,"worst compactness":0.1773,"worst concavity":0.239,"worst concave points":0.1288,"worst symmetry":0.2977,"worst fractal dimension":0.07259};

/* ================= State & helpers ================= */
const $ = id => document.getElementById(id);
const idOf = f => 'inp_' + f.split(' ').join('_');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let featureNames = [], REF = {}, modelVer = '', serverWarn = '';
let historyList = [], caseSeq = 0, selected = [], shown = null, radarGroup = 'mean';
let batchRows = [], batchResults = [], batchBad = [], batchNotes = '';
const fmtTime = ts => new Date(ts).toLocaleString(lang === 'vi' ? 'vi-VN' : 'en-GB');
const parseNum = v => parseFloat(String(v).trim().replace(',', '.'));
const getNum = f => { const el = $(idOf(f)); return el ? parseNum(el.value) : NaN; };
const nextId = () => 'CA-' + String(++caseSeq).padStart(4, '0');

function toast(msg, ok) {
  const d = document.createElement('div'); d.className = 'toast' + (ok ? ' ok' : ''); d.textContent = msg;
  $('toasts').appendChild(d); setTimeout(() => d.remove(), 4200);
}
function download(name, text, mime) {
  const blob = new Blob(['\ufeff' + text], { type: mime || 'text/csv;charset=utf-8' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
const csvCell = v => { const s = String(v ?? ''); return /[",;\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
const toCSV = rows => rows.map(r => r.map(csvCell).join(',')).join('\r\n');

/* ================= Tabs & language ================= */
function switchTab(id) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tabs a').forEach(el => el.classList.toggle('active', el.dataset.tab === id));
  $(id).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
document.querySelectorAll('.tabs a').forEach(a => a.onclick = () => switchTab(a.dataset.tab));

function applyLang() {
  document.documentElement.lang = lang; document.title = t('doc_title');
  document.querySelectorAll('[data-i18n]').forEach(el => el.textContent = t(el.dataset.i18n));
  document.querySelectorAll('[data-i18n-html]').forEach(el => el.innerHTML = t(el.dataset.i18nHtml));
  document.querySelectorAll('[data-i18n-ph]').forEach(el => el.placeholder = t(el.dataset.i18nPh));
  document.querySelectorAll('.lang button').forEach(b => b.classList.toggle('on', b.dataset.lang === lang));
  $('verText').textContent = modelVer ? t('status', modelVer) : '…';
  $('lowTxt').textContent = t('meter_low'); $('highTxt').textContent = t('meter_high');
  document.querySelectorAll('[data-f]').forEach(el => el.textContent = fName(el.dataset.f));
  document.querySelectorAll('[data-fd]').forEach(el => el.textContent = fDesc(el.dataset.fd));
  setRunBtn(false); renderGuide($('guideSearch').value); renderHistory(); renderStats(); renderBatch(); updateFill();
  if (shown) renderResult(false);
}
document.querySelectorAll('.lang button').forEach(b => b.onclick = () => { lang = b.dataset.lang; try { localStorage.setItem('onco-lang', lang); } catch (e) {} applyLang(); });
function setRunBtn(loading) {
  const b = $('btnSubmit'); b.disabled = loading;
  b.innerHTML = loading ? '<i class="fa-solid fa-spinner fa-spin"></i> ' + t('running') : '<i class="fa-solid fa-wand-magic-sparkles"></i> ' + t('run');
}

/* ================= Init ================= */
async function init() {
  loadHistory(); applyLang();
  try {
    const d = await (await fetch('/metadata')).json();
    featureNames = d.feature_names; modelVer = d.model_version || ''; serverWarn = d.warning || '';
  } catch (e) { toast(t('t_meta')); return; }
  try { const r = await fetch('/reference'); if (r.ok) REF = await r.json(); } catch (e) {}
  renderInputs(); applyLang(); setValues(sampleBenign);
}

/* ================= Form ================= */
function renderInputs() {
  const g = { mean: '', error: '', worst: '' };
  featureNames.forEach(f => {
    const html = `<div class="field"><label for="${idOf(f)}" data-f="${f}"></label><div class="hint" data-fd="${f}"></div><input class="fin" type="text" inputmode="decimal" autocomplete="off" data-feat="${f}" name="${f}" id="${idOf(f)}"><div class="warnmsg" data-i18n="out_of_range"></div></div>`;
    g[parseKey(f)[0]] += html;
  });
  $('groupMean').innerHTML = g.mean; $('groupError').innerHTML = g.error; $('groupWorst').innerHTML = g.worst;
  document.querySelectorAll('.fin').forEach(i => i.addEventListener('input', () => { i.parentElement.classList.remove('invalid'); onInputChange(); }));
}
function checkField(f) {
  const el = $(idOf(f)); if (!el) return 0;
  const w = el.parentElement; w.classList.remove('warn');
  const v = parseNum(el.value);
  if (el.value.trim() === '' || !isFinite(v)) return 0;
  if (v < 0) { w.classList.add('invalid'); return 0; }
  const r = REF[f]; if (!r) return 0;
  const span = (r.max - r.min) || 1;
  if (v < r.min - 0.15 * span || v > r.max + 0.15 * span) { w.classList.add('warn'); return 1; }
  return 0;
}
function countWarn() { let n = 0; featureNames.forEach(f => n += checkField(f)); return n; }
function onInputChange() { updateFill(); drawCell(); }
function updateFill() {
  let n = 0; featureNames.forEach(f => { if (isFinite(getNum(f))) n++; });
  $('fillText').textContent = n + '/' + featureNames.length;
  $('fillTrack').style.width = (featureNames.length ? n / featureNames.length * 100 : 0) + '%';
  const w = countWarn(); const wl = $('warnLine');
  wl.style.display = w ? 'flex' : 'none'; $('warnLineT').textContent = t('warn_n', w);
}
function setValues(data) {
  featureNames.forEach(f => { const el = $(idOf(f)); if (el) { el.value = data[f] ?? ''; el.parentElement.classList.remove('invalid'); } });
  updateFill(); drawCell();
}
function loadSampleData(type) { setValues(type === 'malignant' ? sampleMalignant : sampleBenign); toast(t(type === 'malignant' ? 't_lm' : 't_lb'), true); }
function clearForm() { setValues({}); setPatient({}); $('resultCard').classList.remove('show'); shown = null; }
const getPatient = () => ({ name: $('pName').value.trim(), age: $('pAge').value.trim(), pid: $('pId').value.trim(), doctor: $('pDoc').value.trim(), note: $('pNote').value.trim() });
function setPatient(p) { $('pName').value = p.name || ''; $('pAge').value = p.age || ''; $('pId').value = p.pid || ''; $('pDoc').value = p.doctor || ''; $('pNote').value = p.note || ''; }

/* Cell simulation */
function drawCell() {
  let R = getNum('mean radius'), C = getNum('mean concavity'), S = getNum('mean smoothness');
  $('mR').textContent = isFinite(R) ? R.toFixed(2) : '—'; $('mC').textContent = isFinite(C) ? C.toFixed(3) : '—';
  if (!isFinite(R)) R = 12; if (!isFinite(C)) C = 0.05; if (!isFinite(S)) S = 0.09;
  const base = Math.max(5, Math.min(34, R)) * 3.4;
  const irr = Math.max(0, Math.min(C / 0.35, 1.3)), rough = Math.max(0, Math.min((S - 0.06) / 0.1, 1.2));
  const N = 140; let d = '';
  for (let i = 0; i <= N; i++) {
    const a = i / N * Math.PI * 2;
    const k = 1 + irr * 0.14 * Math.sin(5 * a + 0.8) + irr * 0.09 * Math.sin(9 * a + 2.1) + irr * 0.05 * Math.sin(14 * a) + rough * 0.025 * Math.sin(27 * a);
    const r = Math.min(base * k, 126);
    d += (i ? 'L' : 'M') + (130 + r * Math.cos(a)).toFixed(1) + ' ' + (130 + r * Math.sin(a)).toFixed(1);
  }
  $('cellShape').setAttribute('d', d + 'Z');
  $('nuc1').setAttribute('r', (4 + irr * 3).toFixed(1)); $('nuc1').setAttribute('cx', (130 - base * 0.12).toFixed(1)); $('nuc1').setAttribute('cy', (130 - base * 0.1).toFixed(1));
  $('nuc2').setAttribute('cx', (130 + base * 0.14).toFixed(1)); $('nuc2').setAttribute('cy', (130 + base * 0.12).toFixed(1));
}

/* ================= Radar ================= */
function norm(v, f) { const r = REF[f]; if (!r) return 0; return Math.max(0, Math.min(1.15, (v - r.min) / ((r.max - r.min) || 1))); }
function radarSVG(grp, series) {
  const W = 400, H = 350, cx = 200, cy = 175, R = 110, n = BASES.length;
  const pt = (i, v) => { const a = -Math.PI / 2 + i * 2 * Math.PI / n; return [cx + R * v * Math.cos(a), cy + R * v * Math.sin(a)]; };
  let s = `<svg viewBox="0 0 ${W} ${H}" class="radar">`;
  [0.25, 0.5, 0.75, 1].forEach(g => { s += `<polygon points="${BASES.map((_, i) => pt(i, g).map(x => x.toFixed(1)).join(',')).join(' ')}" fill="none" stroke="#d2e0df" stroke-width="1"/>`; });
  BASES.forEach((b, i) => {
    const [x, y] = pt(i, 1), [lx, ly] = pt(i, 1.17);
    const anchor = Math.abs(lx - cx) < 10 ? 'middle' : (lx > cx ? 'start' : 'end');
    s += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#e1ebea"/><text x="${lx.toFixed(1)}" y="${(ly + 4).toFixed(1)}" text-anchor="${anchor}" font-size="11" fill="#3d5a59" font-weight="600" font-family="DM Sans">${BASE_LBL[lang][i]}</text>`;
  });
  series.forEach(se => {
    const pts = BASES.map((b, i) => { const f = featKey(grp, b); const v = se.get(f); return pt(i, isFinite(v) ? norm(v, f) : 0).map(x => x.toFixed(1)).join(','); }).join(' ');
    s += `<polygon points="${pts}" fill="${se.color}" fill-opacity="${se.fill}" stroke="${se.color}" stroke-width="${se.w || 2}" ${se.dash ? 'stroke-dasharray="5 4"' : ''} stroke-linejoin="round"/>`;
  });
  return s + '</svg>';
}
const legendHTML = items => items.map(i => `<span><i style="background:${i.color}"></i>${i.label}</span>`).join('');
function refSeries() {
  if (!Object.keys(REF).length) return [];
  return [
    { label: t('radar_ben'), color: '#2b9a6a', fill: 0.10, w: 1.8, dash: true, get: f => REF[f] && REF[f].mean_benign },
    { label: t('radar_mal'), color: '#cf4545', fill: 0.10, w: 1.8, dash: true, get: f => REF[f] && REF[f].mean_malignant }
  ];
}
function drawRadar() {
  if (!shown) return;
  const ser = refSeries(); ser.push({ label: t('radar_case'), color: '#0f7c78', fill: 0.22, w: 2.6, get: f => shown.features[f] });
  $('radarBox').innerHTML = radarSVG(radarGroup, ser); $('radarLegend').innerHTML = legendHTML(ser);
  document.querySelectorAll('#segRadar button').forEach(b => b.classList.toggle('on', b.dataset.g === radarGroup));
}
document.querySelectorAll('#segRadar button').forEach(b => b.onclick = () => { radarGroup = b.dataset.g; drawRadar(); });

/* ================= Result ================= */
function renderResult(scroll) {
  const e = shown; if (!e) return;
  const pB = e.pb * 100, pM = e.pm * 100, bad = e.mal;
  $('resultCard').classList.add('show');
  $('resId').textContent = '#' + e.id; $('resDate').textContent = fmtTime(e.ts);
  const p = e.patient || {}, any = p.name || p.age || p.pid || p.doctor || p.note;
  $('pblock').style.display = any ? 'grid' : 'none';
  $('rpName').textContent = p.name || '—'; $('rpAge').textContent = p.age || '—'; $('rpId').textContent = p.pid || '—'; $('rpDoc').textContent = p.doctor || '—'; $('rpNote').textContent = p.note || '—';
  $('resBenignText').textContent = pB.toFixed(2) + '%'; $('resMalignantText').textContent = pM.toFixed(2) + '%';
  $('gaugeVal').textContent = pM.toFixed(1) + '%'; $('gaugeVal').style.color = bad ? 'var(--bad)' : 'var(--ok)';
  $('warnText').textContent = serverWarn || t('disc');
  const needle = $('needle'); needle.style.transform = 'rotate(-90deg)';
  $('barBenign').style.width = '0%'; $('barMalignant').style.width = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    needle.style.transform = `rotate(${-90 + pM / 100 * 180}deg)`;
    $('barBenign').style.width = pB + '%'; $('barMalignant').style.width = pM + '%';
  }));
  const a = $('resultAlert');
  if (!bad) {
    $('resPrediction').innerHTML = `<b style="color:var(--ok)">${t('benign')}</b><span>Benign</span>`;
    a.className = 'banner ok'; a.innerHTML = `<i class="fa-solid fa-circle-check" style="margin-top:3px"></i><span>${t('rep_ok')}</span>`;
  } else {
    $('resPrediction').innerHTML = `<b style="color:var(--bad)">${t('malignant')}</b><span>Malignant</span>`;
    a.className = 'banner bad'; a.innerHTML = `<i class="fa-solid fa-triangle-exclamation" style="margin-top:3px"></i><span>${t('rep_bad')}</span>`;
  }
  const ra = $('rangeAlert');
  if (e.warn > 0) { ra.style.display = 'flex'; ra.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="margin-top:3px"></i><span>${t('rep_warn', e.warn)}</span>`; } else ra.style.display = 'none';
  drawRadar();
  if (scroll) $('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function buildSheet() {
  const e = shown; if (!e) return;
  const p = e.patient || {}, bad = e.mal;
  const dots = v => v ? esc(v) : '<span class="ps-blank"></span>';
  const val = f => e.features[f] !== undefined ? e.features[f] : '—';
  const rows = BASES.map((b, i) => {
    const fm = featKey('mean', b), fe = featKey('error', b), fw = featKey('worst', b);
    const ref = REF[fm] ? Number(REF[fm].mean_benign.toPrecision(4)) : '—';
    return `<tr><td class="c">${i + 1}</td><td>${esc(BASE_LBL[lang][i])}</td><td class="n">${val(fm)}</td><td class="n">${val(fe)}</td><td class="n">${val(fw)}</td><td class="n">${ref}</td></tr>`;
  }).join('');
  $('printSheet').innerHTML = `
    <div class="ps-head">
      <div class="ps-brand"><div class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round"><path d="M12 4v16M4 12h16"/></svg></div>
        <div><b>Onco<i>Care</i> AI</b><small>Clinical Decision Support</small></div></div>
      <div class="ps-meta">${t('ps_code')}: <b>${e.id}</b><br>${t('ps_date')}: ${fmtTime(e.ts)}</div>
    </div>
    <div class="ps-title"><h1>${t('ps_title')}</h1><p>${t('ps_sub')}</p></div>
    <div class="ps-sec" style="margin-top:0">${t('ps_info')}</div>
    <table>
      <tr><td class="lab">${t('pat_name')}</td><td colspan="3">${dots(p.name)}</td></tr>
      <tr><td class="lab">${t('pat_age')}</td><td style="width:33%">${dots(p.age)}</td><td class="lab">${t('pat_id')}</td><td>${dots(p.pid)}</td></tr>
      <tr><td class="lab">${t('pat_doc')}</td><td colspan="3">${dots(p.doctor)}</td></tr>
      <tr><td class="lab">${t('pat_note')}</td><td colspan="3">${dots(p.note)}</td></tr>
    </table>
    <div class="ps-sec">${t('ps_res')}</div>
    <table class="ps-res">
      <thead><tr><th>#</th><th style="text-align:left">${t('ps_idx')}</th><th>${t('ps_mean')}</th><th>${t('ps_se')}</th><th>${t('ps_worst')}</th><th>${t('ps_ref')}</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="ps-note">${t('ps_ref_note')}</p>
    <div class="ps-sec">${t('ps_concl')}</div>
    <div class="ps-concl ${bad ? 'bad' : 'ok'}">
      <div><small>${t('verdict_lbl')}</small><b>${t(bad ? 'malignant' : 'benign')}</b></div>
      <div class="pr">${t('ps_prob')}: <b style="font-size:12pt;font-family:var(--mono)">${(e.pm * 100).toFixed(2)}%</b><br>${t('benign')}: ${(e.pb * 100).toFixed(2)}%</div>
    </div>
    <p class="ps-advice">${t(bad ? 'rep_bad' : 'rep_ok')}</p>
    ${e.warn > 0 ? `<p class="ps-warn">${t('rep_warn', e.warn)}</p>` : ''}
    <div class="ps-sign"><div>${t('sign_maker')}<small>${t('ps_sign_hint')}</small></div><div>${t('sign_doc')}<small>${t('ps_sign_hint')}</small></div></div>
    <div class="ps-foot">${t('disc')} · ${t('ps_model')}: SVM${modelVer ? ' v' + modelVer : ''}</div>`;
}
function printReport() {
  if (!shown) return;
  const f = getPatient();
  if (Object.values(f).some(Boolean)) { shown.patient = f; saveHistory(); renderHistory(); renderResult(false); }
  buildSheet();
  const old = document.title; document.title = 'OncoCare-' + (shown ? shown.id : 'report');
  const restore = () => { document.title = old; window.removeEventListener('afterprint', restore); };
  window.addEventListener('afterprint', restore); window.print();
}

/* ================= Submit ================= */
$('predictionForm').addEventListener('submit', async ev => {
  ev.preventDefault();
  const obj = {}, bad = [];
  featureNames.forEach(f => { const el = $(idOf(f)), v = parseNum(el.value); if (el.value.trim() === '' || !isFinite(v) || v < 0) bad.push(el); else obj[f] = v; });
  const age = $('pAge').value.trim();
  if (age !== '' && !(parseFloat(age) >= 0 && parseFloat(age) <= 120)) { toast(t('t_age')); $('pAge').focus(); return; }
  if (bad.length) {
    bad.forEach(el => el.parentElement.classList.add('invalid'));
    bad[0].scrollIntoView({ behavior: 'smooth', block: 'center' }); bad[0].focus(); toast(t('t_fill', bad.length)); return;
  }
  setRunBtn(true);
  try {
    const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features: obj }) });
    if (!res.ok) throw new Error(t('t_server', res.status));
    const d = await res.json();
    const entry = { id: nextId(), ts: Date.now(), patient: getPatient(), mal: d.predicted_label !== 'benign', pm: d.probability_malignant, pb: d.probability_benign, features: obj, warn: countWarn() };
    historyList.unshift(entry); if (historyList.length > 200) historyList.length = 200;
    saveHistory(); shown = entry; renderResult(true); renderHistory(); renderStats();
  } catch (err) { toast(err.message || t('t_net')); }
  finally { setRunBtn(false); }
});

/* ================= History ================= */
function loadHistory() {
  try { historyList = JSON.parse(localStorage.getItem('onco-history') || '[]'); } catch (e) { historyList = []; }
  caseSeq = historyList.reduce((m, h) => Math.max(m, parseInt(String(h.id).replace(/\D/g, '')) || 0), 0);
}
function saveHistory() { try { localStorage.setItem('onco-history', JSON.stringify(historyList)); } catch (e) {} }
function renderHistory() {
  const tb = $('historyTableBody'), nc = $('navCount');
  nc.style.display = historyList.length ? 'inline-block' : 'none'; nc.textContent = historyList.length;
  selected = selected.filter(id => historyList.some(h => h.id === id));
  $('btnCmp').disabled = selected.length !== 2;
  if (!historyList.length) { tb.innerHTML = `<tr><td colspan="7" class="empty"><i class="fa-regular fa-folder-open"></i>${t('his_empty')}</td></tr>`; return; }
  tb.innerHTML = historyList.map(h => {
    const p = h.patient || {}, who = p.name ? `${esc(p.name)}<small>${[p.age ? esc(p.age) + (lang === 'vi' ? ' tuổi' : ' y/o') : '', p.pid ? esc(p.pid) : ''].filter(Boolean).join(' · ')}</small>` : '—';
    return `<tr><td><input type="checkbox" ${selected.includes(h.id) ? 'checked' : ''} onchange="toggleSel('${h.id}',this.checked)"></td>
      <td class="mono">${h.id}</td><td style="color:var(--muted)">${fmtTime(h.ts)}</td><td>${who}</td>
      <td><span class="badge ${h.mal ? 'bad' : 'ok'}">${t(h.mal ? 'malignant' : 'benign')}</span></td><td class="mono">${(h.pm * 100).toFixed(2)}%</td>
      <td style="text-align:right;white-space:nowrap"><button class="btn" style="padding:7px 14px" onclick="viewCase('${h.id}')"><i class="fa-regular fa-eye"></i> ${t('view')}</button>
      <button class="row-btn" onclick="delCase('${h.id}')"><i class="fa-solid fa-xmark"></i></button></td></tr>`;
  }).join('');
}
function toggleSel(id, on) {
  selected = selected.filter(x => x !== id);
  if (on) { selected.push(id); if (selected.length > 2) selected.shift(); }
  renderHistory();
}
function viewCase(id) {
  const h = historyList.find(x => x.id === id); if (!h) return;
  setValues(h.features); setPatient(h.patient || {}); switchTab('tab-predict'); shown = h; renderResult(true);
}
function delCase(id) { historyList = historyList.filter(h => h.id !== id); if (shown && shown.id === id) { shown = null; $('resultCard').classList.remove('show'); } saveHistory(); renderHistory(); renderStats(); }
function clearHistory() { if (!historyList.length || !confirm(t('confirm_clear'))) return; historyList = []; selected = []; shown = null; $('resultCard').classList.remove('show'); saveHistory(); renderHistory(); renderStats(); }
function exportHistoryCSV() {
  if (!historyList.length) return;
  const head = ['id', 'time', 'patient_name', 'age', 'patient_id', 'physician', 'note', 'result', 'prob_malignant', 'prob_benign', ...featureNames];
  const rows = [head, ...historyList.map(h => { const p = h.patient || {}; return [h.id, new Date(h.ts).toISOString(), p.name, p.age, p.pid, p.doctor, p.note, h.mal ? 'malignant' : 'benign', h.pm.toFixed(6), h.pb.toFixed(6), ...featureNames.map(f => h.features[f])]; })];
  download('oncocare_history.csv', toCSV(rows));
}

/* ================= Compare ================= */
let cmpGroup = 'mean', cmpPair = null;
function openCompare() {
  if (selected.length !== 2) { toast(t('cmp_pick')); return; }
  cmpPair = selected.map(id => historyList.find(h => h.id === id)); renderCompare(); $('modal').classList.add('show');
}
function closeModal() { $('modal').classList.remove('show'); }
$('modal').addEventListener('click', e => { if (e.target === $('modal')) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
function setCmpGroup(g) { cmpGroup = g; renderCompare(); }
function renderCompare() {
  const [a, b] = cmpPair, cols = ['#0f7c78', '#6a5acd'];
  const head = (h, i) => { const p = h.patient || {}; return `<div class="cmp-head" style="--c:${cols[i]}"><div><b>${h.id}</b><span class="badge ${h.mal ? 'bad' : 'ok'}">${t(h.mal ? 'malignant' : 'benign')}</span></div><div><small>${esc(p.name || '—')}</small><b>${(h.pm * 100).toFixed(1)}%</b></div><small>${fmtTime(h.ts)}</small></div>`; };
  const ser = [{ label: a.id, color: cols[0], fill: 0.18, w: 2.6, get: f => a.features[f] }, { label: b.id, color: cols[1], fill: 0.18, w: 2.6, get: f => b.features[f] }];
  const seg = ['mean', 'error', 'worst'].map(g => `<button class="${g === cmpGroup ? 'on' : ''}" onclick="setCmpGroup('${g}')">${t('radar_' + (g === 'mean' ? 'mean' : g === 'error' ? 'err' : 'worst'))}</button>`).join('');
  const rows = featureNames.map(f => {
    const va = a.features[f], vb = b.features[f], d = va ? (vb - va) / Math.abs(va) * 100 : 0;
    return `<tr><td>${esc(fName(f))}</td><td class="mono">${va}</td><td class="mono">${vb}</td><td class="mono ${Math.abs(d) > 25 ? 'diff-hi' : ''}">${d > 0 ? '+' : ''}${d.toFixed(1)}%</td></tr>`;
  }).join('');
  $('modalCard').innerHTML = `<div class="modal-top"><h3>${t('cmp_title')}</h3><div style="display:flex;gap:10px;align-items:center"><div class="seg">${seg}</div><button class="btn" onclick="closeModal()"><i class="fa-solid fa-xmark"></i> ${t('close')}</button></div></div>
    <div class="cmp-heads">${head(a, 0)}${head(b, 1)}</div>
    <div class="radar-box">${radarSVG(cmpGroup, ser)}</div><div class="rlegend">${legendHTML(ser)}</div>
    <div class="cmp-tbl"><table><thead><tr><th>${t('cmp_feature')}</th><th>${a.id}</th><th>${b.id}</th><th>${t('cmp_diff')}</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

/* ================= Stats & trend ================= */
function renderStats() {
  const total = historyList.length, mal = historyList.filter(h => h.mal).length, ben = total - mal;
  $('statTotal').textContent = total; $('statBenign').textContent = ben; $('statMalignant').textContent = mal;
  const d = $('donut');
  if (!total) { d.style.background = 'conic-gradient(var(--line) 0 100%)'; d.dataset.label = t('no_data'); $('lgB').textContent = '0%'; $('lgM').textContent = '0%'; }
  else { const b = ben / total * 100; d.style.background = `conic-gradient(var(--ok) 0 ${b}%, var(--bad) ${b}% 100%)`; d.dataset.label = total + '\n' + t('ca_unit'); $('lgB').textContent = b.toFixed(0) + '%'; $('lgM').textContent = (100 - b).toFixed(0) + '%'; }
  renderTrend();
}
function renderTrend() {
  const pts = [...historyList].reverse().slice(-40), box = $('trendBox');
  if (pts.length < 2) { box.innerHTML = `<div class="empty" style="padding:40px 10px!important"><i class="fa-solid fa-chart-line"></i>${t('trend_empty')}</div>`; return; }
  const W = 720, H = 270, l = 46, r = 18, tp = 16, bt = 34, n = pts.length;
  const X = i => l + i * (W - l - r) / (n - 1), Y = p => tp + (1 - p) * (H - tp - bt);
  let s = `<svg viewBox="0 0 ${W} ${H}" font-family="DM Sans">`;
  [0, 0.25, 0.5, 0.75, 1].forEach(g => { s += `<line x1="${l}" y1="${Y(g)}" x2="${W - r}" y2="${Y(g)}" stroke="${g === 0.5 ? '#e8b23c' : '#e1ebea'}" ${g === 0.5 ? 'stroke-dasharray="6 5"' : ''}/><text x="${l - 8}" y="${Y(g) + 4}" text-anchor="end" font-size="11" fill="#7c9291">${g * 100}%</text>`; });
  s += `<polyline fill="none" stroke="#0f7c78" stroke-width="2.5" stroke-linejoin="round" points="${pts.map((h, i) => X(i).toFixed(1) + ',' + Y(h.pm).toFixed(1)).join(' ')}"/>`;
  pts.forEach((h, i) => { s += `<circle cx="${X(i).toFixed(1)}" cy="${Y(h.pm).toFixed(1)}" r="5" fill="${h.mal ? '#cf4545' : '#2b9a6a'}" stroke="#fff" stroke-width="2"><title>${h.id}: ${(h.pm * 100).toFixed(1)}%</title></circle>`; });
  const step = Math.ceil(n / 10);
  pts.forEach((h, i) => { if (i % step === 0 || i === n - 1) s += `<text x="${X(i)}" y="${H - 10}" text-anchor="middle" font-size="10.5" fill="#7c9291">${h.id.replace('CA-', '#')}</text>`; });
  box.innerHTML = s + '</svg>';
}

/* ================= Guide ================= */
function renderGuide(filter) {
  const q = (filter || '').toLowerCase().trim(); let html = '';
  featureNames.forEach((k, i) => {
    const nm = fName(k), ds = fDesc(k);
    if (q && !(k.includes(q) || nm.toLowerCase().includes(q) || ds.toLowerCase().includes(q))) return;
    html += `<div class="guide-item"><h4><em>${String(i + 1).padStart(2, '0')}</em>${esc(nm)}</h4><code>${k}</code><p>${esc(ds)}</p></div>`;
  });
  $('guideContainer').innerHTML = html || `<div class="empty" style="grid-column:1/-1">${t('guide_none')}</div>`;
}
$('guideSearch').addEventListener('input', e => renderGuide(e.target.value));

/* ================= Batch CSV ================= */
function parseCSV(text) {
  text = text.replace(/^\uFEFF/, '');
  const first = text.split(/\r?\n/)[0] || '';
  const delim = [',', ';', '\t'].map(d => [d, first.split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
  const rows = []; let row = [], cur = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += c; }
    else if (c === '"') q = true;
    else if (c === delim) { row.push(cur); cur = ''; }
    else if (c === '\n') { row.push(cur); rows.push(row); row = []; cur = ''; }
    else if (c !== '\r') cur += c;
  }
  if (cur !== '' || row.length) { row.push(cur); rows.push(row); }
  return rows.filter(r => r.some(x => x.trim() !== ''));
}
function normHeader(h) {
  h = h.trim().toLowerCase().replace(/[_\-]+/g, ' ').replace(/\s+/g, ' ');
  if (featureNames.includes(h)) return h;
  let c;
  if (h.endsWith(' mean')) { c = 'mean ' + h.slice(0, -5); if (featureNames.includes(c)) return c; }
  if (h.endsWith(' se')) { c = h.slice(0, -3) + ' error'; if (featureNames.includes(c)) return c; }
  if (h.endsWith(' worst')) { c = 'worst ' + h.slice(0, -6); if (featureNames.includes(c)) return c; }
  return h;
}
function handleFile(file) {
  if (!file) return;
  const rd = new FileReader();
  rd.onload = () => {
    batchRows = []; batchResults = []; batchBad = []; batchNotes = '';
    const rows = parseCSV(String(rd.result));
    if (rows.length < 2) { batchNotes = t('batch_empty'); renderBatch(); return; }
    const hdr = rows[0].map(normHeader), idx = {};
    featureNames.forEach(f => { idx[f] = hdr.indexOf(f); });
    const missing = featureNames.filter(f => idx[f] < 0);
    if (missing.length) { batchNotes = t('batch_missing', missing.slice(0, 6).join(', ') + (missing.length > 6 ? ' +' + (missing.length - 6) : '')); renderBatch(); return; }
    const idCol = hdr.findIndex(h => ['id', 'patient id', 'pid', 'name', 'case', 'ho ten', 'ma benh nhan'].includes(h));
    let data = rows.slice(1);
    if (data.length > 1000) { data = data.slice(0, 1000); toast(t('batch_limit')); }
    data.forEach((r, k) => {
      const feats = {}; let ok = true;
      featureNames.forEach(f => { const v = parseNum(r[idx[f]] ?? ''); if (!isFinite(v) || v < 0) ok = false; else feats[f] = v; });
      if (ok) batchRows.push({ id: (idCol >= 0 && r[idCol] && r[idCol].trim()) || 'R' + (k + 1), features: feats });
      else batchBad.push(t('batch_row_err', k + 2));
    });
    renderBatch();
  };
  rd.readAsText(file);
}
function renderBatch() {
  const st = $('batchStatus'); let html = '';
  if (batchNotes) html += `<div style="color:#a93232">${esc(batchNotes)}</div>`;
  else if (batchRows.length || batchBad.length) {
    html += `<div style="color:var(--ink-2)"><i class="fa-solid fa-circle-check" style="color:var(--ok)"></i> ${t('batch_parse', batchRows.length)}</div>`;
    if (batchBad.length) html += `<div class="errs"><b>${t('batch_bad', batchBad.length)}</b><br>${batchBad.slice(0, 5).map(esc).join('<br>')}${batchBad.length > 5 ? '<br>…' : ''}</div>`;
  }
  st.innerHTML = html;
  $('btnBatchRun').disabled = !batchRows.length;
  $('btnBatchCsv').disabled = !batchResults.length; $('btnBatchSave').disabled = !batchResults.length;
  const wrap = $('batchTableWrap'), chips = $('batchChips');
  if (!batchResults.length) { wrap.style.display = 'none'; chips.innerHTML = ''; return; }
  const mal = batchResults.filter(r => r.predicted_label !== 'benign').length;
  chips.innerHTML = `<span class="chip" style="font-family:var(--sans)">${t('batch_sum', batchResults.length)}</span><span class="badge ok">${t('benign')}: ${batchResults.length - mal}</span><span class="badge bad">${t('malignant')}: ${mal}</span>`;
  wrap.style.display = 'block';
  $('batchBody').innerHTML = batchResults.slice(0, 300).map((r, i) => `<tr><td class="mono">${i + 1}</td><td>${esc(batchRows[i].id)}</td><td><span class="badge ${r.predicted_label !== 'benign' ? 'bad' : 'ok'}">${t(r.predicted_label !== 'benign' ? 'malignant' : 'benign')}</span></td><td class="mono">${(r.probability_malignant * 100).toFixed(2)}%</td></tr>`).join('');
}
async function runBatch() {
  if (!batchRows.length) return;
  const b = $('btnBatchRun'); b.disabled = true;
  try {
    const res = await fetch('/predict_batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rows: batchRows.map(r => r.features) }) });
    if (!res.ok) throw new Error(t('t_server', res.status));
    batchResults = (await res.json()).results; renderBatch();
  } catch (e) { toast(e.message || t('t_net')); }
  finally { b.disabled = !batchRows.length; }
}
function exportBatchCSV() {
  const rows = [['id', 'result', 'prob_malignant', 'prob_benign'], ...batchResults.map((r, i) => [batchRows[i].id, r.predicted_label, r.probability_malignant.toFixed(6), r.probability_benign.toFixed(6)])];
  download('oncocare_batch_results.csv', toCSV(rows));
}
function saveBatchToHistory() {
  const now = Date.now();
  batchResults.forEach((r, i) => historyList.unshift({ id: nextId(), ts: now + i, patient: { name: batchRows[i].id }, mal: r.predicted_label !== 'benign', pm: r.probability_malignant, pb: r.probability_benign, features: batchRows[i].features, warn: 0 }));
  if (historyList.length > 200) historyList.length = 200;
  saveHistory(); renderHistory(); renderStats(); toast(t('batch_saved', batchResults.length), true);
}
function downloadTemplate() {
  const rows = [['id', ...featureNames], ['demo-benign', ...featureNames.map(f => sampleBenign[f])], ['demo-malignant', ...featureNames.map(f => sampleMalignant[f])]];
  download('oncocare_template.csv', toCSV(rows));
}
const drop = $('drop');
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', e => handleFile(e.dataTransfer.files[0]));
$('fileIn').addEventListener('change', e => { handleFile(e.target.files[0]); e.target.value = ''; });

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

@app.get("/reference")
def get_reference():
    return reference

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

@app.post("/predict_batch")
def predict_batch(payload: BatchRequest):
    expected = metadata["feature_names"]
    if not payload.rows:
        raise HTTPException(status_code=422, detail="rows must not be empty")
    if len(payload.rows) > 1000:
        raise HTTPException(status_code=422, detail="At most 1000 rows per request")

    matrix = []
    for i, row in enumerate(payload.rows):
        missing = sorted(set(expected) - set(row))
        extra = sorted(set(row) - set(expected))
        if missing or extra:
            raise HTTPException(status_code=422, detail={"row": i, "missing": missing, "extra": extra})
        matrix.append([row[name] for name in expected])

    x = np.array(matrix, dtype=np.float64)
    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="Features must be finite numbers")

    preds = model.predict(x)
    probas = model.predict_proba(x)
    classes = [int(c) for c in model.named_steps["svc"].classes_]

    results = []
    for pred, proba in zip(preds, probas):
        p = {c: float(v) for c, v in zip(classes, proba)}
        results.append({
            "predicted_class": int(pred),
            "predicted_label": metadata["class_mapping"][str(int(pred))],
            "probability_malignant": p.get(0, 0.0),
            "probability_benign": p.get(1, 0.0),
        })
    return {"results": results, "model_version": metadata["model_version"]}
