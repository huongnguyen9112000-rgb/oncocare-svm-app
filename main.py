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
    description="Educational demonstration only; not a medical diagnosis."
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

@app.get("/", response_class=HTMLResponse)
def home_page():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OncoCare AI - Trung Tâm Phân Tích Khối U</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background-color: #030712; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #f8fafc; }
            .sidebar { min-height: 100vh; background: #050b14; border-right: 1px solid #1e293b; position: fixed; width: 260px; top: 0; left: 0; z-index: 100; }
            .sidebar .brand { padding: 20px; font-size: 1.2rem; font-weight: bold; background: #03060c; border-bottom: 1px solid #1e293b; color: #06b6d4; text-align: center; }
            .sidebar .nav-link { color: #94a3b8; padding: 12px 20px; margin: 5px 15px; border-radius: 8px; cursor: pointer; transition: 0.2s; }
            .sidebar .nav-link:hover, .sidebar .nav-link.active { background: rgba(6, 182, 212, 0.15); color: #06b6d4; }
            .main-content { margin-left: 260px; padding: 30px; }
            .card { border: 1px solid #1e293b; border-radius: 12px; background: #0b1120; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
            .tab-pane-content { display: none; }
            .tab-pane-content.active { display: block; }
            .feature-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
            .feature-card .form-control { background-color: #030712; border-color: #334155; color: #fff; font-weight: bold; }
            .feature-card .form-control:focus { border-color: #06b6d4; background-color: #030712; color: #fff; box-shadow: none; }
            .btn-cyber { background: #06b6d4; color: #030712; font-weight: bold; border: none; }
            .btn-cyber:hover { background: #0891b2; color: #fff; }
        </style>
    </head>
    <body>
        <div class="sidebar d-flex flex-column">
            <div class="brand"><i class="fa-solid fa-microchip"></i> OncoCare AI</div>
            <ul class="nav flex-column mt-3">
                <li class="nav-item"><a class="nav-link active" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-brain"></i> Chẩn Đoán AI</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-history', event)"><i class="fa-solid fa-clock-rotate-left"></i> Lịch Sử Khám</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-stats', event)"><i class="fa-solid fa-chart-pie"></i> Thống Kê</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-guide', event)"><i class="fa-solid fa-book-medical"></i> Hướng Dẫn</a></li>
            </ul>
        </div>

        <div class="main-content">
            <div id="tab-predict" class="tab-pane-content active">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h2>Phân Tích Khối U SVM</h2>
                    <div>
                        <button type="button" class="btn btn-outline-success btn-sm me-2" onclick="loadSampleData('benign')">Mẫu Lành Tính</button>
                        <button type="button" class="btn btn-outline-danger btn-sm" onclick="loadSampleData('malignant')">Mẫu Ác Tính</button>
                    </div>
                </div>

                <div class="card p-4 mb-4">
                    <form id="predictionForm">
                        <h5 class="text-info mb-3">30 Chỉ Số Tế Bào Lâm Sàng</h5>
                        <div class="row g-3" id="featureInputs"></div>
                        <div class="mt-4 text-center">
                            <button type="submit" id="btnSubmit" class="btn btn-cyber btn-lg px-5">Thực Hiện Chẩn Đoán</button>
                        </div>
                    </form>
                </div>

                <div class="card p-4 mb-4" id="resultCard" style="display: none;">
                    <h4 class="mb-3">Kết Quả Chẩn Đoán</h4>
                    <div id="resultAlert" class="alert mb-3" role="alert"></div>
                    <div class="row">
                        <div class="col-md-6 border-end border-secondary">
                            <p class="text-muted mb-1">Kết luận:</p>
                            <h3 id="resPrediction" class="fw-bold"></h3>
                        </div>
                        <div class="col-md-6 ps-md-4">
                            <div class="mb-2">
                                <div class="d-flex justify-content-between mb-1"><span>Lành tính:</span><span id="resBenignText" class="text-success fw-bold"></span></div>
                                <div class="progress" style="height: 10px; background: #030712;"><div id="barBenign" class="progress-bar bg-success" style="width: 0%"></div></div>
                            </div>
                            <div>
                                <div class="d-flex justify-content-between mb-1"><span>Ác tính:</span><span id="resMalignantText" class="text-danger fw-bold"></span></div>
                                <div class="progress" style="height: 10px; background: #030712;"><div id="barMalignant" class="progress-bar bg-danger" style="width: 0%"></div></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="tab-history" class="tab-pane-content">
                <h2>Lịch Sử Khám</h2>
                <div class="card p-4 mt-3">
                    <table class="table table-dark table-hover align-middle">
                        <thead><tr><th>Mã Ca</th><th>Thời Gian</th><th>Kết Luận</th><th>Xác Suất Ác Tính</th><th>Thao Tác</th></tr></thead>
                        <tbody id="historyTableBody"><tr><td colspan="5" class="text-center text-muted">Chưa có lịch sử.</td></tr></tbody>
                    </table>
                </div>
            </div>

            <div id="tab-stats" class="tab-pane-content">
                <h2>Thống Kê Hệ Thống</h2>
                <div class="row g-4 mt-2">
                    <div class="col-md-4"><div class="card p-3 text-center"><h6>Tổng ca</h6><h2 id="statTotal" class="text-info">0</h2></div></div>
                    <div class="col-md-4"><div class="card p-3 text-center"><h6>Lành tính</h6><h2 id="statBenign" class="text-success">0</h2></div></div>
                    <div class="col-md-4"><div class="card p-3 text-center"><h6>Ác tính</h6><h2 id="statMalignant" class="text-danger">0</h2></div></div>
                </div>
            </div>

            <div id="tab-guide" class="tab-pane-content">
                <h2>Hướng Dẫn Chỉ Số</h2>
                <div class="card p-4 mt-3"><div class="row g-3" id="guideContainer"></div></div>
            </div>
        </div>

        <script>
            const featureMap = {
                "mean radius": "Bán kính TB", "mean texture": "Kết cấu TB", "mean perimeter": "Chu vi TB", "mean area": "Diện tích TB",
                "mean smoothness": "Độ nhẵn TB", "mean compactness": "Độ nén TB", "mean concavity": "Độ lõm TB", "mean concave points": "Điểm lõm TB",
                "mean symmetry": "Độ đối xứng TB", "mean fractal dimension": "Chiều Fractal TB", "radius error": "Sai số bán kính", "texture error": "Sai số kết cấu",
                "perimeter error": "Sai số chu vi", "area error": "Sai số diện tích", "smoothness error": "Sai số độ nhẵn", "compactness error": "Sai số độ nén",
                "concavity error": "Sai số độ lõm", "concave points error": "Sai số điểm lõm", "symmetry error": "Sai số đối xứng", "fractal dimension error": "Sai số Fractal",
                "worst radius": "Bán kính lớn nhất", "worst texture": "Kết cấu lớn nhất", "worst perimeter": "Chu vi lớn nhất", "worst area": "Diện tích lớn nhất",
                "worst smoothness": "Độ nhẵn lớn nhất", "worst compactness": "Độ nén lớn nhất", "worst concavity": "Độ lõm lớn nhất", "worst concave points": "Điểm lõm lớn nhất",
                "worst symmetry": "Đối xứng lớn nhất", "worst fractal dimension": "Chiều Fractal lớn nhất"
            };
            const sampleMalignant = {"mean radius":17.99,"mean texture":10.38,"mean perimeter":122.8,"mean area":1001.0,"mean smoothness":0.1184,"mean compactness":0.2776,"mean concavity":0.3001,"mean concave points":0.1471,"mean symmetry":0.2419,"mean fractal dimension":0.07871,"radius error":1.095,"texture error":0.9053,"perimeter error":8.589,"area error":153.4,"smoothness error":0.006399,"compactness error":0.04904,"concavity error":0.05373,"concave points error":0.01587,"symmetry error":0.03003,"fractal dimension error":0.006193,"worst radius":25.38,"worst texture":17.33,"worst perimeter":184.6,"worst area":2019.0,"worst smoothness":0.1622,"worst compactness":0.6656,"worst concavity":0.7119,"worst concave points":0.2654,"worst symmetry":0.4601,"worst fractal dimension":0.1189};
            const sampleBenign = {"mean radius":13.54,"mean texture":14.36,"mean perimeter":87.46,"mean area":566.3,"mean smoothness":0.09779,"mean compactness":0.08129,"mean concavity":0.06664,"mean concave points":0.04781,"mean symmetry":0.1885,"mean fractal dimension":0.05766,"radius error":0.2699,"texture error":0.7886,"perimeter error":2.058,"area error":23.56,"smoothness error":0.008462,"compactness error":0.0146,"concavity error":0.02387,"concave points error":0.01315,"symmetry error":0.0198,"fractal dimension error":0.0023,"worst radius":15.11,"worst texture":19.26,"worst perimeter":99.7,"worst area":711.2,"worst smoothness":0.144,"worst compactness":0.1773,"worst concavity":0.239,"worst concave points":0.1288,"worst symmetry":0.2977,"worst fractal dimension":0.07259};

            let featureNames = [], history = [], stats = { total: 0, benign: 0, malignant: 0 };

            async function init() {
                const res = await fetch('/metadata');
                const data = await res.json();
                featureNames = data.feature_names;
                renderInputs();
                renderGuide();
            }

            function renderInputs() {
                const c = document.getElementById('featureInputs');
                c.innerHTML = '';
                featureNames.forEach(f => {
                    c.innerHTML += `<div class="col-md-4"><div class="feature-card"><label class="form-label small">${featureMap[f] || f}</label><input type="text" class="form-control form-control-sm" name="${f}" id="inp_${f.replace(/\s+/g, '_')}" value="${sampleBenign[f]}"></div></div>`;
                });
            }

            function renderGuide() {
                const g = document.getElementById('guideContainer');
                g.innerHTML = '';
                Object.keys(featureMap).forEach((k, i) => {
                    g.innerHTML += `<div class="col-md-6"><div class="p-3 border border-secondary rounded bg-dark"><h6 class="text-info">${i+1}. ${featureMap[k]}</h6><code>${k}</code></div></div>`;
                });
            }

            function loadSampleData(type) {
                const data = type === 'malignant' ? sampleMalignant : sampleBenign;
                featureNames.forEach(f => {
                    const el = document.getElementById(`inp_${f.replace(/\s+/g, '_')}`);
                    if (el) el.value = data[f];
                });
            }

            function switchTab(id, e) {
                if(e) e.preventDefault();
                document.querySelectorAll('.tab-pane-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.sidebar .nav-link').forEach(el => el.classList.remove('active'));
                document.getElementById(id).classList.add('active');
                if(e && e.currentTarget) e.currentTarget.classList.add('active');
            }

            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const btn = document.getElementById('btnSubmit');
                btn.disabled = true; btn.innerText = 'Đang phân tích...';
                const fd = new FormData(this);
                const obj = {};
                fd.forEach((v, k) => obj[k] = parseFloat(v.replace(',', '.')));
                try {
                    const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({features: obj}) });
                    const data = await res.json();
                    document.getElementById('resultCard').style.display = 'block';
                    const pB = data.probability_benign * 100, pM = data.probability_malignant * 100;
                    document.getElementById('barBenign').style.width = pB + '%';
                    document.getElementById('barMalignant').style.width = pM + '%';
                    document.getElementById('resBenignText.').innerText = pB.toFixed(2) + '%';
                    document.getElementById('resMalignantText').innerText = pM.toFixed(2) + '%';
                    
                    stats.total++;
                    let rec = { id: 'CA-' + Math.floor(1000 + Math.random() * 9000), time: new Date().toLocaleTimeString(), prob: pM.toFixed(2) + '%' };
                    if(data.predicted_label === 'benign') {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-success">LÀNH TÍNH (BENIGN)</span>';
                        document.getElementById('resultAlert').className = 'alert alert-success border-0 bg-success bg-opacity-10 text-success fw-bold';
                        document.getElementById('resultAlert').innerHTML = 'Khối u lành tính, không có dấu hiệu ác tính.';
                        stats.benign++; rec.label = '<span class="badge bg-success">Lành tính</span>';
                    } else {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-danger">ÁC TÍNH (MALIGNANT)</span>';
                        document.getElementById('resultAlert').className = 'alert alert-danger border-0 bg-danger bg-opacity-10 text-danger fw-bold';
                        document.getElementById('resultAlert').innerHTML = 'CẢNH BÁO: Khối u có xu hướng ác tính cao!';
                        stats.malignant++; rec.label = '<span class="badge bg-danger">Ác tính</span>';
                    }
                    document.getElementById('statTotal').innerText = stats.total;
                    document.getElementById('statBenign').innerText = stats.benign;
                    document.getElementById('statMalignant').innerText = stats.malignant;
                    history.unshift(rec);
                    let tb = document.getElementById('historyTableBody');
                    tb.innerHTML = history.map(h => `<tr><td><code>${h.id}</code></td><td>${h.time}</td><td>${h.label}</td><td class="fw-bold">${h.prob}</td><td><button class="btn btn-sm btn-outline-info" onclick="switchTab('tab-predict', event)">Xem</button></td></tr>`).join('');
                } finally {
                    btn.disabled = false; btn.innerText = 'Thực Hiện Chẩn Đoán';
                }
            });
            init();
        </script>
    </body>
    </html>
    """

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
