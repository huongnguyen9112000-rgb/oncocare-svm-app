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
        <title>OncoCare AI - Neural Bio-Tech Core</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-deep: #02040a;
                --bg-panel: #0a0f1d;
                --neon-cyan: #00f0ff;
                --neon-green: #00ff66;
                --neon-rose: #ff0055;
                --border-cyber: rgba(0, 240, 255, 0.2);
            }
            body { background-color: var(--bg-deep); font-family: 'Inter', sans-serif; color: #e2e8f0; overflow-x: hidden; }
            
            /* Sidebar phong cách trạm không gian y tế */
            .sidebar { min-height: 100vh; background: #050811; border-right: 1px solid var(--border-cyber); position: fixed; width: 270px; top: 0; left: 0; z-index: 100; box-shadow: 5px 0 30px rgba(0, 240, 255, 0.03); }
            .sidebar .brand { padding: 25px; font-family: 'Orbitron', sans-serif; font-size: 1.25rem; font-weight: 900; background: #02040a; border-bottom: 1px solid var(--border-cyber); color: var(--neon-cyan); letter-spacing: 1px; text-shadow: 0 0 10px rgba(0, 240, 255, 0.5); }
            .sidebar .nav-link { color: #8a99ad; padding: 14px 20px; margin: 8px 15px; border-radius: 10px; font-weight: 600; cursor: pointer; transition: all 0.3s; border: 1px solid transparent; }
            .sidebar .nav-link:hover, .sidebar .nav-link.active { background: rgba(0, 240, 255, 0.08); color: var(--neon-cyan); border-color: var(--border-cyber); box-shadow: 0 0 15px rgba(0, 240, 255, 0.1); transform: translateX(5px); }
            
            /* Main Content */
            .main-content { margin-left: 270px; padding: 40px; }
            .card { border: 1px solid var(--border-cyber); border-radius: 16px; background: var(--bg-panel); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px); }
            .tab-pane-content { display: none; animation: fadeIn 0.4s ease; }
            .tab-pane-content.active { display: block; }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

            /* Feature inputs cyber */
            .feature-card { background: #070b17; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; transition: 0.2s; }
            .feature-card:hover { border-color: var(--neon-cyan); box-shadow: 0 0 10px rgba(0, 240, 255, 0.15); }
            .feature-card .form-control { background-color: #02040a; border-color: #1e293b; color: var(--neon-cyan); font-family: 'Orbitron', sans-serif; font-size: 0.9rem; font-weight: 700; }
            .feature-card .form-control:focus { border-color: var(--neon-cyan); background-color: #02040a; color: var(--neon-cyan); box-shadow: 0 0 10px rgba(0, 240, 255, 0.3); }

            /* Buttons */
            .btn-cyber { background: linear-gradient(135deg, #00f0ff, #0077ff); color: #02040a; font-family: 'Orbitron', sans-serif; font-weight: 900; border: none; letter-spacing: 1px; box-shadow: 0 0 20px rgba(0, 240, 255, 0.4); transition: 0.3s; }
            .btn-cyber:hover { background: #fff; color: #02040a; box-shadow: 0 0 30px rgba(0, 240, 255, 0.8); transform: scale(1.02); }

            /* Result Box */
            .result-hud { background: linear-gradient(135deg, #0a0f1d, #04070e); border: 1px solid var(--neon-cyan); border-radius: 20px; box-shadow: 0 0 40px rgba(0, 240, 255, 0.15); }
            .progress { background-color: #02040a; border: 1px solid #1e293b; border-radius: 8px; height: 14px; padding: 2px; }
            .progress-bar { border-radius: 6px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }
            
            /* Table */
            .table-cyber th { background: #050811; color: var(--neon-cyan); border-bottom: 1px solid var(--border-cyber); font-family: 'Orbitron', sans-serif; font-size: 0.75rem; letter-spacing: 1px; }
            .table-cyber td { border-bottom: 1px solid #111827; color: #cbd5e1; vertical-align: middle; }
        </style>
    </head>
    <body>

        <div class="sidebar d-flex flex-column">
            <div class="brand"><i class="fa-solid fa-dna text-cyan me-2" style="color: var(--neon-cyan)"></i> ONCOCARE AI</div>
            <ul class="nav flex-column mt-4">
                <li class="nav-item"><a class="nav-link active" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-microchip me-2"></i> Chẩn Đoán AI</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-history', event)"><i class="fa-solid fa-clock-rotate-left me-2"></i> Lịch Sử Khám</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-stats', event)"><i class="fa-solid fa-chart-line me-2"></i> Thống Kê Hệ Thống</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-guide', event)"><i class="fa-solid fa-book-medical me-2"></i> Hướng Dẫn Chỉ Số</a></li>
            </ul>
            <div class="mt-auto p-4 text-muted small border-top border-dark">
                <div class="text-cyan fw-bold mb-1" style="color: var(--neon-cyan)"><i class="fa-solid fa-shield-halved me-1"></i> Neural Protected</div>
                <div style="font-size: 0.75rem;">SVM Engine v1.0.0</div>
            </div>
        </div>

        <div class="main-content">
            <!-- TAB 1: CHẨN ĐOÁN -->
            <div id="tab-predict" class="tab-pane-content active">
                <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
                    <div>
                        <h1 class="fw-bold text-white mb-1" style="font-family: 'Orbitron'; font-size: 1.5rem;"><i class="fa-solid fa-network-wired text-cyan me-2" style="color: var(--neon-cyan)"></i> HỆ THỐNG PHÂN TÍCH SVM THÔNG MINH</h1>
                        <p class="text-muted mb-0">Trạm phân tích hình thái học tế bào u vú thời gian thực bằng mô hình học máy.</p>
                    </div>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-success btn-sm px-3 rounded-pill fw-bold" onclick="loadSampleData('benign')"><i class="fa-solid fa-shield-heart me-1"></i> Mẫu Lành Tính</button>
                        <button type="button" class="btn btn-outline-danger btn-sm px-3 rounded-pill fw-bold" onclick="loadSampleData('malignant')"><i class="fa-solid fa-biohazard me-1"></i> Mẫu Ác Tính</button>
                    </div>
                </div>

                <div class="card p-4 mb-4">
                    <form id="predictionForm">
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <h5 class="text-cyan fw-bold mb-0" style="color: var(--neon-cyan); font-family: 'Orbitron'; font-size: 1rem;"><i class="fa-solid fa-sliders me-2"></i> 30 Chỉ Số Thông Số Tế Bào</h5>
                            <span class="badge bg-dark border border-cyan text-cyan px-3 py-2 rounded-pill font-monospace">CHUẨN SVM DATASET</span>
                        </div>
                        <div class="row g-3" id="featureInputs"></div>
                        <div class="mt-5 text-center">
                            <button type="submit" id="btnSubmit" class="btn btn-cyber btn-lg px-5 py-3 rounded-pill">
                                <i class="fa-solid fa-microchip me-2"></i> THỰC HIỆN CHẨN ĐOÁN AI
                            </button>
                        </div>
                    </form>
                </div>

                <!-- KẾT QUẢ HUD -->
                <div class="result-hud p-4 p-md-5 mb-5" id="resultCard" style="display: none;">
                    <div class="d-flex align-items-center justify-content-between mb-4 pb-3 border-bottom border-secondary">
                        <h3 class="fw-bold text-white mb-0" style="font-family: 'Orbitron'; font-size: 1.2rem;"><i class="fa-solid fa-file-waveform text-cyan me-2" style="color: var(--neon-cyan)"></i> BÁO CÁO KẾT QUẢ PHÂN TÍCH</h3>
                        <span class="badge bg-dark text-cyan border border-cyan px-3 py-2 rounded-pill font-monospace">STATUS: SECURED</span>
                    </div>
                    
                    <div id="resultAlert" class="alert mb-4 py-3 rounded-4 fw-bold border-0 shadow-sm" role="alert"></div>
                    
                    <div class="row align-items-center g-4">
                        <div class="col-lg-5 border-end border-secondary pe-lg-4">
                            <span class="text-muted text-uppercase small fw-bold tracking-wider">Kết luận mô hình:</span>
                            <h2 id="resPrediction" class="fw-bold my-2 display-6" style="font-family: 'Orbitron';"></h2>
                            <p class="small text-muted mb-0 mt-3 p-3 rounded-3" style="background: #03060d; border: 1px solid #1e293b;">
                                <i class="fa-solid fa-circle-info text-cyan me-1" style="color: var(--neon-cyan)"></i> 
                                <strong>Lưu ý y tế:</strong> Kết quả thuật toán hỗ trợ tham khảo lâm sàng, không thay thế chẩn đoán chuyên khoa chính thức.
                            </p>
                        </div>
                        <div class="col-lg-7 ps-lg-4">
                            <div class="mb-3 p-3 rounded-4" style="background: #050811; border: 1px solid #1e293b;">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="fw-bold text-muted">Xác suất Lành tính (Benign):</span>
                                    <span id="resBenignText" class="text-success fw-bold fs-5 font-monospace">0%</span>
                                </div>
                                <div class="progress">
                                    <div id="barBenign" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="p-3 rounded-4" style="background: #050811; border: 1px solid #1e293b;">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="fw-bold text-muted">Xác suất Ác tính (Malignant):</span>
                                    <span id="resMalignantText" class="text-danger fw-bold fs-5 font-monospace">0%</span>
                                </div>
                                <div class="progress">
                                    <div id="barMalignant" class="progress-bar bg-danger" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 2: LỊCH SỬ -->
            <div id="tab-history" class="tab-pane-content">
                <h2 class="fw-bold text-white mb-1" style="font-family: 'Orbitron'; font-size: 1.5rem;">NHẬT KÝ CA KHÁM</h2>
                <p class="text-muted mb-4">Lưu trữ toàn bộ các ca phân tích trong phiên làm việc hiện tại.</p>
                <div class="card p-4">
                    <div class="table-responsive">
                        <table class="table table-cyber align-middle mb-0">
                            <thead>
                                <tr>
                                    <th>Mã Ca</th>
                                    <th>Thời Gian</th>
                                    <th>Kết Luận</th>
                                    <th>Xác Suất Ác Tính</th>
                                    <th>Thao Tác</th>
                                </tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <tr><td colspan="5" class="text-center text-muted py-5">Chưa có lịch sử chẩn đoán nào được ghi nhận.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- TAB 3: THỐNG KÊ -->
            <div id="tab-stats" class="tab-pane-content">
                <h2 class="fw-bold text-white mb-1" style="font-family: 'Orbitron'; font-size: 1.5rem;">THỐNG KÊ HỆ THỐNG</h2>
                <p class="text-muted mb-4">Tổng quan số liệu phân tích và cấu hình mô hình SVM.</p>
                <div class="row g-4 mb-4">
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-cyan border-4">
                            <span class="text-muted fw-bold small text-uppercase">Tổng Số Ca</span>
                            <h2 id="statTotal" class="fw-bold mt-2 display-5 font-monospace text-cyan" style="color: var(--neon-cyan); font-family: 'Orbitron';">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-success border-4">
                            <span class="text-muted fw-bold small text-uppercase">Lành Tính</span>
                            <h2 id="statBenign" class="text-success fw-bold mt-2 display-5 font-monospace" style="font-family: 'Orbitron';">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-danger border-4">
                            <span class="text-muted fw-bold small text-uppercase">Ác Tính</span>
                            <h2 id="statMalignant" class="text-danger fw-bold mt-2 display-5 font-monospace" style="font-family: 'Orbitron';">0</h2>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 4: HƯỚNG DẪN -->
            <div id="tab-guide" class="tab-pane-content">
                <h2 class="fw-bold text-white mb-1" style="font-family: 'Orbitron'; font-size: 1.5rem;">HƯỚNG DẪN 30 CHỈ SỐ</h2>
                <p class="text-muted mb-4">Giải thích chi tiết các chỉ số đặc trưng tế bào học.</p>
                <div class="card p-4">
                    <div class="row g-3" id="guideContainer"></div>
                </div>
            </div>
        </div>

        <script>
            const featureMap = {
                "mean radius": "Bán kính trung bình", "mean texture": "Độ mịn/Kết cấu TB", "mean perimeter": "Chu vi trung bình", "mean area": "Diện tích trung bình",
                "mean smoothness": "Độ trơn nhẵn TB", "mean compactness": "Độ nén trung bình", "mean concavity": "Độ lõm trung bình", "mean concave points": "Số điểm lõm TB",
                "mean symmetry": "Độ đối xứng TB", "mean fractal dimension": "Chiều Fractal TB", "radius error": "Sai số bán kính", "texture error": "Sai số kết cấu",
                "perimeter error": "Sai số chu vi", "area error": "Sai số diện tích", "smoothness error": "Sai số độ trơn nhẵn", "compactness error": "Sai số độ nén",
                "concavity error": "Sai số độ lõm", "concave points error": "Sai số điểm lõm", "symmetry error": "Sai số đối xứng", "fractal dimension error": "Sai số Fractal",
                "worst radius": "Bán kính lớn nhất", "worst texture": "Kết cấu lớn nhất", "worst perimeter": "Chu vi lớn nhất", "worst area": "Diện tích lớn nhất",
                "worst smoothness": "Độ nhẵn lớn nhất", "worst compactness": "Độ nén lớn nhất", "worst concavity": "Độ lõm lớn nhất", "worst concave points": "Điểm lõm lớn nhất",
                "worst symmetry": "Đối xứng lớn nhất", "worst fractal dimension": "Chiều Fractal lớn nhất"
            };
            const sampleMalignant = {"mean radius":17.99,"mean texture":10.38,"mean perimeter":122.8,"mean area":1001.0,"mean smoothness":0.1184,"mean compactness":0.2776,"mean concavity":0.3001,"mean concave points":0.1471,"mean symmetry":0.2419,"mean fractal dimension":0.07871,"radius error":1.095,"texture error":0.9053,"perimeter error":8.589,"area error":153.4,"smoothness error":0.006399,"compactness error":0.04904,"concavity error":0.05373,"concave points error":0.01587,"symmetry error":0.03003,"fractal dimension error":0.006193,"worst radius":25.38,"worst texture":17.33,"worst perimeter":184.6,"worst area":2019.0,"worst smoothness":0.1622,"worst compactness":0.6656,"worst concavity":0.7119,"worst concave points":0.2654,"worst symmetry":0.4601,"worst fractal dimension":0.1189};
            const sampleBenign = {"mean radius":13.54,"mean texture":14.36,"mean perimeter":87.46,"mean area":566.3,"mean smoothness":0.09779,"mean compactness":0.08129,"mean concavity":0.06664,"mean concave points":0.04781,"mean symmetry":0.1885,"mean fractal dimension":0.05766,"radius error":0.2699,"texture error":0.7886,"perimeter error":2.058,"area error":23.56,"smoothness error":0.008462,"compactness error":0.0146,"concavity error":0.02387,"concave points error":0.01315,"symmetry error":0.0198,"fractal dimension error":0.0023,"worst radius":15.11,"worst texture":19.26,"worst perimeter":99.7,"worst area":711.2,"worst smoothness":0.144,"worst compactness":0.1773,"worst concavity":0.239,"worst concave points":0.1288,"worst symmetry":0.2977,"worst fractal dimension":0.07259};

            let featureNames = [], historyList = [], stats = { total: 0, benign: 0, malignant: 0 };

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
                    c.innerHTML += `<div class="col-md-4"><div class="feature-card"><label class="form-label text-light small fw-bold">${featureMap[f] || f}</label><input type="text" class="form-control form-control-sm" name="${f}" id="inp_${f.replace(/\s+/g, '_')}" value="${sampleBenign[f]}"></div></div>`;
                });
            }

            function renderGuide() {
                const g = document.getElementById('guideContainer');
                g.innerHTML = '';
                Object.keys(featureMap).forEach((k, i) => {
                    g.innerHTML += `<div class="col-md-6"><div class="p-3 border border-secondary rounded bg-dark"><h6 class="text-cyan fw-bold" style="color:var(--neon-cyan)">${i+1}. ${featureMap[k]}</h6><code class="text-muted">${k}</code></div></div>`;
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
                btn.disabled = true; btn.innerText = 'ĐANG XỬ LÝ NEURAL...';
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
                    document.getElementById('resBenignText').innerText = pB.toFixed(2) + '%';
                    document.getElementById('resMalignantText').innerText = pM.toFixed(2) + '%';
                    
                    stats.total++;
                    let rec = { id: 'CA-' + Math.floor(1000 + Math.random() * 9000), time: new Date().toLocaleTimeString(), prob: pM.toFixed(2) + '%' };
                    if(data.predicted_label === 'benign') {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-success">LÀNH TÍNH (BENIGN)</span>';
                        document.getElementById('resultAlert').className = 'alert alert-success border-0 fw-bold';
                        document.getElementById('resultAlert').style.background = 'rgba(0, 255, 102, 0.1)';
                        document.getElementById('resultAlert').style.color = '#00ff66';
                        document.getElementById('resultAlert').innerHTML = '<i class="fa-solid fa-shield-heart me-2"></i> Khối u lành tính, không phát hiện dấu hiệu ác tính đáng kể.';
                        stats.benign++; rec.label = '<span class="badge bg-success">Lành tính</span>';
                    } else {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-danger">ÁC TÍNH (MALIGNANT)</span>';
                        document.getElementById('resultAlert').className = 'alert alert-danger border-0 fw-bold';
                        document.getElementById('resultAlert').style.background = 'rgba(255, 0, 85, 0.1)';
                        document.getElementById('resultAlert').style.color = '#ff0055';
                        document.getElementById('resultAlert').innerHTML = '<i class="fa-solid fa-triangle-exclamation me-2"></i> CẢNH BÁO: Khối u có xu hướng ác tính cao! Khuyến nghị hội chẩn sinh thiết.';
                        stats.malignant++; rec.label = '<span class="badge bg-danger">Ác tính</span>';
                    }
                    document.getElementById('statTotal').innerText = stats.total;
                    document.getElementById('statBenign').innerText = stats.benign;
                    document.getElementById('statMalignant').innerText = stats.malignant;
                    
                    historyList.unshift(rec);
                    let tb = document.getElementById('historyTableBody');
                    tb.innerHTML = historyList.map(h => `<tr><td><code class="text-cyan" style="color:var(--neon-cyan)">${h.id}</code></td><td>${h.time}</td><td>${h.label}</td><td class="fw-bold">${h.prob}</td><td><button class="btn btn-sm btn-outline-info rounded-pill px-3" onclick="switchTab('tab-predict', event)">Xem</button></td></tr>`).join('');
                } finally {
                    btn.disabled = false; btn.innerText = 'THỰC HIỆN CHẨN ĐOÁN AI';
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
