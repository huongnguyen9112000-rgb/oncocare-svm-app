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
        <title>OncoCare AI - Hệ Thống Y Tế Thông Minh</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-app: #f4f7f6;
                --sidebar-bg: #ffffff;
                --primary-teal: #0d9488;
                --primary-teal-hover: #0f766e;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-muted: #64748b;
                --border-color: #e2e8f0;
            }
            body { background-color: var(--bg-app); font-family: 'Inter', sans-serif; color: var(--text-main); }
            
            /* Sidebar chuẩn y tế sáng sủa */
            .sidebar { min-height: 100vh; background: var(--sidebar-bg); border-right: 1px solid var(--border-color); position: fixed; width: 270px; top: 0; left: 0; z-index: 100; box-shadow: 4px 0 20px rgba(0, 0, 0, 0.02); }
            .sidebar .brand { padding: 24px; font-size: 1.2rem; font-weight: 700; background: #ffffff; border-bottom: 1px solid var(--border-color); color: var(--primary-teal); display: flex; align-items: center; gap: 10px; }
            .sidebar .nav-link { color: var(--text-muted); padding: 12px 20px; margin: 6px 15px; border-radius: 10px; font-weight: 600; cursor: pointer; transition: 0.2s; display: flex; align-items: center; gap: 12px; }
            .sidebar .nav-link:hover, .sidebar .nav-link.active { background: #f0fdf4; color: var(--primary-teal); }
            .sidebar .nav-link.active { background: #ccfbf1; color: #115e59; }
            
            /* Main Content */
            .main-content { margin-left: 270px; padding: 40px; }
            .card { border: 1px solid var(--border-color); border-radius: 16px; background: var(--card-bg); box-shadow: 0 4px 20px rgba(0,0,0,0.03); }
            .tab-pane-content { display: none; animation: fadeIn 0.3s ease; }
            .tab-pane-content.active { display: block; }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

            /* Inputs */
            .feature-card { background: #f8fafc; border: 1px solid var(--border-color); border-radius: 10px; padding: 14px; transition: 0.2s; }
            .feature-card:hover { border-color: var(--primary-teal); background: #ffffff; box-shadow: 0 4px 12px rgba(13, 148, 136, 0.08); }
            .feature-card .form-control { background-color: #ffffff; border-color: #cbd5e1; color: #0f172a; font-weight: 600; }
            .feature-card .form-control:focus { border-color: var(--primary-teal); box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.15); background-color: #ffffff; }

            /* Buttons */
            .btn-medical { background: var(--primary-teal); color: #fff; font-weight: 600; border: none; border-radius: 10px; padding: 12px 28px; transition: 0.2s; box-shadow: 0 4px 12px rgba(13, 148, 136, 0.25); }
            .btn-medical:hover { background: var(--primary-teal-hover); color: #fff; transform: translateY(-1px); }

            /* Result Box */
            .result-box { background: #ffffff; border: 1px solid var(--border-color); border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
            .progress { background-color: #f1f5f9; border-radius: 8px; height: 12px; padding: 2px; }
            .progress-bar { border-radius: 6px; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }
            
            /* Table */
            .table-medical th { background: #f8fafc; color: var(--text-muted); font-weight: 700; text-transform: uppercase; font-size: 0.75rem; border-bottom: 2px solid var(--border-color); }
            .table-medical td { border-bottom: 1px solid #f1f5f9; color: var(--text-main); vertical-align: middle; font-weight: 500; }
        </style>
    </head>
    <body>

        <div class="sidebar d-flex flex-column">
            <div class="brand"><i class="fa-solid fa-notes-medical fs-4"></i> OncoCare AI</div>
            <ul class="nav flex-column mt-3">
                <li class="nav-item"><a class="nav-link active" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-stethoscope"></i> Chẩn Đoán AI</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-history', event)"><i class="fa-solid fa-clock-rotate-left"></i> Lịch Sử Khám</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-stats', event)"><i class="fa-solid fa-chart-pie"></i> Thống Kê Hệ Thống</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-guide', event)"><i class="fa-solid fa-book-medical"></i> Hướng Dẫn Chỉ Số</a></li>
            </ul>
            <div class="mt-auto p-4 text-muted small border-top border-light">
                <div class="fw-bold text-dark mb-1">Hệ Thống SVM Y Tế</div>
                <div style="font-size: 0.75rem;">Phiên bản chuẩn FDA v1.0</div>
            </div>
        </div>

        <div class="main-content">
            <!-- TAB 1: CHẨN ĐOÁN -->
            <div id="tab-predict" class="tab-pane-content active">
                <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom">
                    <div>
                        <h1 class="fw-bold text-dark mb-1 fs-3">Phân Tích & Chẩn Đoán Khối U Vú</h1>
                        <p class="text-muted mb-0">Hệ thống hỗ trợ lâm sàng ứng dụng mô hình SVM tiêu chuẩn.</p>
                    </div>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-success btn-sm px-3 rounded-pill fw-semibold shadow-sm" onclick="loadSampleData('benign')"><i class="fa-solid fa-shield-halved me-1"></i> Mẫu Lành Tính</button>
                        <button type="button" class="btn btn-outline-danger btn-sm px-3 rounded-pill fw-semibold shadow-sm" onclick="loadSampleData('malignant')"><i class="fa-solid fa-biohazard me-1"></i> Mẫu Ác Tính</button>
                    </div>
                </div>

                <div class="card p-4 mb-4">
                    <form id="predictionForm">
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <h5 class="fw-bold text-dark mb-0"><i class="fa-solid fa-microscope text-teal me-2" style="color:var(--primary-teal)"></i> 30 Chỉ Số Hình Thái Học Tế Bào</h5>
                            <span class="badge bg-light text-secondary border px-3 py-2 rounded-pill">Định dạng chuẩn</span>
                        </div>
                        <div class="row g-3" id="featureInputs"></div>
                        <div class="mt-5 text-center">
                            <button type="submit" id="btnSubmit" class="btn btn-medical btn-lg px-5 py-3 rounded-pill">
                                <i class="fa-solid fa-calculator me-2"></i> Thực Hiện Chẩn Đoán AI
                            </button>
                        </div>
                    </form>
                </div>

                <!-- KẾT QUẢ -->
                <div class="result-box p-4 p-md-5 mb-5" id="resultCard" style="display: none;">
                    <div class="d-flex align-items-center justify-content-between mb-4 pb-3 border-bottom">
                        <h3 class="fw-bold text-dark mb-0 fs-4"><i class="fa-solid fa-file-medical text-teal me-2" style="color:var(--primary-teal)"></i> Kết Quả Phân Tích Chuyên Sâu</h3>
                        <span class="badge bg-light text-secondary border px-3 py-2 rounded-pill font-monospace">ID: #DIAG-2026</span>
                    </div>
                    
                    <div id="resultAlert" class="alert mb-4 py-3 rounded-4 fw-semibold border-0 shadow-sm" role="alert"></div>
                    
                    <div class="row align-items-center g-4">
                        <div class="col-lg-5 border-end pe-lg-4">
                            <span class="text-muted text-uppercase small fw-bold">Kết luận mô hình:</span>
                            <h2 id="resPrediction" class="fw-bold my-2 display-6"></h2>
                            <p class="small text-muted mb-0 mt-3 p-3 rounded-3 bg-light border">
                                <i class="fa-solid fa-circle-info text-teal me-1" style="color:var(--primary-teal)"></i> 
                                <strong>Lưu ý:</strong> Kết quả mang tính chất tham khảo lâm sàng hỗ trợ bác sĩ chuyên khoa, không thay thế chẩn đoán chính thức.
                            </p>
                        </div>
                        <div class="col-lg-7 ps-lg-4">
                            <div class="mb-3 p-3 rounded-4 bg-light border">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="fw-semibold text-secondary">Xác suất Lành tính (Benign):</span>
                                    <span id="resBenignText" class="text-success fw-bold fs-5">0%</span>
                                </div>
                                <div class="progress">
                                    <div id="barBenign" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="p-3 rounded-4 bg-light border">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="fw-semibold text-secondary">Xác suất Ác tính (Malignant):</span>
                                    <span id="resMalignantText" class="text-danger fw-bold fs-5">0%</span>
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
                <h1 class="fw-bold text-dark mb-1 fs-3">Lịch Sử Ca Chẩn Đoán</h1>
                <p class="text-muted mb-4">Các ca phân tích đã thực hiện trong phiên làm việc hiện tại.</p>
                <div class="card p-4">
                    <div class="table-responsive">
                        <table class="table table-medical align-middle mb-0">
                            <thead>
                                <tr>
                                    <th>Mã Ca Khám</th>
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
                <h1 class="fw-bold text-dark mb-1 fs-3">Thống Kê Hệ Thống</h1>
                <p class="text-muted mb-4">Tổng quan phân phối dữ liệu phân tích và cấu hình mô hình SVM.</p>
                <div class="row g-4 mb-4">
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-teal border-4" style="border-top-color: var(--primary-teal) !important;">
                            <span class="text-muted fw-bold small text-uppercase">Tổng Số Ca</span>
                            <h2 id="statTotal" class="fw-bold mt-2 display-5 text-teal" style="color: var(--primary-teal)">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-success border-4">
                            <span class="text-muted fw-bold small text-uppercase">Lành Tính</span>
                            <h2 id="statBenign" class="text-success fw-bold mt-2 display-5">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-danger border-4">
                            <span class="text-muted fw-bold small text-uppercase">Ác Tính</span>
                            <h2 id="statMalignant" class="text-danger fw-bold mt-2 display-5">0</h2>
                        </div>
                    </div>
                </div>
                <div class="card p-4">
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-server me-2 text-teal" style="color:var(--primary-teal)"></i> Cấu Hình Kỹ Thuật</h5>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item d-flex justify-content-between py-3"><span>Thuật toán:</span> <strong class="text-dark">Support Vector Machine (SVC)</strong></li>
                        <li class="list-group-item d-flex justify-content-between py-3"><span>Phiên bản:</span> <strong id="modelVersionDisplay" class="text-dark">1.0.0</strong></li>
                        <li class="list-group-item d-flex justify-content-between py-3"><span>Không gian đặc trưng:</span> <strong class="text-dark">30 chiều</strong></li>
                    </ul>
                </div>
            </div>

            <!-- TAB 4: HƯỚNG DẪN -->
            <div id="tab-guide" class="tab-pane-content">
                <h1 class="fw-bold text-dark mb-1 fs-3">Tra Cứu 30 Chỉ Số Lâm Sàng</h1>
                <p class="text-muted mb-4">Ý nghĩa kỹ thuật của các đặc trưng trích xuất từ tế bào học.</p>
                <div class="card p-4">
                    <div class="row g-3" id="guideContainer"></div>
                </div>
            </div>
        </div>

        <script>
            const featureMap = {
                "mean radius": { name: "Bán kính trung bình", desc: "Kích thước trung bình của các tế bào" },
                "mean texture": { name: "Độ mịn/Kết cấu bề mặt", desc: "Độ chuẩn sai của giá trị xám trên bề mặt" },
                "mean perimeter": { name: "Chu vi trung bình", desc: "Chu vi trung bình của nhân tế bào" },
                "mean area": { name: "Diện tích trung bình", desc: "Diện tích trung bình của nhân tế bào" },
                "mean smoothness": { name: "Độ trơn nhẵn TB", desc: "Độ biến thiên cục bộ của bán kính" },
                "mean compactness": { name: "Độ nén trung bình", desc: "Tính theo (chu vi^2 / diện tích - 1.0)" },
                "mean concavity": { name: "Độ lõm trung bình", desc: "Mức độ sâu của các phần lõm trên viền" },
                "mean concave points": { name: "Số điểm lõm TB", desc: "Số lượng các phần lõm trên đường viền" },
                "mean symmetry": { name: "Độ đối xứng TB", desc: "Mức độ cân đối hình dáng nhân tế bào" },
                "mean fractal dimension": { name: "Độ chiều Fractal", desc: "Độ phức tạp bờ viền" },
                "radius error": { name: "Sai số bán kính", desc: "Độ lệch chuẩn của kích thước bán kính" },
                "texture error": { name: "Sai số kết cấu", desc: "Độ lệch chuẩn của độ mịn bề mặt" },
                "perimeter error": { name: "Sai số chu vi", desc: "Độ lệch chuẩn của chu vi nhân tế bào" },
                "area error": { name: "Sai số diện tích", desc: "Độ lệch chuẩn của diện tích tế bào" },
                "smoothness error": { name: "Sai số độ nhẵn", desc: "Độ biến thiên sai số độ mịn" },
                "compactness error": { name: "Sai số độ nén", desc: "Sai số mức độ nén tế bào" },
                "concavity error": { name: "Sai số độ lõm", desc: "Sai số độ sâu phần lõm viền nhân" },
                "concave points error": { name: "Sai số điểm lõm", desc: "Sai số số lượng điểm lõm trên viền" },
                "symmetry error": { name: "Sai số độ đối xứng", desc: "Sai số mức độ cân đối" },
                "fractal dimension error": { name: "Sai số chiều Fractal", desc: "Sai số độ phức tạp bờ viền" },
                "worst radius": { name: "Bán kính lớn nhất", desc: "Giá trị trung bình của 3 giá trị lớn nhất" },
                "worst texture": { name: "Kết cấu lớn nhất", desc: "Giá trị kết cấu lớn nhất trong mẫu" },
                "worst perimeter": { name: "Chu vi lớn nhất", desc: "Chu vi lớn nhất ghi nhận được" },
                "worst area": { name: "Diện tích lớn nhất", desc: "Diện tích lớn nhất ghi nhận được" },
                "worst smoothness": { name: "Độ nhẵn lớn nhất", desc: "Độ trơn nhẵn cao nhất" },
                "worst compactness": { name: "Độ nén lớn nhất", desc: "Độ nén cao nhất của nhóm lớn nhất" },
                "worst concavity": { name: "Độ lõm lớn nhất", desc: "Độ lõm sâu nhất ghi nhận trong mẫu" },
                "worst concave points": { name: "Điểm lõm lớn nhất", desc: "Số điểm lõm nhiều nhất ghi nhận" },
                "worst symmetry": { name: "Đối xứng lớn nhất", desc: "Độ lệch đối xứng lớn nhất" },
                "worst fractal dimension": { name: "Chiều Fractal lớn nhất", desc: "Độ phức tạp bờ viền lớn nhất" }
            };
            const sampleMalignant = {"mean radius":17.99,"mean texture":10.38,"mean perimeter":122.8,"mean area":1001.0,"mean smoothness":0.1184,"mean compactness":0.2776,"mean concavity":0.3001,"mean concave points":0.1471,"mean symmetry":0.2419,"mean fractal dimension":0.07871,"radius error":1.095,"texture error":0.9053,"perimeter error":8.589,"area error":153.4,"smoothness error":0.006399,"compactness error":0.04904,"concavity error":0.05373,"concave points error":0.01587,"symmetry error":0.03003,"fractal dimension error":0.006193,"worst radius":25.38,"worst texture":17.33,"worst perimeter":184.6,"worst area":2019.0,"worst smoothness":0.1622,"worst compactness":0.6656,"worst concavity":0.7119,"worst concave points":0.2654,"worst symmetry":0.4601,"worst fractal dimension":0.1189};
            const sampleBenign = {"mean radius":13.54,"mean texture":14.36,"mean perimeter":87.46,"mean area":566.3,"mean smoothness":0.09779,"mean compactness":0.08129,"mean concavity":0.06664,"mean concave points":0.04781,"mean symmetry":0.1885,"mean fractal dimension":0.05766,"radius error":0.2699,"texture error":0.7886,"perimeter error":2.058,"area error":23.56,"smoothness error":0.008462,"compactness error":0.0146,"concavity error":0.02387,"concave points error":0.01315,"symmetry error":0.0198,"fractal dimension error":0.0023,"worst radius":15.11,"worst texture":19.26,"worst perimeter":99.7,"worst area":711.2,"worst smoothness":0.144,"worst compactness":0.1773,"worst concavity":0.239,"worst concave points":0.1288,"worst symmetry":0.2977,"worst fractal dimension":0.07259};

            let featureNames = [], historyList = [], stats = { total: 0, benign: 0, malignant: 0 };

            async function init() {
                const res = await fetch('/metadata');
                const data = await res.json();
                featureNames = data.feature_names;
                document.getElementById('modelVersionDisplay').innerText = data.model_version;
                renderInputs();
                renderGuide();
            }

            function renderInputs() {
                const c = document.getElementById('featureInputs');
                c.innerHTML = '';
                featureNames.forEach(f => {
                    let info = featureMap[f] || { name: f, desc: "Chỉ số tế bào" };
                    c.innerHTML += `<div class="col-md-4"><div class="feature-card h-100"><label class="form-label text-dark small fw-bold mb-1">${info.name}</label><div class="text-muted small mb-2" style="font-size:0.75rem; height:2.2em; overflow:hidden;">${info.desc}</div><input type="text" class="form-control form-control-sm" name="${f}" id="inp_${f.replace(/\s+/g, '_')}" value="${sampleBenign[f]}"></div></div>`;
                });
            }

            function renderGuide() {
                const g = document.getElementById('guideContainer');
                g.innerHTML = '';
                Object.keys(featureMap).forEach((k, i) => {
                    let info = featureMap[k];
                    g.innerHTML += `<div class="col-md-6"><div class="p-3 border rounded bg-light h-100"><h6 class="text-teal fw-bold mb-1" style="color:var(--primary-teal)">${i+1}. ${info.name}</h6><code class="text-muted small">key: ${k}</code><p class="small text-secondary mb-0 mt-1">${info.desc}</p></div></div>`;
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
                btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Đang phân tích...';
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
                        document.getElementById('resultAlert').className = 'alert alert-success border-0 bg-success bg-opacity-10 text-success fw-bold py-3 shadow-sm';
                        document.getElementById('resultAlert').innerHTML = '<i class="fa-solid fa-circle-check me-2"></i> Khối u lành tính, không phát hiện dấu hiệu ác tính đáng kể.';
                        stats.benign++; rec.label = '<span class="badge bg-success">Lành tính</span>';
                    } else {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-danger">ÁC TÍNH (MALIGNANT)</span>';
                        document.getElementById('resultAlert').className = 'alert alert-danger border-0 bg-danger bg-opacity-10 text-danger fw-bold py-3 shadow-sm';
                        document.getElementById('resultAlert').innerHTML = '<i class="fa-solid fa-triangle-exclamation me-2"></i> CẢNH BÁO: Khối u có xu hướng ác tính cao! Khuyến nghị hội chẩn chuyên khoa.';
                        stats.malignant++; rec.label = '<span class="badge bg-danger">Ác tính</span>';
                    }
                    document.getElementById('statTotal').innerText = stats.total;
                    document.getElementById('statBenign').innerText = stats.benign;
                    document.getElementById('statMalignant').innerText = stats.malignant;
                    
                    historyList.unshift(rec);
                    let tb = document.getElementById('historyTableBody');
                    tb.innerHTML = historyList.map(h => `<tr><td><strong class="text-dark">${h.id}</strong></td><td class="text-muted">${h.time}</td><td>${h.label}</td><td class="fw-bold">${h.prob}</td><td><button class="btn btn-sm btn-outline-teal px-3 rounded-pill fw-semibold" style="border-color:var(--primary-teal); color:var(--primary-teal);" onclick="switchTab('tab-predict', event)">Xem</button></td></tr>`).join('');
                    
                    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } finally {
                    btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-calculator me-2"></i> Thực Hiện Chẩn Đoán AI';
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
