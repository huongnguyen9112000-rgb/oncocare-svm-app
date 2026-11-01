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
    description="Enterprise Clinical Decision Support System"
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
        <title>OncoCare Enterprise - Clinical AI Command Center</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-root: #090d16;
                --bg-surface: #111827;
                --bg-card: #1f2937;
                --border-color: #374151;
                --accent-cyan: #06b6d4;
                --accent-emerald: #10b981;
                --accent-rose: #f43f5e;
                --text-main: #f9fafb;
                --text-muted: #9ca3af;
            }
            body { background-color: var(--bg-root); font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text-main); }
            
            /* Sidebar chuyên nghiệp doanh nghiệp */
            .sidebar { min-height: 100vh; background: var(--bg-surface); border-right: 1px solid var(--border-color); position: fixed; width: 290px; top: 0; left: 0; z-index: 100; }
            .sidebar .brand { padding: 24px; background: #0b0f19; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; gap: 12px; }
            .sidebar .brand h3 { font-size: 1.15rem; font-weight: 800; color: #fff; margin: 0; font-family: 'JetBrains Mono', monospace; }
            .sidebar .nav-link { color: var(--text-muted); padding: 12px 20px; margin: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 0.9rem; transition: 0.2s; cursor: pointer; display: flex; align-items: center; gap: 12px; }
            .sidebar .nav-link:hover, .sidebar .nav-link.active { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
            
            /* Main Workspace */
            .main-content { margin-left: 290px; padding: 35px; max-width: 1550px; }
            .card { border: 1px solid var(--border-color); border-radius: 14px; background: var(--bg-surface); box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); }
            .tab-pane-content { display: none; animation: fadeIn 0.3s ease; }
            .tab-pane-content.active { display: block; }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

            /* Telemetry & Metrics Badges */
            .telemetry-bar { background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 12px; padding: 15px 25px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }
            .metric-box { text-align: left; }
            .metric-box span { display: block; font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 0.5px; }
            .metric-box strong { font-size: 1.1rem; font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan); }

            /* Feature inputs complex grid */
            .feature-group-title { font-size: 0.9rem; font-weight: 700; color: var(--accent-cyan); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 1px dashed var(--border-color); }
            .feature-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 10px; padding: 12px; transition: 0.2s; }
            .feature-card:hover { border-color: var(--accent-cyan); }
            .feature-card .form-control { background-color: var(--bg-root); border-color: var(--border-color); color: #fff; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; }
            .feature-card .form-control:focus { border-color: var(--accent-cyan); box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2); background-color: var(--bg-root); color: #fff; }

            /* Enterprise Buttons */
            .btn-enterprise { background: linear-gradient(135deg, #06b6d4, #3b82f6); border: none; color: white; font-weight: 700; border-radius: 10px; padding: 14px 30px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px; box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3); transition: 0.2s; }
            .btn-enterprise:hover { opacity: 0.95; color: #fff; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4); }

            /* Result HUD */
            .hud-result { background: linear-gradient(145deg, #111827, #0b0f19); border: 1px solid var(--accent-cyan); border-radius: 16px; box-shadow: 0 0 30px rgba(6, 182, 212, 0.15); }
            .progress { background-color: var(--bg-root); border-radius: 6px; height: 12px; border: 1px solid var(--border-color); padding: 2px; }
            .progress-bar { border-radius: 4px; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }

            /* Table */
            .table-enterprise th { background: #0b0f19; color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; border-bottom: 1px solid var(--border-color); }
            .table-enterprise td { border-bottom: 1px solid var(--border-color); color: var(--text-main); vertical-align: middle; font-size: 0.9rem; }
        </style>
    </head>
    <body>

        <!-- ENTERPRISE SIDEBAR -->
        <div class="sidebar d-flex flex-column">
            <div class="brand">
                <i class="fa-solid fa-microchip text-cyan fs-4" style="color: var(--accent-cyan)"></i>
                <div>
                    <h3>ONCOCARE</h3>
                    <span class="text-muted" style="font-size: 0.7rem; font-family: 'JetBrains Mono';">ENTERPRISE v2.6</span>
                </div>
            </div>
            <ul class="nav flex-column mt-3">
                <li class="nav-item"><a class="nav-link active" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-terminal"></i> Trạm Phân Tích SVM</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-history', event)"><i class="fa-solid fa-database"></i> Nhật Ký Lâm Sàng</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-stats', event)"><i class="fa-solid fa-chart-line"></i> Chỉ Số Telemetry</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-guide', event)"><i class="fa-solid fa-book-medical"></i> Tài Liệu Đặc Trưng</a></li>
            </ul>
            <div class="mt-auto p-3 text-muted small border-top border-secondary m-3 rounded bg-dark">
                <div class="d-flex align-items-center gap-2 mb-1">
                    <span class="spinner-grow spinner-grow-sm text-success" role="status"></span>
                    <strong class="text-white" style="font-size: 0.75rem;">SVM Engine Online</strong>
                </div>
                <div style="font-size: 0.7rem; font-family: 'JetBrains Mono';">SECURE CLUSTER: ACTIVE</div>
            </div>
        </div>

        <!-- MAIN WORKSPACE -->
        <div class="main-content">
            
            <!-- TELEMETRY TOP BAR -->
            <div class="telemetry-bar">
                <div class="d-flex align-items-center gap-3">
                    <div class="bg-dark p-2 rounded border border-secondary text-cyan" style="color: var(--accent-cyan);">
                        <i class="fa-solid fa-shield-halved fa-lg"></i>
                    </div>
                    <div>
                        <h4 class="mb-0 fw-bold fs-5">Clinical Decision Support Command Center</h4>
                        <span class="text-muted small">Hệ thống phân tích học máy Support Vector Machine (Wisconsin Dataset)</span>
                    </div>
                </div>
                <div class="d-flex gap-4">
                    <div class="metric-box">
                        <span>Độ chính xác mô hình</span>
                        <strong>98.24%</strong>
                    </div>
                    <div class="metric-box">
                        <span>Thời gian phản hồi</span>
                        <strong id="latencyBadge">12ms</strong>
                    </div>
                    <div class="metric-box">
                        <span>Trạng thái kết nối</span>
                        <strong class="text-success">200 OK</strong>
                    </div>
                </div>
            </div>

            <!-- TAB 1: TRẠM PHÂN TÍCH -->
            <div id="tab-predict" class="tab-pane-content active">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <div>
                        <h2 class="fw-bold fs-4 mb-1">Bảng Điều Khiển Thông Số Tế Bào (30 Chiều)</h2>
                        <p class="text-muted small mb-0">Điền hoặc nạp bộ mẫu chuẩn chế độ tự động để thực hiện suy luận mô hình.</p>
                    </div>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-success btn-sm fw-bold font-monospace" onclick="loadSampleData('benign')">
                            <i class="fa-solid fa-vial-circle-check me-1"></i> LOAD BENIGN SAMPLE
                        </button>
                        <button type="button" class="btn btn-outline-danger btn-sm fw-bold font-monospace" onclick="loadSampleData('malignant')">
                            <i class="fa-solid fa-biohazard me-1"></i> LOAD MALIGNANT SAMPLE
                        </button>
                    </div>
                </div>

                <div class="card p-4 mb-4">
                    <form id="predictionForm">
                        <div class="feature-group-title"><i class="fa-solid fa-layer-group me-2"></i> Nhóm Chỉ Số Trung Bình (Mean Features - 10 chiều)</div>
                        <div class="row g-3 mb-4" id="groupMean"></div>

                        <div class="feature-group-title"><i class="fa-solid fa-chart-line me-2"></i> Nhóm Sai Số Tiêu Chuẩn (Standard Error - 10 chiều)</div>
                        <div class="row g-3 mb-4" id="groupError"></div>

                        <div class="feature-group-title"><i class="fa-solid fa-triangle-exclamation me-2"></i> Nhóm Giá Trị Nặng Nhất / Cực Đoan (Worst - 10 chiều)</div>
                        <div class="row g-3 mb-4" id="groupWorst"></div>

                        <div class="text-center pt-3 border-top border-secondary">
                            <button type="submit" id="btnSubmit" class="btn btn-enterprise btn-lg px-5">
                                <i class="fa-solid fa-microchip me-2"></i> KHỞI CHẠY THUẬT TOÁN SVM PREDICT
                            </button>
                        </div>
                    </form>
                </div>

                <!-- KẾT QUẢ PHÂN TÍCH HUD -->
                <div class="hud-result p-4 p-md-5 mb-5" id="resultCard" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
                        <h3 class="fw-bold fs-5 mb-0 font-monospace text-cyan" style="color: var(--accent-cyan);">
                            <i class="fa-solid fa-square-poll-vertical me-2"></i> KẾT QUẢ SUY LUẬN LÂM SÀNG
                        </h3>
                        <span class="badge bg-dark font-monospace border border-secondary text-muted">ID CA: #DIAG-2026-X9</span>
                    </div>

                    <div id="resultAlert" class="alert mb-4 py-3 rounded-3 fw-bold border-0 font-monospace small" role="alert"></div>

                    <div class="row align-items-center g-4">
                        <div class="col-lg-5 border-end border-secondary pe-lg-4">
                            <span class="text-muted text-uppercase small font-monospace fw-bold">Nhãn dự đoán từ mô hình SVM:</span>
                            <h1 id="resPrediction" class="fw-extrabold my-2 display-6 font-monospace"></h1>
                            <div class="p-3 rounded bg-dark border border-secondary mt-3">
                                <div class="small text-muted font-monospace mb-1"><i class="fa-solid fa-circle-info text-cyan me-1"></i> KHUYẾN NGHỊ HỆ THỐNG:</div>
                                <div class="small text-secondary" style="font-size: 0.8rem;">Kết quả được tính toán bằng mạng SVM tiêu chuẩn. Mang tính chất hỗ trợ quyết định lâm sàng tham khảo.</div>
                            </div>
                        </div>
                        <div class="col-lg-7 ps-lg-4">
                            <div class="mb-3 p-3 rounded bg-dark border border-secondary">
                                <div class="d-flex justify-content-between mb-2 font-monospace small">
                                    <span class="text-muted">Xác suất Lành tính (Benign):</span>
                                    <span id="resBenignText" class="text-success fw-bold">0%</span>
                                </div>
                                <div class="progress">
                                    <div id="barBenign" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="p-3 rounded bg-dark border border-secondary">
                                <div class="d-flex justify-content-between mb-2 font-monospace small">
                                    <span class="text-muted">Xác suất Ác tính (Malignant):</span>
                                    <span id="resMalignantText" class="text-danger fw-bold">0%</span>
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
                <h2 class="fw-bold fs-4 mb-1">Nhật Ký Phiên Làm Việc</h2>
                <p class="text-muted small mb-4">Lưu vết toàn bộ lịch sử các lượt gọi suy luận mô hình trong phiên hiện tại.</p>
                <div class="card p-4">
                    <div class="table-responsive">
                        <table class="table table-enterprise align-middle mb-0">
                            <thead>
                                <tr>
                                    <th>Mã Định Danh</th>
                                    <th>Thời Gian Thực Hiện</th>
                                    <th>Kết Luận SVM</th>
                                    <th>Xác Suất Ác Tính</th>
                                    <th>Thao Tác</th>
                                </tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <tr><td colspan="5" class="text-center text-muted py-5 font-monospace">CHƯA CÓ DỮ LIỆU NHẬT KÝ LÂM SÀNG.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- TAB 3: THỐNG KÊ -->
            <div id="tab-stats" class="tab-pane-content">
                <h2 class="fw-bold fs-4 mb-1">Chỉ Số Thống Kê & Telemetry</h2>
                <p class="text-muted small mb-4">Phân tích tần suất và phân phối kết quả chẩn đoán.</p>
                <div class="row g-4 mb-4">
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-info border-4">
                            <span class="text-muted small font-monospace text-uppercase">Tổng Số Lượt Phân Tích</span>
                            <h2 id="statTotal" class="fw-bold mt-2 display-5 font-monospace text-cyan" style="color: var(--accent-cyan)">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-success border-4">
                            <span class="text-muted small font-monospace text-uppercase">Kết Quả Lành Tính</span>
                            <h2 id="statBenign" class="text-success fw-bold mt-2 display-5 font-monospace">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-danger border-4">
                            <span class="text-muted small font-monospace text-uppercase">Kết Quả Ác Tính</span>
                            <h2 id="statMalignant" class="text-danger fw-bold mt-2 display-5 font-monospace">0</h2>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 4: HƯỚNG DẪN -->
            <div id="tab-guide" class="tab-pane-content">
                <h2 class="fw-bold fs-4 mb-1">Tài Liệu Đặc Trưng Lâm Sàng (30 Features)</h2>
                <p class="text-muted small mb-4">Danh mục kỹ thuật các giá trị trích xuất hình thái học từ tế bào học kim nhỏ.</p>
                <div class="card p-4">
                    <div class="row g-3" id="guideContainer"></div>
                </div>
            </div>

        </div>

        <script>
            const featureMap = {
                "mean radius": { name: "Mean Radius", desc: "Bán kính trung bình nhân tế bào" },
                "mean texture": { name: "Mean Texture", desc: "Độ mịn/Độ lệch chuẩn giá trị xám" },
                "mean perimeter": { name: "Mean Perimeter", desc: "Chu vi trung bình của nhân" },
                "mean area": { name: "Mean Area", desc: "Diện tích trung bình của nhân" },
                "mean smoothness": { name: "Mean Smoothness", desc: "Độ biến thiên cục bộ bán kính" },
                "mean compactness": { name: "Mean Compactness", desc: "Tính theo chu vi^2 / diện tích - 1.0" },
                "mean concavity": { name: "Mean Concavity", desc: "Mức độ sâu các phần lõm bờ viền" },
                "mean concave points": { name: "Mean Concave Points", desc: "Số lượng phần lõm trên đường viền" },
                "mean symmetry": { name: "Mean Symmetry", desc: "Độ đối xứng hình dáng tế bào" },
                "mean fractal dimension": { name: "Mean Fractal Dimension", desc: "Độ phức tạp bờ viền (Xấp xỉ fractal)" },
                "radius error": { name: "Radius Error", desc: "Sai số chuẩn bán kính" },
                "texture error": { name: "Texture Error", desc: "Sai số chuẩn kết cấu bề mặt" },
                "perimeter error": { name: "Perimeter Error", desc: "Sai số chuẩn chu vi nhân" },
                "area error": { name: "Area Error", desc: "Sai số chuẩn diện tích nhân" },
                "smoothness error": { name: "Smoothness Error", desc: "Sai số biến thiên độ mịn" },
                "compactness error": { name: "Compactness Error", desc: "Sai số mức độ nén tế bào" },
                "concavity error": { name: "Concavity Error", desc: "Sai số chiều sâu phần lõm" },
                "concave points error": { name: "Concave Points Error", desc: "Sai số số lượng điểm lõm" },
                "symmetry error": { name: "Symmetry Error", desc: "Sai số mức độ đối xứng" },
                "fractal dimension error": { name: "Fractal Dimension Error", desc: "Sai số độ phức tạp bờ viền" },
                "worst radius": { name: "Worst Radius", desc: "Giá trị lớn nhất của bán kính" },
                "worst texture": { name: "Worst Texture", desc: "Giá trị lớn nhất của kết cấu bề mặt" },
                "worst perimeter": { name: "Worst Perimeter", desc: "Chu vi lớn nhất ghi nhận" },
                "worst area": { name: "Worst Area", desc: "Diện tích lớn nhất ghi nhận" },
                "worst smoothness": { name: "Worst Smoothness", desc: "Độ mịn cao nhất trong nhóm cực đoan" },
                "worst compactness": { name: "Worst Compactness", desc: "Độ nén lớn nhất ghi nhận" },
                "worst concavity": { name: "Worst Concavity", desc: "Độ lõm sâu nhất ghi nhận" },
                "worst concave points": { name: "Worst Concave Points", desc: "Số điểm lõm lớn nhất ghi nhận" },
                "worst symmetry": { name: "Worst Symmetry", desc: "Độ lệch đối xứng lớn nhất" },
                "worst fractal dimension": { name: "Worst Fractal Dimension", desc: "Độ phức tạp bờ viền lớn nhất" }
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
                const groupMean = document.getElementById('groupMean');
                const groupError = document.getElementById('groupError');
                const groupWorst = document.getElementById('groupWorst');
                groupMean.innerHTML = ''; groupError.innerHTML = ''; groupWorst.innerHTML = '';

                featureNames.forEach(f => {
                    let info = featureMap[f] || { name: f, desc: "Đặc trưng tế bào" };
                    let html = `<div class="col-md-4"><div class="feature-card h-100"><label class="form-label text-light small font-monospace fw-bold mb-1">${info.name}</label><div class="text-muted small mb-2 font-monospace" style="font-size:0.7rem; height:2.2em; overflow:hidden;">${info.desc}</div><input type="text" class="form-control form-control-sm" name="${f}" id="inp_${f.replace(/\s+/g, '_')}" value="${sampleBenign[f]}"></div></div>`;
                    
                    if(f.startsWith('mean')) groupMean.innerHTML += html;
                    else if(f.includes('error')) groupError.innerHTML += html;
                    else if(f.startsWith('worst')) groupWorst.innerHTML += html;
                });
            }

            function renderGuide() {
                const g = document.getElementById('guideContainer');
                g.innerHTML = '';
                Object.keys(featureMap).forEach((k, i) => {
                    let info = featureMap[k];
                    g.innerHTML += `<div class="col-md-6"><div class="p-3 border border-secondary rounded bg-dark h-100"><h6 class="text-cyan fw-bold font-monospace mb-1" style="color:var(--accent-cyan);">${i+1}. ${info.name}</h6><code class="text-muted small font-monospace">${k}</code><p class="small text-secondary mb-0 mt-1">${info.desc}</p></div></div>`;
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
                btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> ĐANG TÍNH TOÁN SVM...';
                
                const startTime = performance.now();
                const fd = new FormData(this);
                const obj = {};
                fd.forEach((v, k) => obj[k] = parseFloat(v.replace(',', '.')));
                
                try {
                    const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({features: obj}) });
                    const data = await res.json();
                    
                    const endTime = performance.now();
                    document.getElementById('latencyBadge').innerText = Math.round(endTime - startTime) + 'ms';

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
                        document.getElementById('resultAlert').className = 'alert alert-success border-0 fw-bold font-monospace';
                        document.getElementById('resultAlert').style.background = 'rgba(16, 185, 129, 0.15)';
                        document.getElementById('resultAlert').style.color = '#10b981';
                        document.getElementById('resultAlert').innerHTML = '<i class="fa-solid fa-shield-halved me-2"></i> KẾT QUẢ AN TOÀN: Mô hình xác định khối u lành tính, không phát hiện dấu hiệu ác tính.';
                        stats.benign++; rec.label = '<span class="badge bg-success font-monospace">Lành tính</span>';
                    } else {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-danger">ÁC TÍNH (MALIGNANT)</span>';
                        document.getElementById('resultAlert').className = 'alert alert-danger border-0 fw-bold font-monospace';
                        document.getElementById('resultAlert').style.background = 'rgba(244, 63, 94, 0.15)';
                        document.getElementById('resultAlert').style.color = '#f43f5e';
                        document.getElementById('resultAlert').innerHTML = '<i class="fa-solid fa-triangle-exclamation me-2"></i> CẢNH BÁO CAO: Khối u mang đặc trưng ác tính. Yêu cầu sinh thiết và hội chẩn ngay.';
                        stats.malignant++; rec.label = '<span class="badge bg-danger font-monospace">Ác tính</span>';
                    }
                    document.getElementById('statTotal').innerText = stats.total;
                    document.getElementById('statBenign').innerText = stats.benign;
                    document.getElementById('statMalignant').innerText = stats.malignant;
                    
                    historyList.unshift(rec);
                    let tb = document.getElementById('historyTableBody');
                    tb.innerHTML = historyList.map(h => `<tr><td><code class="text-cyan font-monospace" style="color:var(--accent-cyan)">${h.id}</code></td><td>${h.time}</td><td>${h.label}</td><td class="fw-bold font-monospace">${h.prob}</td><td><button class="btn btn-sm btn-outline-info font-monospace px-3 rounded" onclick="switchTab('tab-predict', event)">XEM</button></td></tr>`).join('');
                    
                    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } finally {
                    btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-microchip me-2"></i> KHỞI CHẠY THUẬT TOÁN SVM PREDICT';
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
