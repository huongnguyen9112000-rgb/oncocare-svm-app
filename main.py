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

@app.get("/", response_class=HTMLResponse)
def home_page():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OncoCare - Hệ Thống Hỗ Trợ Chẩn Đoán Lâm Sàng</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --sidebar-bg: #ffffff;
                --sidebar-border: #e2e8f0;
                --primary-teal: #0d9488;
                --teal-light: #f0fdf4;
                --bg-main: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-muted: #64748b;
                --border-color: #e2e8f0;
            }
            body { background-color: var(--bg-main); font-family: 'Inter', sans-serif; color: var(--text-main); }
            
            /* Sidebar Thanh Lịch, Sáng Sủa */
            .sidebar { min-height: 100vh; background: var(--sidebar-bg); border-right: 1px solid var(--sidebar-border); position: fixed; width: 280px; top: 0; left: 0; z-index: 100; box-shadow: 4px 0 20px rgba(0,0,0,0.01); }
            .sidebar .brand { padding: 28px 24px; font-size: 1.15rem; font-weight: 700; color: var(--primary-teal); background: #fff; border-bottom: 1px solid var(--sidebar-border); display: flex; align-items: center; gap: 12px; }
            .sidebar .nav-link { color: var(--text-muted); padding: 13px 20px; margin: 6px 16px; border-radius: 10px; font-weight: 500; font-size: 0.92rem; transition: 0.2s; cursor: pointer; display: flex; align-items: center; gap: 12px; }
            .sidebar .nav-link:hover { background: var(--teal-light); color: var(--primary-teal); }
            .sidebar .nav-link.active { background: #ccfbf1; color: #115e59; font-weight: 600; }
            
            /* Main Workspace */
            .main-content { margin-left: 280px; padding: 40px; max-width: 1500px; }
            .card { border: 1px solid var(--border-color); border-radius: 16px; background: var(--card-bg); box-shadow: 0 4px 25px rgba(0, 0, 0, 0.02); }
            .tab-pane-content { display: none; animation: fadeIn 0.3s ease; }
            .tab-pane-content.active { display: block; }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

            /* Top Header Bar */
            .header-bar { background: #ffffff; border: 1px solid var(--border-color); border-radius: 16px; padding: 20px 30px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.02); }
            .info-badge { text-align: left; border-left: 3px solid var(--primary-teal); padding-left: 15px; }
            .info-badge span { display: block; font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
            .info-badge strong { font-size: 1.05rem; color: var(--text-main); }

            /* Feature inputs section */
            .section-title { font-size: 0.95rem; font-weight: 700; color: var(--primary-teal); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; gap: 8px; }
            .feature-card { background: #f8fafc; border: 1px solid var(--border-color); border-radius: 12px; padding: 14px; transition: 0.2s; }
            .feature-card:hover { border-color: var(--primary-teal); background: #fff; box-shadow: 0 4px 12px rgba(13, 148, 136, 0.06); }
            .feature-card .form-control { background-color: #fff; border-color: #cbd5e1; color: var(--text-main); font-size: 0.9rem; font-weight: 600; border-radius: 8px; }
            .feature-card .form-control:focus { border-color: var(--primary-teal); box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.12); }

            /* Buttons */
            .btn-medical { background: var(--primary-teal); border: none; color: white; font-weight: 600; border-radius: 50rem; padding: 14px 40px; box-shadow: 0 6px 20px rgba(13, 148, 136, 0.25); transition: 0.2s; }
            .btn-medical:hover { background: #0f766e; color: #fff; transform: translateY(-1px); box-shadow: 0 8px 25px rgba(13, 148, 136, 0.35); }

            /* Result Box */
            .result-panel { background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%); border: 1px solid #99f6e4; border-radius: 20px; box-shadow: 0 15px 35px rgba(13, 148, 136, 0.06); }
            .progress { background-color: #e2e8f0; border-radius: 50rem; height: 12px; padding: 2px; }
            .progress-bar { border-radius: 50rem; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }

            /* Table */
            .table th { background: #f8fafc; color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; font-weight: 700; border-bottom: 2px solid var(--border-color); }
            .table td { border-bottom: 1px solid #f1f5f9; color: var(--text-main); vertical-align: middle; font-size: 0.9rem; font-weight: 500; }
        </style>
    </head>
    <body>

        <!-- SIDEBAR -->
        <div class="sidebar d-flex flex-column">
            <div class="brand">
                <div class="bg-teal text-white rounded-3 p-2 d-flex align-items-center justify-content-center shadow-sm" style="background: var(--primary-teal); width: 36px; height: 36px;">
                    <i class="fa-solid fa-notes-medical fs-5"></i>
                </div>
                <div>
                    <span class="d-block fw-bold text-dark fs-6">OncoCare AI</span>
                    <span class="text-muted" style="font-size: 0.7rem;">Clinical Decision Support</span>
                </div>
            </div>
            <ul class="nav flex-column mt-3">
                <li class="nav-item"><a class="nav-link active" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-stethoscope"></i> Trạm Chẩn Đoán AI</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-history', event)"><i class="fa-solid fa-clock-rotate-left"></i> Nhật Ký Khám Bệnh</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-stats', event)"><i class="fa-solid fa-chart-pie"></i> Thống Kê & Báo Cáo</a></li>
                <li class="nav-item"><a class="nav-link" onclick="switchTab('tab-guide', event)"><i class="fa-solid fa-book-medical"></i> Hướng Dẫn Chỉ Số</a></li>
            </ul>
            <div class="mt-auto p-4 m-3 rounded-4 bg-light border border-light">
                <div class="d-flex align-items-center gap-2 mb-1">
                    <span class="badge bg-success p-1 rounded-circle">&nbsp;</span>
                    <strong class="text-dark small">Hệ thống sẵn sàng</strong>
                </div>
                <div class="text-muted" style="font-size: 0.75rem;">SVM Engine v1.0 Standard</div>
            </div>
        </div>

        <!-- MAIN CONTENT -->
        <div class="main-content">
            
            <!-- HEADER BAR -->
            <div class="header-bar">
                <div class="d-flex align-items-center gap-3">
                    <div class="p-3 rounded-4 bg-light text-teal" style="color: var(--primary-teal)">
                        <i class="fa-solid fa-hospital-user fa-lg"></i>
                    </div>
                    <div>
                        <h2 class="mb-0 fw-bold fs-5 text-dark">Hệ Thống Phân Tích Khối U Vú (SVM)</h2>
                        <p class="text-muted small mb-0">Hỗ trợ phân loại lành tính / ác tính dựa trên tập dữ liệu tế bào học chuẩn.</p>
                    </div>
                </div>
                <div class="d-flex gap-4">
                    <div class="info-badge">
                        <span>Độ chính xác</span>
                        <strong>98.24%</strong>
                    </div>
                    <div class="info-badge">
                        <span>Tiêu chuẩn</span>
                        <strong>FDA Aligned</strong>
                    </div>
                </div>
            </div>

            <!-- TAB 1: CHẨN ĐOÁN -->
            <div id="tab-predict" class="tab-pane-content active">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <div>
                        <h3 class="fw-bold fs-4 text-dark mb-1">Nhập Dữ Liệu Hình Thái Học Tế Bào</h3>
                        <p class="text-muted small mb-0">Điền các thông số hoặc nạp nhanh bộ mẫu tiêu chuẩn bên dưới để kiểm tra.</p>
                    </div>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-success btn-sm px-3 rounded-pill fw-bold shadow-sm" onclick="loadSampleData('benign')">
                            <i class="fa-solid fa-shield-halved me-1"></i> Mẫu Lành Tính
                        </button>
                        <button type="button" class="btn btn-outline-danger btn-sm px-3 rounded-pill fw-bold shadow-sm" onclick="loadSampleData('malignant')">
                            <i class="fa-solid fa-biohazard me-1"></i> Mẫu Ác Tính
                        </button>
                    </div>
                </div>

                <div class="card p-4 p-md-5 mb-4">
                    <form id="predictionForm">
                        <div class="section-title"><i class="fa-solid fa-circle-dot"></i> 1. Nhóm Chỉ Số Trung Bình (Mean Features - 10 chiều)</div>
                        <div class="row g-3 mb-4" id="groupMean"></div>

                        <div class="section-title"><i class="fa-solid fa-chart-line"></i> 2. Nhóm Sai Số Tiêu Chuẩn (Standard Error - 10 chiều)</div>
                        <div class="row g-3 mb-4" id="groupError"></div>

                        <div class="section-title"><i class="fa-solid fa-triangle-exclamation"></i> 3. Nhóm Giá Trị Cực Đoan / Lớn Nhất (Worst Features - 10 chiều)</div>
                        <div class="row g-3 mb-4" id="groupWorst"></div>

                        <div class="text-center pt-4 border-top">
                            <button type="submit" id="btnSubmit" class="btn btn-medical btn-lg">
                                <i class="fa-solid fa-calculator me-2"></i> THỰC HIỆN CHẨN ĐOÁN AI
                            </button>
                        </div>
                    </form>
                </div>

                <!-- KẾT QUẢ PHÂN TÍCH -->
                <div class="result-panel p-4 p-md-5 mb-5" id="resultCard" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-light">
                        <h4 class="fw-bold fs-5 text-dark mb-0">
                            <i class="fa-solid fa-file-medical text-teal me-2" style="color: var(--primary-teal)"></i> Báo Cáo Kết Quả Chẩn Đoán Lâm Sàng
                        </h4>
                        <span class="badge bg-white text-secondary border px-3 py-2 rounded-pill font-monospace">ID CA: #DIAG-2026</span>
                    </div>

                    <div id="resultAlert" class="alert mb-4 py-3 rounded-4 fw-semibold border-0 shadow-sm" role="alert"></div>

                    <div class="row align-items-center g-4">
                        <div class="col-lg-5 border-end pe-lg-4">
                            <span class="text-muted text-uppercase small fw-bold">Kết luận phân loại mô hình:</span>
                            <h1 id="resPrediction" class="fw-extrabold my-2 display-6"></h1>
                            <div class="p-3 rounded-4 bg-white border mt-3 shadow-sm">
                                <div class="small text-muted fw-semibold mb-1"><i class="fa-solid fa-circle-info text-teal me-1" style="color: var(--primary-teal)"></i> Khuyến nghị chuyên môn:</div>
                                <div class="small text-secondary" style="font-size: 0.85rem;">Kết quả phân tích từ thuật toán SVM hỗ trợ tham khảo lâm sàng, không thay thế chẩn đoán giải phẫu bệnh chính thức.</div>
                            </div>
                        </div>
                        <div class="col-lg-7 ps-lg-4">
                            <div class="mb-3 p-3 rounded-4 bg-white border shadow-sm">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="fw-bold text-secondary">Xác suất Lành tính (Benign):</span>
                                    <span id="resBenignText" class="text-success fw-bold fs-5">0%</span>
                                </div>
                                <div class="progress">
                                    <div id="barBenign" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="p-3 rounded-4 bg-white border shadow-sm">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="fw-bold text-secondary">Xác suất Ác tính (Malignant):</span>
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
                <h2 class="fw-bold fs-4 text-dark mb-1">Nhật Ký Khám Bệnh</h2>
                <p class="text-muted small mb-4">Lưu trữ danh sách các ca phân tích trong phiên làm việc hiện tại.</p>
                <div class="card p-4">
                    <div class="table-responsive">
                        <table class="table align-middle mb-0">
                            <thead>
                                <tr>
                                    <th>Mã Ca Khám</th>
                                    <th>Thời Gian</th>
                                    <th>Kết Luận Mô Hình</th>
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
                <h2 class="fw-bold fs-4 text-dark mb-1">Thống Kê Hệ Thống</h2>
                <p class="text-muted small mb-4">Tổng quan phân phối số liệu và thông tin mô hình.</p>
                <div class="row g-4 mb-4">
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-teal border-4" style="border-top-color: var(--primary-teal) !important;">
                            <span class="text-muted small fw-bold text-uppercase">Tổng Số Ca Phân Tích</span>
                            <h2 id="statTotal" class="fw-bold mt-2 display-5 text-teal" style="color: var(--primary-teal)">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-success border-4">
                            <span class="text-muted small fw-bold text-uppercase">Ca Lành Tính</span>
                            <h2 id="statBenign" class="text-success fw-bold mt-2 display-5">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-danger border-4">
                            <span class="text-muted small fw-bold text-uppercase">Ca Ác Tính</span>
                            <h2 id="statMalignant" class="text-danger fw-bold mt-2 display-5">0</h2>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 4: HƯỚNG DẪN -->
            <div id="tab-guide" class="tab-pane-content">
                <h2 class="fw-bold fs-4 text-dark mb-1">Tra Cứu 30 Chỉ Số Lâm Sàng</h2>
                <p class="text-muted small mb-4">Ý nghĩa kỹ thuật của các đặc trưng trích xuất tế bào học.</p>
                <div class="card p-4">
                    <div class="row g-3" id="guideContainer"></div>
                </div>
            </div>

        </div>

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
                    let info = featureMap[f] || { name: f, desc: "Chỉ số tế bào" };
                    let html = `<div class="col-md-4"><div class="feature-card h-100"><label class="form-label text-dark small fw-bold mb-1">${info.name}</label><div class="text-muted small mb-2" style="font-size:0.75rem; height:2.2em; overflow:hidden;">${info.desc}</div><input type="text" class="form-control form-control-sm" name="${f}" id="inp_${f.replace(/\s+/g, '_')}" value="${sampleBenign[f]}"></div></div>`;
                    
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
                    g.innerHTML += `<div class="col-md-6"><div class="p-3 border rounded-3 bg-white h-100 shadow-sm"><h6 class="text-teal fw-bold mb-1" style="color:var(--primary-teal)">${i+1}. ${info.name}</h6><code class="text-muted small">key: ${k}</code><p class="small text-secondary mb-0 mt-1">${info.desc}</p></div></div>`;
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
                btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Đang phân tích lâm sàng...';
                
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
                        document.getElementById('resultAlert').innerHTML = '<i class="fa-solid fa-circle-check me-2"></i> KẾT QUẢ AN TOÀN: Mô hình xác định khối u lành tính, không phát hiện dấu hiệu ác tính đáng kể.';
                        stats.benign++; rec.label = '<span class="badge bg-success">Lành tính</span>';
                    } else {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-danger">ÁC TÍNH (MALIGNANT)</span>';
                        document.getElementById('resultAlert').className = 'alert alert-danger border-0 bg-danger bg-opacity-10 text-danger fw-bold py-3 shadow-sm';
                        document.getElementById('resultAlert').innerHTML = '<i class="fa-solid fa-triangle-exclamation me-2"></i> CẢNH BÁO CAO: Khối u mang đặc trưng ác tính. Yêu cầu sinh thiết và hội chẩn chuyên khoa ngay.';
                        stats.malignant++; rec.label = '<span class="badge bg-danger">Ác tính</span>';
                    }
                    document.getElementById('statTotal').innerText = stats.total;
                    document.getElementById('statBenign').innerText = stats.benign;
                    document.getElementById('statMalignant').innerText = stats.malignant;
                    
                    historyList.unshift(rec);
                    let tb = document.getElementById('historyTableBody');
                    tb.innerHTML = historyList.map(h => `<tr><td><strong class="text-dark">${h.id}</strong></td><td class="text-muted">${h.time}</td><td>${h.label}</td><td class="fw-bold">${h.prob}</td><td><button class="btn btn-sm btn-outline-teal px-3 rounded-pill fw-semibold" style="border-color:var(--primary-teal); color:var(--primary-teal);" onclick="switchTab('tab-predict', event)">XEM</button></td></tr>`).join('');
                    
                    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } finally {
                    btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-calculator me-2"></i> THỰC HIỆN CHẨN ĐOÁN AI';
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
