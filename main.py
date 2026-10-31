from pathlib import Path
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Định nghĩa đường dẫn và nạp artifact, metadata ở thư mục gốc
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "metadata.json"

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Educational demonstration only; not a medical diagnosis."
)

# Schema dữ liệu
class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Exactly 30 named numeric features matching breast cancer dataset"
    )

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
        raise HTTPException(
            status_code=422,
            detail={"missing": missing, "extra": extra}
        )

    values = np.array(
        [[payload.features[name] for name in expected]],
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise HTTPException(status_code=422, detail="Features must be finite numbers")
    return values

# Giao diện App Web Đa Danh Mục (Multi-Tab App UI) phiên bản cao cấp
@app.get("/", response_class=HTMLResponse)
def home_page():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hệ Thống Y Tế Thông Minh - OncoCare AI</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary-color: #0f766e;
                --primary-hover: #115e59;
                --bg-main: #f8fafc;
                --sidebar-bg: #0f172a;
            }
            body { background-color: var(--bg-main); font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; color: #334155; }
            
            /* Sidebar hiện đại */
            .sidebar { min-height: 100vh; background: var(--sidebar-bg); color: #94a3b8; position: fixed; width: 280px; top: 0; left: 0; z-index: 100; box-shadow: 4px 0 20px rgba(0,0,0,0.05); }
            .sidebar .brand { padding: 24px 20px; font-size: 1.25rem; font-weight: 700; color: #fff; background: #020617; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #1e293b; }
            .sidebar .nav-link { color: #94a3b8; padding: 14px 20px; margin: 6px 16px; border-radius: 10px; font-weight: 500; transition: all 0.2s ease; cursor: pointer; display: flex; align-items: center; gap: 12px; }
            .sidebar .nav-link:hover, .sidebar .nav-link.active { background: var(--primary-color); color: white; box-shadow: 0 4px 12px rgba(15, 118, 110, 0.3); }
            .sidebar .nav-link i { font-size: 1.1rem; width: 24px; text-align: center; }
            
            /* Main Content */
            .main-content { margin-left: 280px; padding: 40px; }
            .card { border: none; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); background: #ffffff; transition: transform 0.2s; }
            .tab-pane-content { display: none; animation: fadeIn 0.3s ease-in-out; }
            .tab-pane-content.active { display: block; }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to { opacity: 1; transform: translateY(0); }
            }

            /* Feature Card */
            .feature-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; transition: all 0.2s; }
            .feature-card:hover { border-color: var(--primary-color); box-shadow: 0 4px 12px rgba(15, 118, 110, 0.08); transform: translateY(-2px); }
            .feature-card .form-control { border-radius: 8px; border-color: #cbd5e1; font-weight: 600; color: #1e293b; }
            .feature-card .form-control:focus { border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.15); }

            /* Buttons */
            .btn-primary-custom { background-color: var(--primary-color); border: none; color: white; font-weight: 600; border-radius: 10px; padding: 12px 28px; transition: all 0.2s; }
            .btn-primary-custom:hover { background-color: var(--primary-hover); color: white; box-shadow: 0 6px 16px rgba(15, 118, 110, 0.3); }

            /* Progress bar custom */
            .progress { border-radius: 8px; background-color: #f1f5f9; overflow: hidden; }
            .progress-bar { transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }
        </style>
    </head>
    <body>

        <!-- THANH ĐIỀU HƯỚNG BÊN TRÁI -->
        <div class="sidebar d-flex flex-column">
            <div class="brand">
                <i class="fa-solid fa-notes-medical text-teal fs-4 text-white"></i> 
                <span>OncoCare AI</span>
            </div>
            <ul class="nav flex-column mt-4">
                <li class="nav-item">
                    <a class="nav-link active" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-stethoscope"></i> Chẩn Đoán AI</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" onclick="switchTab('tab-history', event)"><i class="fa-solid fa-clock-rotate-left"></i> Lịch Sử Khám</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" onclick="switchTab('tab-stats', event)"><i class="fa-solid fa-chart-pie"></i> Thống Kê & Báo Cáo</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" onclick="switchTab('tab-guide', event)"><i class="fa-solid fa-book-medical"></i> Hướng Dẫn Chỉ Số</a>
                </li>
            </ul>
            <div class="mt-auto p-4 text-muted small border-top border-dark">
                <div class="fw-semibold text-slate-300">Hệ thống SVM Lâm Sàng</div>
                <div style="font-size: 0.8rem;" class="mt-1">Phiên bản Y tế thông minh © 2026</div>
            </div>
        </div>

        <!-- NỘI DUNG CHÍNH -->
        <div class="main-content">
            
            <!-- DANH MỤC 1: CHẨN ĐOÁN AI -->
            <div id="tab-predict" class="tab-pane-content active">
                <div class="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom">
                    <div>
                        <h2 class="fw-bold text-dark mb-1">Phân Tích & Chẩn Đoán Ung Thư Vú</h2>
                        <p class="text-muted mb-0">Hệ thống trí tuệ nhân tạo hỗ trợ phân loại khối u dựa trên mô hình SVM chuẩn hóa.</p>
                    </div>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-success btn-sm px-3 rounded-pill fw-semibold" onclick="loadSampleData('benign')">
                            <i class="fa-solid fa-shield-halved me-1"></i> Mẫu Lành Tính
                        </button>
                        <button type="button" class="btn btn-outline-danger btn-sm px-3 rounded-pill fw-semibold" onclick="loadSampleData('malignant')">
                            <i class="fa-solid fa-triangle-exclamation me-1"></i> Mẫu Ác Tính
                        </button>
                        <a href="/docs" target="_blank" class="btn btn-light btn-sm px-3 rounded-pill border fw-semibold text-secondary">
                            <i class="fa-solid fa-code me-1"></i> API Docs
                        </a>
                    </div>
                </div>

                <div class="card p-4 mb-4">
                    <form id="predictionForm">
                        <div class="d-flex align-items-center justify-content-between mb-3">
                            <h5 class="fw-bold text-primary mb-0" style="color: var(--primary-color) !important;">
                                <i class="fa-solid fa-microscope me-2"></i> Thông Số Tế Bào Lâm Sàng (30 Chỉ Số)
                            </h5>
                            <span class="badge bg-light text-secondary border">Đơn vị chuẩn hóa FDA</span>
                        </div>
                        <div class="row g-3" id="featureInputs">
                            <!-- Các chỉ số tiếng Việt được render động -->
                        </div>
                        <div class="mt-4 text-center">
                            <button type="submit" id="btnSubmit" class="btn btn-primary-custom btn-lg px-5 shadow-sm">
                                <i class="fa-solid fa-calculator me-2"></i> Thực Hiện Chẩn Đoán AI
                            </button>
                        </div>
                    </form>
                </div>

                <!-- Khu vực kết quả -->
                <div class="card p-4 shadow-sm mb-5 border-start border-4 border-primary" id="resultCard" style="display: none;">
                    <h4 class="mb-3 fw-bold text-dark"><i class="fa-solid fa-file-medical text-primary me-2"></i> Kết Quả Phân Tích Chuyên Sâu</h4>
                    <div id="resultAlert" class="alert mb-4 py-3 rounded-3" role="alert"></div>
                    <div class="row align-items-center">
                        <div class="col-md-5 border-end-md">
                            <span class="text-muted text-uppercase small fw-bold tracking-wider">Kết luận chẩn đoán mô hình:</span>
                            <h2 id="resPrediction" class="fw-extrabold my-2 display-6"></h2>
                            <p class="small text-muted mb-0 mt-3">
                                <i class="fa-solid fa-circle-info text-primary me-1"></i> 
                                Lưu ý: Kết quả mang tính chất tham khảo lâm sàng hỗ trợ bác sĩ chuyên khoa, không thay thế chẩn đoán giải phẫu bệnh chính thức.
                            </p>
                        </div>
                        <div class="col-md-7 ps-md-4 mt-3 mt-md-0">
                            <div class="mb-3 p-3 bg-light rounded-3">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="fw-semibold text-secondary">Xác suất Lành tính (Benign):</span>
                                    <span id="resBenignText" class="text-success fw-bold fs-5">0%</span>
                                </div>
                                <div class="progress" style="height: 12px;">
                                    <div id="barBenign" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="p-3 bg-light rounded-3">
                                <div class="d-flex justify-content-between mb-2">
                                    <span class="fw-semibold text-secondary">Xác suất Ác tính (Malignant):</span>
                                    <span id="resMalignantText" class="text-danger fw-bold fs-5">0%</span>
                                </div>
                                <div class="progress" style="height: 12px;">
                                    <div id="barMalignant" class="progress-bar bg-danger" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- DANH MỤC 2: LỊCH SỬ KHÁM -->
            <div id="tab-history" class="tab-pane-content">
                <h2 class="fw-bold text-dark mb-1">Lịch Sử Các Ca Chẩn Đoán</h2>
                <p class="text-muted mb-4">Danh sách các ca phân tích đã thực hiện trong phiên làm việc hiện tại.</p>
                <div class="card p-4">
                    <div class="table-responsive">
                        <table class="table table-hover align-middle">
                            <thead class="table-light text-uppercase fs-7">
                                <tr>
                                    <th>Mã Ca Khám</th>
                                    <th>Thời Gian</th>
                                    <th>Kết Luận</th>
                                    <th>Xác Suất Ác Tính</th>
                                    <th>Thao Tác</th>
                                </tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <tr>
                                    <td colspan="5" class="text-center text-muted py-5">Chưa có lịch sử chẩn đoán nào được ghi nhận.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- DANH MỤC 3: THỐNG KÊ & BÁO CÁO -->
            <div id="tab-stats" class="tab-pane-content">
                <h2 class="fw-bold text-dark mb-1">Thống Kê Hệ Thống & Mô Hình</h2>
                <p class="text-muted mb-4">Tổng quan hiệu suất phân loại và thông số cấu hình mô hình SVM.</p>
                <div class="row g-4 mb-4">
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-primary border-4">
                            <span class="text-muted fw-semibold small text-uppercase">Tổng Số Ca Phân Tích</span>
                            <h2 id="statTotal" class="text-primary fw-bold mt-2 display-5">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-success border-4">
                            <span class="text-muted fw-semibold small text-uppercase">Ca Lành Tính (Benign)</span>
                            <h2 id="statBenign" class="text-success fw-bold mt-2 display-5">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-4 text-center border-top border-danger border-4">
                            <span class="text-muted fw-semibold small text-uppercase">Ca Ác Tính (Malignant)</span>
                            <h2 id="statMalignant" class="text-danger fw-bold mt-2 display-5">0</h2>
                        </div>
                    </div>
                </div>
                <div class="card p-4">
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-server me-2 text-primary"></i> Thông Tin Kỹ Thuật Mô Hình</h5>
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item d-flex justify-content-between py-3"><span>Thuật toán cốt lõi:</span> <strong class="text-dark">Support Vector Machine (SVM Classifier)</strong></li>
                        <li class="list-group-item d-flex justify-content-between py-3"><span>Phiên bản triển khai:</span> <strong id="modelVersionDisplay" class="text-dark">1.0.0</strong></li>
                        <li class="list-group-item d-flex justify-content-between py-3"><span>Không gian đặc trưng:</span> <strong class="text-dark">30 chiều hình thái học nhân tế bào</strong></li>
                        <li class="list-group-item d-flex justify-content-between py-3"><span>Môi trường chạy:</span> <strong class="text-dark">FastAPI + Render Cloud Infrastructure</strong></li>
                    </ul>
                </div>
            </div>

            <!-- DANH MỤC 4: HƯỚNG DẪN CHỈ SỐ -->
            <div id="tab-guide" class="tab-pane-content">
                <h2 class="fw-bold text-dark mb-1">Tra Cứu Ý Nghĩa 30 Chỉ Số Lâm Sàng</h2>
                <p class="text-muted mb-4">Giải thích chi tiết các đặc trưng hình thái học trích xuất từ ảnh chụp tế bào học kim nhỏ (FNA).</p>
                <div class="card p-4">
                    <div class="row g-3" id="guideContainer">
                        <!-- Danh sách tra cứu -->
                    </div>
                </div>
            </div>

        </div>

        <script>
            const featureVietnameseMap = {
                "mean radius": { name: "Bán kính trung bình", desc: "Kích thước trung bình của các tế bào" },
                "mean texture": { name: "Độ mịn/Kết cấu bề mặt trung bình", desc: "Độ chuẩn sai của giá trị xám trên bề mặt" },
                "mean perimeter": { name: "Chu vi trung bình", desc: "Chu vi trung bình của nhân tế bào" },
                "mean area": { name: "Diện tích trung bình", desc: "Diện tích trung bình của nhân tế bào" },
                "mean smoothness": { name: "Độ trơn nhẵn trung bình", desc: "Độ biến thiên cục bộ của bán kính" },
                "mean compactness": { name: "Độ nén trung bình", desc: "Tính theo (chu vi^2 / diện tích - 1.0)" },
                "mean concavity": { name: "Độ lõm trung bình", desc: "Mức độ sâu của các phần lõm trên viền nhân" },
                "mean concave points": { name: "Số điểm lõm trung bình", desc: "Số lượng các phần lõm trên đường viền" },
                "mean symmetry": { name: "Độ đối xứng trung bình", desc: "Mức độ cân đối hình dáng nhân tế bào" },
                "mean fractal dimension": { name: "Độ chiều Fractal trung bình", desc: "Độ phức tạp bờ viền" },
                "radius error": { name: "Sai số bán kính", desc: "Độ lệch chuẩn của kích thước bán kính" },
                "texture error": { name: "Sai số kết cấu", desc: "Độ lệch chuẩn của độ mịn bề mặt" },
                "perimeter error": { name: "Sai số chu vi", desc: "Độ lệch chuẩn của chu vi nhân tế bào" },
                "area error": { name: "Sai số diện tích", desc: "Độ lệch chuẩn của diện tích tế bào" },
                "smoothness error": { name: "Sai số độ trơn nhẵn", desc: "Độ biến thiên sai số độ mịn" },
                "compactness error": { name: "Sai số độ nén", desc: "Sai số mức độ nén tế bào" },
                "concavity error": { name: "Sai số độ lõm", desc: "Sai số độ sâu phần lõm viền nhân" },
                "concave points error": { name: "Sai số điểm lõm", desc: "Sai số số lượng điểm lõm trên viền" },
                "symmetry error": { name: "Sai số độ đối xứng", desc: "Sai số mức độ cân đối" },
                "fractal dimension error": { name: "Sai số chiều Fractal", desc: "Sai số độ phức tạp bờ viền" },
                "worst radius": { name: "Bán kính lớn nhất (Worst)", desc: "Giá trị trung bình của 3 giá trị lớn nhất" },
                "worst texture": { name: "Kết cấu lớn nhất (Worst)", desc: "Giá trị kết cấu lớn nhất trong mẫu" },
                "worst perimeter": { name: "Chu vi lớn nhất (Worst)", desc: "Chu vi lớn nhất ghi nhận được" },
                "worst area": { name: "Diện tích lớn nhất (Worst)", desc: "Diện tích lớn nhất ghi nhận được" },
                "worst smoothness": { name: "Độ trơn nhẵn lớn nhất (Worst)", desc: "Độ trơn nhẵn cao nhất của nhóm lớn nhất" },
                "worst compactness": { name: "Độ nén lớn nhất (Worst)", desc: "Độ nén cao nhất của nhóm lớn nhất" },
                "worst concavity": { name: "Độ lõm lớn nhất (Worst)", desc: "Độ lõm sâu nhất ghi nhận trong mẫu" },
                "worst concave points": { name: "Số điểm lõm lớn nhất (Worst)", desc: "Số điểm lõm nhiều nhất ghi nhận" },
                "worst symmetry": { name: "Độ đối xứng lớn nhất (Worst)", desc: "Độ lệch đối xứng lớn nhất" },
                "worst fractal dimension": { name: "Chiều Fractal lớn nhất (Worst)", desc: "Độ phức tạp bờ viền lớn nhất" }
            };

            const sampleMalignant = {
                "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8, "mean area": 1001.0,
                "mean smoothness": 0.1184, "mean compactness": 0.2776, "mean concavity": 0.3001, "mean concave points": 0.1471,
                "mean symmetry": 0.2419, "mean fractal dimension": 0.07871, "radius error": 1.095, "texture error": 0.9053,
                "perimeter error": 8.589, "area error": 153.4, "smoothness error": 0.006399, "compactness error": 0.04904,
                "concavity error": 0.05373, "concave points error": 0.01587, "symmetry error": 0.03003, "fractal dimension error": 0.006193,
                "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0,
                "worst smoothness": 0.1622, "worst compactness": 0.6656, "worst concavity": 0.7119, "worst concave points": 0.2654,
                "worst symmetry": 0.4601, "worst fractal dimension": 0.1189
            };

            const sampleBenign = {
                "mean radius": 13.54, "mean texture": 14.36, "mean perimeter": 87.46, "mean area": 566.3,
                "mean smoothness": 0.09779, "mean compactness": 0.08129, "mean concavity": 0.06664, "mean concave points": 0.04781,
                "mean symmetry": 0.1885, "mean fractal dimension": 0.05766, "radius error": 0.2699, "texture error": 0.7886,
                "perimeter error": 2.058, "area error": 23.56, "smoothness error": 0.008462, "compactness error": 0.0146,
                "concavity error": 0.02387, "concave points error": 0.01315, "symmetry error": 0.0198, "fractal dimension error": 0.0023,
                "worst radius": 15.11, "worst texture": 19.26, "worst perimeter": 99.7, "worst area": 711.2,
                "worst smoothness": 0.144, "worst compactness": 0.1773, "worst concavity": 0.239, "worst concave points": 0.1288,
                "worst symmetry": 0.2977, "worst fractal dimension": 0.07259
            };

            let featureNames = [];
            let diagnosisHistory = [];
            let stats = { total: 0, benign: 0, malignant: 0 };

            async function fetchMetadata() {
                try {
                    const res = await fetch('/metadata');
                    const data = await res.json();
                    featureNames = data.feature_names;
                    document.getElementById('modelVersionDisplay').innerText = data.model_version;
                    renderInputs();
                    renderGuide();
                } catch (e) {
                    console.error("Không tải được metadata", e);
                }
            }

            function renderInputs() {
                const formContainer = document.getElementById('featureInputs');
                formContainer.innerHTML = '';
                featureNames.forEach((feature) => {
                    const info = featureVietnameseMap[feature] || { name: feature, desc: "Chỉ số tế bào học" };
                    const defaultVal = sampleBenign[feature] !== undefined ? sampleBenign[feature] : 10.0;
                    formContainer.innerHTML += `
                        <div class="col-md-4">
                            <div class="feature-card h-100">
                                <label class="form-label text-truncate fw-bold small text-dark d-block mb-1" title="${info.name}">${info.name}</label>
                                <div class="text-muted small mb-2" style="font-size: 0.75rem; height: 2.4em; overflow: hidden;" title="${info.desc}">${info.desc}</div>
                                <input type="text" class="form-control form-control-sm" name="${feature}" id="input_${feature.replace(/\s+/g, '_')}" required value="${defaultVal}">
                            </div>
                        </div>
                    `;
                });
            }

            function renderGuide() {
                const guideContainer = document.getElementById('guideContainer');
                guideContainer.innerHTML = '';
                Object.keys(featureVietnameseMap).forEach((key, idx) => {
                    const item = featureVietnameseMap[key];
                    guideContainer.innerHTML += `
                        <div class="col-md-6">
                            <div class="p-3 border rounded-3 bg-light h-100">
                                <h6 class="fw-bold text-dark mb-1">${idx + 1}. ${item.name}</h6>
                                <p class="small text-muted mb-1"><em>Tên kỹ thuật:</em> <code>${key}</code></p>
                                <p class="small mb-0 text-secondary">${item.desc}</p>
                            </div>
                        </div>
                    `;
                });
            }

            function loadSampleData(type) {
                const targetSample = type === 'malignant' ? sampleMalignant : sampleBenign;
                featureNames.forEach(feature => {
                    const inputEl = document.getElementById(`input_${feature.replace(/\s+/g, '_')}`);
                    if (inputEl && targetSample[feature] !== undefined) {
                        inputEl.value = targetSample[feature];
                    }
                });
            }

            function switchTab(tabId, event) {
                if (event) event.preventDefault();
                document.querySelectorAll('.tab-pane-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.sidebar .nav-link').forEach(el => el.classList.remove('active'));
                
                document.getElementById(tabId).classList.add('active');
                if (event && event.currentTarget) {
                    event.currentTarget.classList.add('active');
                }
            }

            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const btnSubmit = document.getElementById('btnSubmit');
                btnSubmit.disabled = true;
                btnSubmit.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Đang phân tích...';

                const formData = new FormData(this);
                const featuresObj = {};
                formData.forEach((value, key) => {
                    let cleanVal = String(value).trim().replace(',', '.');
                    featuresObj[key] = parseFloat(cleanVal);
                });

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features: featuresObj })
                    });
                    const data = await response.json();
                    
                    const resultCard = document.getElementById('resultCard');
                    const resultAlert = document.getElementById('resultAlert');
                    resultCard.style.display = 'block';

                    const pBenign = (data.probability_benign * 100);
                    const pMalignant = (data.probability_malignant * 100);

                    document.getElementById('barBenign').style.width = pBenign + '%';
                    document.getElementById('barMalignant').style.width = pMalignant + '%';
                    document.getElementById('resBenignText').innerText = pBenign.toFixed(2) + '%';
                    document.getElementById('resMalignantText').innerText = pMalignant.toFixed(2) + '%';

                    let diagnosisTime = new Date().toLocaleTimeString();
                    let record = {
                        id: 'CA-' + Math.floor(1000 + Math.random() * 9000),
                        time: diagnosisTime,
                        label: data.predicted_label,
                        malignantProb: pMalignant.toFixed(2) + '%'
                    };

                    stats.total++;
                    if (data.predicted_label === 'benign') {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-success">LÀNH TÍNH (BENIGN)</span>';
                        resultAlert.className = 'alert alert-success border-0 bg-success bg-opacity-10 text-success fw-semibold py-3';
                        resultAlert.innerHTML = '<i class="fa-solid fa-circle-check me-2"></i> <strong>Kết quả an toàn:</strong> Khối u mang đặc điểm lành tính, không có dấu hiệu ác tính đáng kể.';
                        stats.benign++;
                        record.badgeClass = 'bg-success';
                        record.vietLabel = 'Lành tính';
                    } else {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-danger">ÁC TÍNH (MALIGNANT)</span>';
                        resultAlert.className = 'alert alert-danger border-0 bg-danger bg-opacity-10 text-danger fw-semibold py-3';
                        resultAlert.innerHTML = '<i class="fa-solid fa-triangle-exclamation me-2"></i> <strong>CẢNH BÁO:</strong> Khối u có xu hướng ác tính cao. Khuyến nghị bệnh nhân thực hiện sinh thiết và hội chẩn chuyên khoa ngay.';
                        stats.malignant++;
                        record.badgeClass = 'bg-danger';
                        record.vietLabel = 'Ác tính';
                    }

                    document.getElementById('statTotal').innerText = stats.total;
                    document.getElementById('statBenign').innerText = stats.benign;
                    document.getElementById('statMalignant').innerText = stats.malignant;

                    diagnosisHistory.unshift(record);
                    updateHistoryTable();

                    // Cuộn mượt xuống khu vực kết quả
                    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

                } catch (err) {
                    alert('Lỗi kết nối đến server API!');
                } finally {
                    btnSubmit.disabled = false;
                    btnSubmit.innerHTML = '<i class="fa-solid fa-calculator me-2"></i> Thực Hiện Chẩn Đoán AI';
                }
            });

            function updateHistoryTable() {
                const tbody = document.getElementById('historyTableBody');
                if (diagnosisHistory.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-5">Chưa có lịch sử chẩn đoán nào được ghi nhận.</td></tr>';
                    return;
                }
                tbody.innerHTML = '';
                diagnosisHistory.forEach(item => {
                    tbody.innerHTML += `
                        <tr>
                            <td><strong class="text-dark">${item.id}</strong></td>
                            <td class="text-secondary">${item.time}</td>
                            <td><span class="badge ${item.badgeClass} px-3 py-2">${item.vietLabel}</span></td>
                            <td class="fw-bold text-dark">${item.malignantProb}</td>
                            <td><button class="btn btn-sm btn-outline-primary px-3 rounded-pill" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-eye me-1"></i> Xem</button></td>
                        </tr>
                    `;
                });
            }

            fetchMetadata();
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": metadata["model_version"]
    }

@app.get("/metadata")
def get_metadata():
    return metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    x = build_vector(payload)
    predicted_class = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    
    classes = list(model.named_steps["svc"].classes_ )
    p = {int(c): float(v) for c, v in zip(classes, probabilities)}

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata["class_mapping"][str(predicted_class)],
        probability_malignant=p.get(0, 0.0),
        probability_benign=p.get(1, 0.0),
        model_version=metadata["model_version"],
        warning=metadata["warning"]
    )
