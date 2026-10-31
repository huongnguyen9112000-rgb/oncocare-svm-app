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

# Giao diện App Web Phong Cách Cyber-Medical Dark Mode Độc Lập
@app.get("/", response_class=HTMLResponse)
def home_page():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OncoCare AI - Trung Tâm Phân Tích Khối U Thông Minh</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-core: #030712;
                --bg-card: #0b1120;
                --border-neon: #1e293b;
                --accent-cyan: #06b6d4;
                --accent-emerald: #10b981;
                --accent-rose: #f43f5e;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
            }
            body { background-color: var(--bg-core); font-family: 'Outfit', sans-serif; color: var(--text-main); overflow-x: hidden; }
            
            /* Sidebar Tương Lai */
            .sidebar { min-height: 100vh; background: #050b14; border-right: 1px solid var(--border-neon); position: fixed; width: 280px; top: 0; left: 0; z-index: 100; }
            .sidebar .brand { padding: 30px 24px; font-size: 1.4rem; font-weight: 800; color: #fff; background: #03060c; display: flex; align-items: center; gap: 14px; border-bottom: 1px solid var(--border-neon); letter-spacing: 0.5px; }
            .sidebar .nav-link { color: var(--text-muted); padding: 14px 22px; margin: 8px 16px; border-radius: 12px; font-weight: 600; font-size: 0.95rem; transition: all 0.3s ease; cursor: pointer; display: flex; align-items: center; gap: 14px; border: 1px solid transparent; }
            .sidebar .nav-link:hover, .sidebar .nav-link.active { background: rgba(6, 182, 212, 0.1); color: var(--accent-cyan); border-color: rgba(6, 182, 212, 0.3); box-shadow: 0 0 20px rgba(6, 182, 212, 0.15); transform: translateX(4px); }
            .sidebar .nav-link i { font-size: 1.15rem; width: 24px; text-align: center; }
            
            /* Main Content */
            .main-content { margin-left: 280px; padding: 40px; max-width: 1450px; }
            .card { border: 1px solid var(--border-neon); border-radius: 20px; background: var(--bg-card); box-shadow: 0 10px 30px rgba(0,0,0,0.5); transition: all 0.3s ease; }
            .tab-pane-content { display: none; animation: cyberFade 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
            .tab-pane-content.active { display: block; }
            
            @keyframes cyberFade {
                from { opacity: 0; transform: scale(0.98) translateY(10px); }
                to { opacity: 1; transform: scale(1) translateY(0); }
            }

            /* Feature Card Cyber Style */
            .feature-card { background: #0f172a; border: 1px solid var(--border-neon); border-radius: 14px; padding: 16px; transition: all 0.3s; }
            .feature-card:hover { border-color: var(--accent-cyan); box-shadow: 0 0 15px rgba(6, 182, 212, 0.15); transform: translateY(-2px); }
            .feature-card .form-control { border-radius: 10px; border-color: #334155; font-weight: 700; color: #fff; padding: 10px 14px; background-color: #030712; }
            .feature-card .form-control:focus { border-color: var(--accent-cyan); box-shadow: 0 0 10px rgba(6, 182, 212, 0.3); background-color: #030712; color: #fff; }

            /* Cyber Buttons */
            .btn-cyber { background: linear-gradient(135deg, var(--accent-cyan), #0284c7); border: none; color: white; font-weight: 700; border-radius: 14px; padding: 14px 32px; transition: all 0.3s; box-shadow: 0 4px 20px rgba(6, 182, 212, 0.4); }
            .btn-cyber:hover { opacity: 0.9; box-shadow: 0 6px 25px rgba(6, 182, 212, 0.6); transform: translateY(-2px); color: #fff; }

            /* Result Box Custom */
            .result-cyber { background: linear-gradient(145deg, #0b1120, #030712); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 24px; box-shadow: 0 20px 50px rgba(0,0,0,0.8); }
            
            /* Progress bars */
            .progress { border-radius: 10px; background-color: #030712; overflow: hidden; padding: 3px; border: 1px solid var(--border-neon); }
            .progress-bar { border-radius: 8px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }
            
            /* Table Custom */
            .table-cyber th { font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1px; color: var(--text-muted); background-color: #050b14; border-bottom: 1px solid var(--border-neon); }
            .table-cyber td { padding: 16px 12px; vertical-align: middle; color: var(--text-main); font-weight: 500; border-bottom: 1px solid var(--border-neon); }
        </style>
    </head>
    <body>

        <!-- SIDEBAR -->
        <div class="sidebar d-flex flex-column">
            <div class="brand">
                <div class="text-cyan rounded-3 p-2 d-flex align-items-center justify-content-center shadow-sm" style="background: rgba(6, 182, 212, 0.15); border: 1px solid var(--accent-cyan); width: 42px; height: 42px;">
                    <i class="fa-solid fa-microchip fs-5 text-cyan" style="color: var(--accent-cyan)"></i>
                </div>
                <span>ONCOCARE <span style="color: var(--accent-cyan)">AI</span></span>
            </div>
            <ul class="nav flex-column mt-4">
                <li class="nav-item">
                    <a class="nav-link active" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-brain"></i> Phân Tích SVM</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" onclick="switchTab('tab-history', event)"><i class="fa-solid fa-timeline"></i> Lịch Sử Phiên</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" onclick="switchTab('tab-stats', event)"><i class="fa-solid fa-chart-line"></i> Hiệu Suất Hệ Thống</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" onclick="switchTab('tab-guide', event)"><i class="fa-solid fa-book-atlas"></i> Tra Cứu Chỉ Số</a>
                </li>
            </ul>
            <div class="mt-auto p-4 text-muted small border-top border-dark">
                <div class="fw-bold text-white mb-1"><i class="fa-solid fa-shield-halved text-success me-1"></i> Secure Core</div>
                <div style="font-size: 0.75rem;">Neural SVM v1.0.0 © 2026</div>
            </div>
        </div>

        <!-- MAIN CONTENT -->
        <div class="main-content">
            
            <!-- TAB 1: PHÂN TÍCH SVM -->
            <div id="tab-predict" class="tab-pane-content active">
                <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-dark">
                    <div>
                        <h1 class="fw-extrabold text-white mb-1 fs-3"><i class="fa-solid fa-network-wired text-cyan me-2" style="color: var(--accent-cyan)"></i> Trung Tâm Phân Tích Khối U SVM</h1>
                        <p class="text-muted mb-0">Hệ thống mạng lưới máy vector hỗ trợ chẩn đoán hình thái học tế bào học thời gian thực.</p>
                    </div>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-success btn-sm px-3 rounded-pill fw-bold" onclick="loadSampleData('benign')">
                            <i class="fa-solid fa-shield-heart me-1"></i> Mẫu Lành Tính
                        </button>
                        <button type="button" class="btn btn-outline-danger btn-sm px-3 rounded-pill fw-bold" onclick="loadSampleData('malignant')">
                            <i class="fa-solid fa-triangle-exclamation me-1"></i> Mẫu Ác Tính
                        </button>
                        <a href="/docs" target="_blank" class="btn btn-outline-info btn-sm px-3 rounded-pill fw-bold">
                            <i class="fa-solid fa-code me-1"></i> API Docs
                        </a>
                    </div>
                </div>

                <div class="card p-4 mb-4">
                    <form id="predictionForm">
                        <div class="d-flex align-items-center justify-content-between mb-4">
                            <h5 class="fw-bold text-white mb-0 d-flex align-items-center gap-2">
                                <span class="badge px-3 py-2 rounded-pill font-monospace" style="background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.3)">30 Đặc Trưng Đầu Vào</span>
                            </h5>
                            <span class="text-muted small">Nhập hoặc chọn mẫu tự động để chạy thuật toán</span>
                        </div>
                        <div class="row g-3" id="featureInputs">
                            <!-- Render động bằng JS -->
                        </div>
                        <div class="mt-5 text-center">
                            <button type="submit" id="btnSubmit" class="btn btn-cyber btn-lg px-5">
                                <i class="fa-solid fa-wand-magic-sparkles me-2"></i> Khởi Chạy Thuật Toán SVM
                            </button>
                        </div>
                    </form>
                </div>

                <!-- KẾT QUẢ PHÂN TÍCH -->
                <div class="result-cyber p-4 p-md-5 mb-5" id="resultCard" style="display: none;">
                    <div class="d-flex align-items-center justify-content-between mb-4 pb-3 border-bottom border-dark">
                        <h3 class="fw-bold text-white mb-0 d-flex align-items-center gap-2">
                            <i class="fa-solid fa-chart-pie text-cyan" style="color: var(--accent-cyan)"></i> Báo Cáo Kết Quả SVM
                        </h3>
                        <span class="badge bg-dark text-cyan border border-cyan px-3 py-2 rounded-pill font-monospace">STATUS: ANALYZED</span>
                    </div>
                    
                    <div id="resultAlert" class="alert mb-4 py-3 rounded-4 fw-semibold border-0 shadow-sm" role="alert"></div>
                    
                    <div class="row align-items-center g-4">
                        <div class="col-lg-5 border-end border-dark pe-lg-4">
                            <span class="text-muted text-uppercase small fw-bold tracking-wider">Kết luận phân loại:</span>
