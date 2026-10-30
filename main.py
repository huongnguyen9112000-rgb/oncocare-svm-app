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
        dtype=float
    )
    if not np.isfinite(values).all():
        raise HTTPException(status_code=422, detail="Features must be finite numbers")
    return values

# Giao diện App Web Đa Danh Mục (Multi-Tab App UI) hoàn chỉnh
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
            body { background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .sidebar { min-height: 100vh; background: #2c3e50; color: white; position: fixed; width: 260px; top: 0; left: 0; z-index: 100; transition: all 0.3s; }
            .sidebar .brand { padding: 20px; font-size: 1.2rem; font-weight: bold; background: #1a252f; text-align: center; border-bottom: 1px solid #34495e; }
            .sidebar .nav-link { color: #bdc3c7; padding: 12px 20px; margin: 5px 15px; border-radius: 8px; transition: 0.2s; cursor: pointer; }
            .sidebar .nav-link:hover, .sidebar .nav-link.active { background: #3498db; color: white; }
            .sidebar .nav-link i { margin-right: 10px; width: 20px; }
            .main-content { margin-left: 260px; padding: 30px; }
            .card { border: none; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
            .tab-pane-content { display: none; }
            .tab-pane-content.active { display: block; }
            .feature-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; transition: 0.2s; }
            .feature-card:hover { border-color: #3498db; box-shadow: 0 2px 8px rgba(52,152,219,0.1); }
        </style>
    </head>
    <body>

        <!-- THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR) -->
        <div class="sidebar d-flex flex-column">
            <div class="brand">
                <i class="fa-solid fa-heart-pulse text-danger"></i> OncoCare AI v1.0
            </div>
            <ul class="nav flex-column mt-3">
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
            <div class="mt-auto p-3 text-center text-muted small border-top border-secondary">
                Hệ thống hỗ trợ lâm sàng SVM<br>© 2026 Phiên bản Y tế
            </div>
        </div>

        <!-- NỘI DUNG CHÍNH (MAIN CONTENT) -->
        <div class="main-content">
            
            <!-- DANH MỤC 1: CHẨN ĐOÁN AI -->
            <div id="tab-predict" class="tab-pane-content active">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <div>
                        <h2>Phân Tích & Chẩn Đoán Ung Thư Vú</h2>
                        <p class="text-muted">Nhập các thông số tế bào học để hệ thống SVM đưa ra dự đoán chuyên sâu.</p>
                    </div>
                    <div>
                        <button type="button" class="btn btn-outline-secondary btn-sm me-2" onclick="loadSampleData()"><i class="fa-solid fa-wand-magic-sparkles"></i> Điền Dữ Liệu Mẫu</button>
                        <a href="/docs" target="_blank" class="btn btn-outline-info btn-sm"><i class="fa-solid fa-code"></i> Tài Liệu API</a>
                    </div>
                </div>

                <div class="card p-4 mb-4">
                    <form id="predictionForm">
                        <h5 class="text-primary mb-3"><i class="fa-solid fa-microscope"></i> Thông Số Tế Bào Lâm Sàng (30 chỉ số chuẩn tiếng Việt)</h5>
                        <div class="row g-3" id="featureInputs">
                            <!-- Các chỉ số tiếng Việt sẽ được sinh tự động ở đây -->
                        </div>
                        <div class="mt-4 text-center">
                            <button type="submit" class="btn btn-primary btn-lg px-5 shadow"><i class="fa-solid fa-calculator"></i> Thực Hiện Chẩn Đoán</button>
                        </div>
                    </form>
                </div>

                <!-- Khu vực kết quả -->
                <div class="card p-4 shadow-sm" id="resultCard" style="display: none;">
                    <h4 class="mb-3 text-secondary"><i class="fa-solid fa-file-medical"></i> Kết Quả Phân Tích Chuyên Sâu</h4>
                    <div id="resultAlert" class="alert mb-3" role="alert"></div>
                    <div class="row">
                        <div class="col-md-6 border-end">
                            <p class="mb-1 text-muted">Kết luận chẩn đoán:</p>
                            <h3 id="resPrediction" class="fw-bold mb-3"></h3>
                            <p class="small text-secondary mb-0"><i class="fa-solid fa-triangle-exclamation text-warning"></i> Lưu ý: Kết quả mang tính chất tham khảo hỗ trợ bác sĩ, không thay thế chẩn đoán lâm sàng tuyệt đối.</p>
                        </div>
                        <div class="col-md-6 ps-md-4 d-flex flex-column justify-content-center">
                            <div class="mb-2">
                                <div class="d-flex justify-content-between mb-1">
                                    <span class="fw-semibold">Xác suất Lành tính (Benign):</span>
                                    <span id="resBenignText" class="text-success fw-bold"></span>
                                </div>
                                <div class="progress" style="height: 10px;">
                                    <div id="barBenign" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="mt-2">
                                <div class="d-flex justify-content-between mb-1">
                                    <span class="fw-semibold">Xác suất Ác tính (Malignant):</span>
                                    <span id="resMalignantText" class="text-danger fw-bold"></span>
                                </div>
                                <div class="progress" style="height: 10px;">
                                    <div id="barMalignant" class="progress-bar bg-danger" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- DANH MỤC 2: LỊCH SỬ KHÁM -->
            <div id="tab-history" class="tab-pane-content">
                <h2 class="mb-3">Lịch Sử Các Ca Chẩn Đoán Gần Đây</h2>
                <div class="card p-4">
                    <div class="table-responsive">
                        <table class="table table-hover align-middle">
                            <thead class="table-light">
                                <tr>
                                    <th># Mã Ca</th>
                                    <th>Thời Gian</th>
                                    <th>Kết Luận</th>
                                    <th>Xác Suất Ác Tính</th>
                                    <th>Thao Tác</th>
                                </tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <tr>
                                    <td colspan="5" class="text-center text-muted py-4">Chưa có lịch sử chẩn đoán nào trong phiên làm việc này.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- DANH MỤC 3: THỐNG KÊ & BÁO CÁO -->
            <div id="tab-stats" class="tab-pane-content">
                <h2 class="mb-3">Thống Kê Hệ Thống & Mô Hình</h2>
                <div class="row g-4">
                    <div class="col-md-4">
                        <div class="card p-3 text-center border-start border-primary border-4">
                            <h6 class="text-muted">Tổng Số Ca Đã Phân Tích</h6>
                            <h2 id="statTotal" class="text-primary fw-bold">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-3 text-center border-start border-success border-4">
                            <h6 class="text-muted">Số Ca Lành Tính (Benign)</h6>
                            <h2 id="statBenign" class="text-success fw-bold">0</h2>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card p-3 text-center border-start border-danger border-4">
                            <h6 class="text-muted">Số Ca Ác Tính (Malignant)</h6>
                            <h2 id="statMalignant" class="text-danger fw-bold">0</h2>
                        </div>
                    </div>
                </div>
                <div class="card p-4 mt-4">
                    <h5>Thông Tin Mô Hình Đang Hoạt Động</h5>
                    <ul class="list-group list-group-flush mt-3">
                        <li class="list-group-item d-flex justify-content-between"><span>Thuật toán:</span> <strong>Support Vector Machine (SVM)</strong></li>
                        <li class="list-group-item d-flex justify-content-between"><span>Phiên bản mô hình:</span> <strong id="modelVersionDisplay">1.0.0</strong></li>
                        <li class="list-group-item d-flex justify-content-between"><span>Đặc trưng đầu vào:</span> <strong>30 chiều chỉ số tế bào nhân</strong></li>
                    </ul>
                </div>
            </div>

            <!-- DANH MỤC 4: HƯỚNG DẪN CHỈ SỐ -->
            <div id="tab-guide" class="tab-pane-content">
                <h2 class="mb-3">Tra Cứu Ý Nghĩa 30 Chỉ Số Lâm Sàng</h2>
                <p class="text-muted">Giải thích chi tiết các đặc trưng hình thái học nhân tế bào được trích xuất từ ảnh chụp tế bào học kim nhỏ (FNA).</p>
                <div class="card p-4">
                    <div class="row g-3" id="guideContainer">
                        <!-- Danh sách giải thích chi tiết tiếng Việt -->
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
                    const randomVal = (Math.random() * 15 + 2).toFixed(2).replace('.', ',');
                    formContainer.innerHTML += `
                        <div class="col-md-4">
                            <div class="feature-card">
                                <label class="form-label text-truncate fw-semibold small text-dark d-block mb-1" title="${info.name}">${info.name}</label>
                                <div class="text-muted small mb-2" style="font-size: 0.75rem; height: 2.2em; overflow: hidden;" title="${info.desc}">${info.desc}</div>
                                <input type="text" class="form-control form-control-sm" name="${feature}" required value="${randomVal}">
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
                            <div class="p-3 border rounded bg-light">
                                <h6 class="text-primary mb-1">${idx + 1}. ${item.name}</h6>
                                <p class="small text-muted mb-1"><em>Tên kỹ thuật:</em> <code>${key}</code></p>
                                <p class="small mb-0">${item.desc}</p>
                            </div>
                        </div>
                    `;
                });
            }

            function loadSampleData() {
                const inputs = document.querySelectorAll('#featureInputs input');
                inputs.forEach(input => {
                    input.value = (Math.random() * 15 + 2).toFixed(2).replace('.', ',');
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
                        resultAlert.className = 'alert alert-success';
                        resultAlert.innerHTML = '<i class="fa-solid fa-circle-check"></i> <strong>Kết quả an toàn:</strong> Khối u mang đặc điểm lành tính, không có dấu hiệu ác tính đáng kể.';
                        stats.benign++;
                        record.badgeClass = 'bg-success';
                        record.vietLabel = 'Lành tính';
                    } else {
                        document.getElementById('resPrediction').innerHTML = '<span class="text-danger">ÁC TÍNH (MALIGNANT)</span>';
                        resultAlert.className = 'alert alert-danger';
                        resultAlert.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> <strong>CẢNH BÁO:</strong> Khối u có xu hướng ác tính cao. Khuyến nghị bệnh nhân thực hiện sinh thiết và hội chẩn chuyên khoa ngay.';
                        stats.malignant++;
                        record.badgeClass = 'bg-danger';
                        record.vietLabel = 'Ác tính';
                    }

                    document.getElementById('statTotal').innerText = stats.total;
                    document.getElementById('statBenign').innerText = stats.benign;
                    document.getElementById('statMalignant').innerText = stats.malignant;

                    diagnosisHistory.unshift(record);
                    updateHistoryTable();

                } catch (err) {
                    alert('Lỗi kết nối đến server API!');
                }
            });

            function updateHistoryTable() {
                const tbody = document.getElementById('historyTableBody');
                if (diagnosisHistory.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-4">Chưa có lịch sử chẩn đoán nào trong phiên làm việc này.</td></tr>';
                    return;
                }
                tbody.innerHTML = '';
                diagnosisHistory.forEach(item => {
                    tbody.innerHTML += `
                        <tr>
                            <td><strong>${item.id}</strong></td>
                            <td>${item.time}</td>
                            <td><span class="badge ${item.badgeClass}">${item.vietLabel}</span></td>
                            <td class="fw-bold">${item.malignantProb}</td>
                            <td><button class="btn btn-sm btn-outline-primary" onclick="switchTab('tab-predict', event)"><i class="fa-solid fa-eye"></i> Xem</button></td>
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
    classes = list(model.named_steps["svc"].classes_)
    p = {int(c): float(v) for c, v in zip(classes, probabilities)}

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata["class_mapping"][str(predicted_class)],
        probability_malignant=p[0],
        probability_benign=p[1],
        model_version=metadata["model_version"],
        warning=metadata["warning"]
    )
