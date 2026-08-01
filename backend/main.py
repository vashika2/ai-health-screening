import sys
import os
import base64
sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import shutil, uuid
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import hf_hub_download

WEIGHTS_REPO = "vashika20/ai-health-screening-weights"

def ensure_weights():
    """Downloads model weights from Hugging Face Hub if not already present locally."""
    weights_dir = os.path.join(BASE_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    tb_path = os.path.join(weights_dir, "tb_model_best.pth")
    dr_path = os.path.join(weights_dir, "dr_model_best.pth")

    if not os.path.exists(tb_path):
        print("Downloading TB model weights from Hugging Face...")
        downloaded = hf_hub_download(repo_id=WEIGHTS_REPO, filename="tb_model_best.pth")
        shutil.copy(downloaded, tb_path)
        print("TB weights downloaded!")

    if not os.path.exists(dr_path):
        print("Downloading DR model weights from Hugging Face...")
        downloaded = hf_hub_download(repo_id=WEIGHTS_REPO, filename="dr_model_best.pth")
        shutil.copy(downloaded, dr_path)
        print("DR weights downloaded!")

ensure_weights()

from models.tb_model import TBDetector
from models.dr_model import DRGrader
from utils.preprocess import enhance_xray, enhance_retinal, check_image_quality
from utils.report_generator import generate_tb_report, generate_dr_report

app = FastAPI(
    title="AI Health Screening API",
    description="TB and Diabetic Retinopathy early detection",
    version="1.0.0"
)

# Custom middleware to add CORS headers to every response
class CORSMiddlewareCustom(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            response = JSONResponse(content={})
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            return response
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

app.add_middleware(CORSMiddlewareCustom)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI models...")
tb_detector = TBDetector(weights_path=os.getenv("TB_MODEL_PATH"))
dr_grader   = DRGrader(weights_path=os.getenv("DR_MODEL_PATH"))
print("Models ready!")

@app.get("/")
def root():
    return {"message": "AI Health Screening API is running", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy", "models": ["TB Detector", "DR Grader"]}

@app.post("/predict/tb")
async def predict_tb(
        file: UploadFile = File(...),
        age: str = Form(default=""),
        symptoms: str = Form(default=""),
        duration: str = Form(default="")
):
    file_id = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, f"{file_id}_original.jpg")
    enhanced_path = os.path.join(UPLOAD_DIR, f"{file_id}_enhanced.jpg")
    heatmap_path = os.path.join(UPLOAD_DIR, f"{file_id}_heatmap.jpg")

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    quality = check_image_quality(original_path)
    if not quality["valid"]:
        return JSONResponse(status_code=400, content={"error": quality["reason"]})

    enhance_xray(original_path, enhanced_path)
    prediction = tb_detector.predict(enhanced_path)

    # Generate Grad-CAM heatmap
    tb_detector.generate_heatmap(
        enhanced_path,
        heatmap_path,
        prediction["predicted_class_idx"]
    )

    # Convert heatmap to base64 so frontend can display it directly
    with open(heatmap_path, "rb") as img_file:
        heatmap_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    patient_info = {"age": age, "symptoms": symptoms, "duration": duration}
    report = generate_tb_report(prediction, patient_info)

    return {
        "disease": "Tuberculosis",
        "prediction": prediction,
        "report": report,
        "image_quality": quality,
        "heatmap": f"data:image/jpeg;base64,{heatmap_base64}"
    }
@app.post("/predict/dr")
async def predict_dr(
        file: UploadFile = File(...),
        age: str = Form(default=""),
        diabetes_years: str = Form(default=""),
        hba1c: str = Form(default="")
):
    file_id = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, f"{file_id}_original.jpg")
    enhanced_path = os.path.join(UPLOAD_DIR, f"{file_id}_enhanced.jpg")
    heatmap_path = os.path.join(UPLOAD_DIR, f"{file_id}_heatmap.jpg")

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    quality = check_image_quality(original_path)
    if not quality["valid"]:
        return JSONResponse(status_code=400, content={"error": quality["reason"]})

    enhance_retinal(original_path, enhanced_path)
    prediction = dr_grader.predict(enhanced_path)

    # Generate Grad-CAM heatmap
    dr_grader.generate_heatmap(
        enhanced_path,
        heatmap_path,
        prediction["predicted_class_idx"]
    )

    with open(heatmap_path, "rb") as img_file:
        heatmap_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    patient_info = {"age": age, "diabetes_years": diabetes_years, "hba1c": hba1c}
    report = generate_dr_report(prediction, patient_info)

    return {
        "disease": "Diabetic Retinopathy",
        "prediction": prediction,
        "report": report,
        "image_quality": quality,
        "heatmap": f"data:image/jpeg;base64,{heatmap_base64}"
    }