"""
Vercel Serverless Gateway powered by FastAPI.
Routes incoming HTTP requests on Vercel to the Python ML engine and serves the Web Dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
from pathlib import Path
import json

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.predict import Predictor
from src.data_loader import DataLoader
from src.config import METRICS_JSON_PATH, EVALUATION_REPORT_PDF_PATH
from src.utils import load_json

app = FastAPI(
    title="RealEstate.AI API",
    description="Vercel Serverless API for House Price Prediction and Analytics",
    version="1.0.0"
)

# Enable CORS for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
static_dir = root_dir / "static"
if static_dir.exists():
    css_dir = static_dir / "css"
    js_dir = static_dir / "js"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")


class PredictionInput(BaseModel):
    bedrooms: int = 3
    bathrooms: float = 2.25
    sqft_living: int = 2100
    sqft_lot: int = 7500
    floors: float = 2.0
    waterfront: int = 0
    view: int = 0
    condition: int = 3
    grade: int = 7
    sqft_above: int = 1600
    sqft_basement: int = 500
    yr_built: int = 1985
    yr_renovated: int = 0
    zipcode: int = 98052
    lat: float = 47.560
    long: float = -122.213
    sqft_living15: int = 1900
    sqft_lot15: int = 7500


@app.get("/")
def serve_home():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "online", "message": "RealEstate.AI Vercel Application"}


@app.get("/api/health")
def health_check():
    return {"status": "online", "service": "House Price Prediction API", "platform": "Vercel Serverless"}


@app.get("/api/metrics")
def get_metrics():
    metrics = load_json(METRICS_JSON_PATH)
    if not metrics:
        return {"message": "Metrics not found. Please train models."}
    return metrics


@app.get("/api/eda")
def get_eda():
    try:
        loader = DataLoader()
        meta = loader.get_metadata()
        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
def predict_price(data: PredictionInput):
    try:
        predictor = Predictor()
        input_dict = data.model_dump()
        result = predictor.predict_single(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/report")
def download_pdf_report():
    if EVALUATION_REPORT_PDF_PATH.exists():
        return FileResponse(
            path=str(EVALUATION_REPORT_PDF_PATH),
            filename="evaluation_report.pdf",
            media_type="application/pdf"
        )
    raise HTTPException(status_code=404, detail="PDF evaluation report not found.")
