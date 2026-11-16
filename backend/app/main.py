from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes.upload import router as upload_router
from backend.app.routes.dashboard import router as dashboard_router
from backend.app.routes.analytics import router as analytics_router
app = FastAPI(
    title="MorphoLens API",
    description="Deep Learning-Based Microscopic Morphology Analytics and Anomaly Detection",
    version="1.0.0"
)
# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# Register API routers
app.include_router(upload_router)
app.include_router(dashboard_router)
app.include_router(analytics_router)
@app.get("/")
def home():
    return {
        "success": True,
        "message": "Welcome to MorphoLens API",
        "version": "1.0.0",
        "modules": [
            "Image Upload",
            "U-Net++ Segmentation",
            "Watershed Instance Segmentation",
            "Morphological Feature Extraction",
            "PCA",
            "UMAP",
            "K-Means Clustering",
            "DBSCAN Clustering",
            "Isolation Forest Anomaly Detection",
            "Dashboard Analytics"
        ]
    }
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "MorphoLens"
    }