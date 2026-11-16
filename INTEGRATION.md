# MorphoLens remaining implementation

## 1. Add backend files
Copy `backend/app/services/analytics.py` and `backend/app/routes/analytics.py` into the existing project.
Replace `backend/app/routes/dashboard.py` with the included version only if you want its three dashboard endpoints.

## 2. Register analytics router
In `backend/app/main.py`, add:
```python
from backend.app.routes.analytics import router as analytics_router
app.include_router(analytics_router)
```
Keep your existing upload and dashboard router registrations. Do not replace your existing `main.py`.

## 3. Install optional UMAP
In the active `(morpholens)` environment:
```bash
python -m pip install umap-learn
```
UMAP is optional. Without it, PCA, K-Means, DBSCAN and Isolation Forest still run. If installed, UMAP coordinates are written to the existing `umap1` and `umap2` columns.

## 4. Run backend
From the MorphoLens project root:
```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```
Open `http://127.0.0.1:8000/docs`.
Upload an image using the existing `/api/upload/analyze` endpoint, then run:
`POST /api/analytics/run/{analysis_id}`

## 5. Run frontend
Open a second terminal:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`.

## Important scientific notes
- Analytics currently fit StandardScaler, PCA, K-Means, DBSCAN and Isolation Forest independently for each analysis. Therefore cluster IDs and anomaly scores are local to that analysis and are not comparable across separate uploads.
- This is a demo/integration implementation, not a validated diagnostic model.
- The existing database is accessed with SQLite directly; no schema migration is performed.
- The instance label PNG should remain a uint16 label map for computation. For a browser-friendly display, use a separately colorized RGB visualization rather than converting the label map and losing IDs.
