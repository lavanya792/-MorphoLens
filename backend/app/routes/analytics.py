from fastapi import APIRouter, HTTPException
from backend.app.services.analytics import run_analytics

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

@router.post("/run/{analysis_id}")
def execute_analytics(analysis_id: int):
    try:
        return run_analytics(analysis_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {exc}")
