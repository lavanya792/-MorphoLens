from pathlib import Path
import sqlite3
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import APIRouter, HTTPException, Query

BASE_DIR = Path(__file__).resolve().parents[3]
DB_PATH = BASE_DIR / "backend" / "data" / "morpholens.db"
router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

def query(sql, params=()):
    conn=sqlite3.connect(DB_PATH)
    conn.row_factory=sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql,params).fetchall()]
    finally:
        conn.close()

@router.get("/stats")
def stats(analysis_id: int | None = Query(default=None)):
    try:
        where=" WHERE analysis_id=?" if analysis_id is not None else ""
        params=(analysis_id,) if analysis_id is not None else ()
        total=query(f"SELECT COUNT(*) AS n FROM nuclei_features{where}",params)[0]["n"]
        anomalies=query(f"SELECT COUNT(*) AS n FROM nuclei_features{where}{' AND' if where else ' WHERE'} anomaly_status='Anomaly'",params)[0]["n"]
        normal=query(f"SELECT COUNT(*) AS n FROM nuclei_features{where}{' AND' if where else ' WHERE'} anomaly_status='Normal'",params)[0]["n"]
        analyses=query("SELECT COUNT(*) AS n FROM analyses")[0]["n"]
        return {"total_analyses":analyses,"analysis_id":analysis_id,"total_nuclei":total,
                "normal_nuclei":normal,"anomalies":anomalies,
                "anomaly_percentage":round(anomalies/total*100,2) if total else 0}
    except Exception as exc:
        raise HTTPException(status_code=500,detail=str(exc))

@router.get("/analyses")
def list_analyses():
    try:
        return query("""SELECT id AS analysis_id,image_name,width,height,segmentation_model,
                        total_nuclei,status,created_at FROM analyses ORDER BY id DESC""")
    except Exception as exc:
        raise HTTPException(status_code=500,detail=str(exc))

@router.get("/analysis/{analysis_id}")
def analysis_details(analysis_id:int):
    try:
        analyses=query("SELECT id FROM analyses WHERE id=?",(analysis_id,))
        if not analyses: raise HTTPException(status_code=404,detail="Analysis not found")
        nuclei=query("""SELECT nucleus_id,area,perimeter,circularity,eccentricity,aspect_ratio,
                        solidity,extent,major_axis_length,minor_axis_length,centroid_x,centroid_y,
                        mean_intensity,std_intensity,min_intensity,max_intensity,glcm_contrast,
                        glcm_correlation,glcm_energy,glcm_homogeneity,glcm_dissimilarity,
                        pc1,pc2,umap1,umap2,kmeans_cluster,dbscan_cluster,anomaly_status,anomaly_score
                        FROM nuclei_features WHERE analysis_id=? ORDER BY nucleus_id""",(analysis_id,))
        metrics=query("SELECT metric_name,metric_value,model_name FROM analysis_metrics WHERE analysis_id=?",(analysis_id,))
        return {"analysis_id":analysis_id,"nuclei":nuclei,"metrics":metrics}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500,detail=str(exc))


@router.get("/analysis/{analysis_id}/image")
def get_analysis_image(analysis_id: int):
    """Serve the original image using the path stored with this analysis."""
    rows = query("SELECT image_path FROM analyses WHERE id=?", (analysis_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Analysis not found")
    image_path = Path(rows[0]["image_path"])
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Original image file is missing: {image_path.name}")
    return FileResponse(image_path)

@router.delete("/analysis/{analysis_id}")
def delete_analysis(analysis_id: int):
    """Delete one analysis, its nucleus/metric rows, and its associated image/output files."""
    rows = query("SELECT image_path FROM analyses WHERE id=?", (analysis_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Analysis not found")

    image_path = Path(rows[0]["image_path"])
    image_id = image_path.stem

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM nuclei_features WHERE analysis_id=?", (analysis_id,))
        conn.execute("DELETE FROM analysis_metrics WHERE analysis_id=?", (analysis_id,))
        conn.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Remove the original and any output files sharing this generated image ID.
    candidates = [image_path]
    for folder_name in ("uploads", "outputs"):
        folder = BASE_DIR / "backend" / folder_name
        if folder.exists():
            candidates.extend(folder.glob(f"{image_id}*"))
    deleted_files = 0
    for candidate in set(candidates):
        try:
            if candidate.is_file():
                candidate.unlink()
                deleted_files += 1
        except OSError:
            pass

    return {"success": True, "deleted_analysis_id": analysis_id,
            "deleted_files": deleted_files}


def _analysis_output(analysis_id: int, kind: str):
    rows = query("SELECT image_path FROM analyses WHERE id=?", (analysis_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Analysis not found")
    image_stem = Path(rows[0]["image_path"]).stem
    folders = [BASE_DIR / "backend" / "outputs", BASE_DIR / "backend" / "uploads"]
    matches = []
    for folder in folders:
        if folder.exists():
            matches.extend(folder.glob(f"{image_stem}*"))
    # Match by descriptive suffix; filenames are expected to share the upload UUID.
    preferred = [p for p in matches if kind in p.stem.lower()]
    if not preferred:
        raise HTTPException(status_code=404, detail=f"{kind} image not found for this analysis")
    return FileResponse(preferred[0])

@router.get("/analysis/{analysis_id}/mask")
def get_analysis_mask(analysis_id: int):
    return _analysis_output(analysis_id, "mask")

@router.get("/analysis/{analysis_id}/overlay")
def get_analysis_overlay(analysis_id: int):
    return _analysis_output(analysis_id, "overlay")
