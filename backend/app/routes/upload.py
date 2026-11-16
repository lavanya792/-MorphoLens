from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import Analysis, NucleusFeature

from backend.app.services.segmentation import segmenter
from backend.app.services.instance_segmentation import separate_nuclei
from backend.app.services.feature_extraction import extract_features


router = APIRouter(
    prefix="/api/upload",
    tags=["Image Analysis"]
)

BASE_DIR = Path(__file__).resolve().parents[3]

UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
OUTPUT_DIR = BASE_DIR / "backend" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff"
}

MAX_FILE_SIZE = 20 * 1024 * 1024


@router.post("/analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    image_id = str(uuid4())

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Empty image uploaded"
        )

    if len(image_bytes) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 20 MB"
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid microscopic image"
        )

    if image.shape[0] < 32 or image.shape[1] < 32:

        raise HTTPException(
            status_code=400,
            detail="Image dimensions are too small"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_path = UPLOAD_DIR / f"{image_id}{extension}"

    mask_path = OUTPUT_DIR / f"{image_id}_mask.png"

    instance_path = OUTPUT_DIR / f"{image_id}_instances.png"

    overlay_path = OUTPUT_DIR / f"{image_id}_overlay.png"

    try:

        # Save original image
        if not cv2.imwrite(
            str(original_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError("Could not save original image")

        # STEP 1: U-Net++ segmentation
        binary_mask = segmenter.predict(image)

        # STEP 2: Save binary mask
        cv2.imwrite(
            str(mask_path),
            binary_mask * 255
        )

        # STEP 3: Watershed instance segmentation
        instances = separate_nuclei(binary_mask)

        total_nuclei = int(instances.max())

        # STEP 4: Feature extraction
        features = extract_features(
            image,
            instances
        )

        # STEP 5: Create colored visualization
        rng = np.random.default_rng(42)

        colors = rng.integers(
            50, 256,
            size=(max(total_nuclei + 1, 1), 3),
            dtype=np.uint8
        )

        colors[0] = 0

        colored_instances = colors[instances]

        overlay = cv2.addWeighted(
            image,
            0.65,
            colored_instances,
            0.35,
            0
        )

        # Save instance map and overlay
        cv2.imwrite(
            str(instance_path),
            instances.astype(np.uint16)
        )

        cv2.imwrite(
            str(overlay_path),
            cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )
        )

        # STEP 6: Store analysis in SQLite
        analysis = Analysis(
            image_name=file.filename,
            image_path=str(original_path),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            segmentation_model="U-Net++ ResNet34",
            total_nuclei=total_nuclei,
            status="completed"
        )

        db.add(analysis)
        db.flush()

        # STEP 7: Store individual nucleus measurements
        records = []

        for feature in features:

            feature["sample_id"] = image_id
            feature["analysis_id"] = analysis.id

            records.append(
                NucleusFeature(**feature)
            )

        db.add_all(records)

        db.commit()

        return {

            "success": True,

            "analysis_id": analysis.id,

            "image_id": image_id,

            "filename": file.filename,

            "width": int(image.shape[1]),

            "height": int(image.shape[0]),

            "segmentation_model": "U-Net++ ResNet34",

            "device": str(segmenter.device),

            "total_nuclei": total_nuclei,

            "feature_count": len(features),

            "images": {

                "original": f"/api/upload/image/{image_id}",

                "mask": f"/api/upload/mask/{image_id}",

                "instances": f"/api/upload/instances/{image_id}",

                "overlay": f"/api/upload/overlay/{image_id}"

            },

            "features": features[:10]

        }

    except Exception as exc:

        db.rollback()

        for path in [
            original_path,
            mask_path,
            instance_path,
            overlay_path
        ]:
            path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(exc)}"
        )

    finally:

        await file.close()


def get_output_file(image_id, suffix):

    if not image_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid image ID")

    path = OUTPUT_DIR / f"{image_id}_{suffix}"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image result not found"
        )

    return FileResponse(path)


@router.get("/mask/{image_id}")
def get_mask(image_id: str):

    return get_output_file(
        image_id,
        "mask.png"
    )


@router.get("/instances/{image_id}")
def get_instances(image_id: str):

    return get_output_file(
        image_id,
        "instances.png"
    )


@router.get("/overlay/{image_id}")
def get_overlay(image_id: str):

    return get_output_file(
        image_id,
        "overlay.png"
    )