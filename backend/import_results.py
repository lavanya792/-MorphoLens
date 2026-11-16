import pandas as pd
import numpy as np

from pathlib import Path
from sqlalchemy import select, insert

from backend.app.database.database import SessionLocal
from backend.app.database.models import Analysis, NucleusFeature


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

FEATURES_FILE = RESULTS_DIR / "morphological_features.csv"
ML_FILE = RESULTS_DIR / "morphology_ml_results.csv"
REDUCTION_FILE = RESULTS_DIR / "dimensionality_reduction.csv"


# --------------------------------------------------
# LOAD CSV FILES
# --------------------------------------------------

print("\nLoading CSV files...")

features = pd.read_csv(FEATURES_FILE)
ml_results = pd.read_csv(ML_FILE)
reduction = pd.read_csv(REDUCTION_FILE)

keys = ["sample_id", "nucleus_id"]

print("Morphological features:", len(features))
print("ML results:", len(ml_results))
print("Dimensionality reduction:", len(reduction))


# --------------------------------------------------
# VALIDATE DATA
# --------------------------------------------------

for name, df in [
    ("Features", features),
    ("ML Results", ml_results),
    ("Reduction", reduction)
]:
    if df.duplicated(keys).any():
        raise ValueError(f"Duplicate nucleus identifiers in {name}")

    if df[keys].isna().any().any():
        raise ValueError(f"Missing nucleus identifiers in {name}")


# --------------------------------------------------
# MERGE DATASETS
# --------------------------------------------------

print("\nMerging datasets...")

merged = features.merge(
    ml_results,
    on=keys,
    suffixes=("_features", ""),
    validate="one_to_one"
)

merged = merged.merge(
    reduction,
    on=keys,
    validate="one_to_one"
)

if len(merged) != 7860:
    raise ValueError(
        f"Expected 7860 nuclei, found {len(merged)}"
    )


# --------------------------------------------------
# CHECK MORPHOLOGICAL VALUES
# --------------------------------------------------

print("\nChecking morphological consistency...")

morph_columns = [
    "area", "perimeter", "circularity",
    "eccentricity", "aspect_ratio", "solidity",
    "extent", "major_axis_length", "minor_axis_length",
    "centroid_x", "centroid_y",
    "mean_intensity", "std_intensity",
    "min_intensity", "max_intensity",
    "glcm_contrast", "glcm_correlation",
    "glcm_energy", "glcm_homogeneity",
    "glcm_dissimilarity"
]

for col in morph_columns:

    match = np.isclose(
        merged[f"{col}_features"],
        merged[col],
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True
    )

    if not match.all():
        raise ValueError(
            f"Morphological mismatch in {col}"
        )

print("All morphological values validated.")


# --------------------------------------------------
# PREPARE DATABASE RECORDS
# --------------------------------------------------

print("\nPreparing database records...")

column_mapping = {
    "sample_id": "sample_id",
    "nucleus_id": "nucleus_id",

    "area": "area",
    "perimeter": "perimeter",
    "circularity": "circularity",
    "eccentricity": "eccentricity",
    "aspect_ratio": "aspect_ratio",
    "solidity": "solidity",
    "extent": "extent",
    "major_axis_length": "major_axis_length",
    "minor_axis_length": "minor_axis_length",

    "centroid_x": "centroid_x",
    "centroid_y": "centroid_y",

    "mean_intensity": "mean_intensity",
    "std_intensity": "std_intensity",
    "min_intensity": "min_intensity",
    "max_intensity": "max_intensity",

    "glcm_contrast": "glcm_contrast",
    "glcm_correlation": "glcm_correlation",
    "glcm_energy": "glcm_energy",
    "glcm_homogeneity": "glcm_homogeneity",
    "glcm_dissimilarity": "glcm_dissimilarity",

    "PC1": "pc1",
    "PC2": "pc2",
    "UMAP1": "umap1",
    "UMAP2": "umap2",

    "kmeans_cluster": "kmeans_cluster",
    "dbscan_cluster": "dbscan_cluster",

    "anomaly_status": "anomaly_status",
    "anomaly_score": "anomaly_score"
}


def clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


records = []

for row in merged.to_dict(orient="records"):

    record = {
        db_column: clean_value(row[csv_column])
        for csv_column, db_column in column_mapping.items()
    }

    records.append(record)


# --------------------------------------------------
# INSERT INTO SQLITE
# --------------------------------------------------

print("\nInserting records into SQLite...")

db = SessionLocal()

try:

    existing = db.execute(
        select(Analysis).where(
            Analysis.image_name == "BBBC038_heldout_test_batch"
        )
    ).scalar_one_or_none()

    if existing:
        raise ValueError(
            "This dataset has already been imported. "
            "No records were inserted."
        )

    analysis = Analysis(
        image_name="BBBC038_heldout_test_batch",
        image_path="BBBC038/stage1_train_heldout_test_split",
        segmentation_model="U-Net++",
        total_nuclei=len(records),
        status="completed"
    )

    db.add(analysis)
    db.flush()

    for record in records:
        record["analysis_id"] = analysis.id

    db.execute(
        insert(NucleusFeature),
        records
    )

    db.commit()

    print("\nIMPORT SUCCESSFUL!")
    print("Analysis ID:", analysis.id)
    print("Total nuclei imported:", len(records))

except Exception:

    db.rollback()
    raise

finally:
    db.close()