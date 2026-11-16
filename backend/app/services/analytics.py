from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parents[3]
DB_PATH = BASE_DIR / "backend" / "data" / "morpholens.db"

FEATURE_COLUMNS = [
    "area", "perimeter", "circularity", "eccentricity", "aspect_ratio",
    "solidity", "extent", "major_axis_length", "minor_axis_length",
    "mean_intensity", "std_intensity", "min_intensity", "max_intensity",
    "glcm_contrast", "glcm_correlation", "glcm_energy",
    "glcm_homogeneity", "glcm_dissimilarity"
]

def run_analytics(analysis_id: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM nuclei_features WHERE analysis_id=? ORDER BY nucleus_id",
            conn, params=(analysis_id,)
        )
        if df.empty:
            raise ValueError(f"No nuclei found for analysis_id={analysis_id}")
        if len(df) < 4:
            raise ValueError("At least 4 nuclei are required for PCA and clustering.")

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Database is missing feature columns: {missing}")

        X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0.0)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(Xs)

        k = min(3, len(df) - 1)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(Xs)

        # DBSCAN is exploratory and has no native predict() for future samples.
        db = DBSCAN(eps=2.0, min_samples=min(10, len(df)))
        db_labels = db.fit_predict(Xs)

        iso = IsolationForest(contamination=0.03, random_state=42, n_estimators=200)
        iso_pred = iso.fit_predict(Xs)
        iso_score = iso.decision_function(Xs)
        anomaly = np.where(iso_pred == -1, "Anomaly", "Normal")

        cursor = conn.cursor()
        for i, row in df.reset_index(drop=True).iterrows():
            cursor.execute("""
                UPDATE nuclei_features
                SET pc1=?, pc2=?, kmeans_cluster=?, dbscan_cluster=?,
                    anomaly_status=?, anomaly_score=?
                WHERE id=?
            """, (
                float(pcs[i, 0]), float(pcs[i, 1]),
                int(km_labels[i]), int(db_labels[i]),
                str(anomaly[i]), float(iso_score[i]), int(row["id"])
            ))

        # UMAP is optional: if installed, compute coordinates for this analysis.
        umap_status = "not_installed"
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42,
                                n_neighbors=min(15, max(2, len(df)-1)))
            embedding = reducer.fit_transform(Xs)
            for i, row in df.reset_index(drop=True).iterrows():
                cursor.execute(
                    "UPDATE nuclei_features SET umap1=?, umap2=? WHERE id=?",
                    (float(embedding[i, 0]), float(embedding[i, 1]), int(row["id"]))
                )
            umap_status = "computed"
        except ImportError:
            pass

        unique_k = np.unique(km_labels)
        sil = None
        if 1 < len(unique_k) < len(df):
            sil = float(silhouette_score(Xs, km_labels))

        cursor.execute("DELETE FROM analysis_metrics WHERE analysis_id=?", (analysis_id,))
        metrics = {
            "pca_variance_pc1": float(pca.explained_variance_ratio_[0]),
            "pca_variance_pc2": float(pca.explained_variance_ratio_[1]),
            "pca_total_variance": float(pca.explained_variance_ratio_.sum()),
            "kmeans_cluster_count": int(len(unique_k)),
            "dbscan_cluster_count": int(len(set(db_labels) - {-1})),
            "dbscan_noise_count": int(np.sum(db_labels == -1)),
            "anomaly_count": int(np.sum(iso_pred == -1)),
            "normal_count": int(np.sum(iso_pred == 1)),
        }
        if sil is not None:
            metrics["kmeans_silhouette"] = sil

        for name, value in metrics.items():
            cursor.execute("""
                INSERT INTO analysis_metrics
                (analysis_id, metric_name, metric_value, model_name)
                VALUES (?, ?, ?, ?)
            """, (analysis_id, name, value, "MorphoLens"))

        conn.commit()
        return {
            "success": True,
            "analysis_id": analysis_id,
            "total_nuclei": int(len(df)),
            "umap_status": umap_status,
            "metrics": metrics,
            "nuclei": [{
                "nucleus_id": int(df.iloc[i]["nucleus_id"]),
                "pc1": float(pcs[i, 0]), "pc2": float(pcs[i, 1]),
                "kmeans_cluster": int(km_labels[i]),
                "dbscan_cluster": int(db_labels[i]),
                "anomaly_status": str(anomaly[i]),
                "anomaly_score": float(iso_score[i])
            } for i in range(len(df))]
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
