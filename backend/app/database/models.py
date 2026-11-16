from sqlalchemy import (
    Column, Integer, String, Float, ForeignKey,
    DateTime, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("Analysis", back_populates="user")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    image_name = Column(String(255))
    image_path = Column(String(500))

    width = Column(Integer)
    height = Column(Integer)

    segmentation_model = Column(String(100))
    total_nuclei = Column(Integer, default=0)

    status = Column(String(50), default="completed")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")
    nuclei = relationship(
        "NucleusFeature",
        back_populates="analysis",
        cascade="all, delete-orphan"
    )

    metrics = relationship(
        "AnalysisMetric",
        back_populates="analysis",
        cascade="all, delete-orphan"
    )


class NucleusFeature(Base):
    __tablename__ = "nuclei_features"

    __table_args__ = (
        UniqueConstraint(
            "analysis_id",
            "sample_id",
            "nucleus_id",
            name="unique_nucleus"
        ),
        Index("idx_sample_id", "sample_id"),
    )

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id"),
        nullable=False
    )

    sample_id = Column(String(100), nullable=False)
    nucleus_id = Column(Integer, nullable=False)

    # Shape features
    area = Column(Float)
    perimeter = Column(Float)
    circularity = Column(Float)
    eccentricity = Column(Float)
    aspect_ratio = Column(Float)
    solidity = Column(Float)
    extent = Column(Float)
    major_axis_length = Column(Float)
    minor_axis_length = Column(Float)

    # Spatial features
    centroid_x = Column(Float)
    centroid_y = Column(Float)

    # Intensity features
    mean_intensity = Column(Float)
    std_intensity = Column(Float)
    min_intensity = Column(Float)
    max_intensity = Column(Float)

    # Texture features
    glcm_contrast = Column(Float)
    glcm_correlation = Column(Float)
    glcm_energy = Column(Float)
    glcm_homogeneity = Column(Float)
    glcm_dissimilarity = Column(Float)

    # Dimensionality reduction
    pc1 = Column(Float)
    pc2 = Column(Float)
    umap1 = Column(Float)
    umap2 = Column(Float)

    # Clustering
    kmeans_cluster = Column(Integer)
    dbscan_cluster = Column(Integer)

    # Anomaly detection
    anomaly_status = Column(String(50))
    anomaly_score = Column(Float)

    analysis = relationship("Analysis", back_populates="nuclei")


class AnalysisMetric(Base):
    __tablename__ = "analysis_metrics"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id"),
        nullable=False
    )

    metric_name = Column(String(100))
    metric_value = Column(Float)
    model_name = Column(String(100))

    analysis = relationship("Analysis", back_populates="metrics")