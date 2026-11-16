import cv2
import numpy as np

from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops


def extract_features(image, instances):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    features = []

    regions = regionprops(
        instances,
        intensity_image=gray
    )

    for region in regions:

        nucleus_mask = (
            instances == region.label
        ).astype(np.uint8)

        area = float(region.area)

        perimeter = float(region.perimeter)

        circularity = (
            4 * np.pi * area / (perimeter ** 2)
            if perimeter > 0 else 0
        )

        min_row, min_col, max_row, max_col = region.bbox

        nucleus_crop = gray[
            min_row:max_row,
            min_col:max_col
        ]

        crop_mask = nucleus_mask[
            min_row:max_row,
            min_col:max_col
        ]

        pixels = gray[nucleus_mask > 0]

        # Texture extraction
        texture_image = nucleus_crop.copy()

        texture_image[crop_mask == 0] = 0

        quantized = (
            texture_image // 16
        ).astype(np.uint8)

        glcm = graycomatrix(
            quantized,
            distances=[1],
            angles=[0],
            levels=16,
            symmetric=True,
            normed=True
        )

        contrast = float(
            graycoprops(glcm, "contrast")[0, 0]
        )

        correlation = float(
            graycoprops(glcm, "correlation")[0, 0]
        )

        energy = float(
            graycoprops(glcm, "energy")[0, 0]
        )

        homogeneity = float(
            graycoprops(glcm, "homogeneity")[0, 0]
        )

        dissimilarity = float(
            graycoprops(glcm, "dissimilarity")[0, 0]
        )

        feature = {

            "nucleus_id": int(region.label),

            # Shape
            "area": area,
            "perimeter": perimeter,
            "circularity": float(circularity),
            "eccentricity": float(region.eccentricity),
            "aspect_ratio": float(
                region.axis_major_length /
                max(region.axis_minor_length, 1e-6)
            ),
            "solidity": float(region.solidity),
            "extent": float(region.extent),

            "major_axis_length": float(
                region.axis_major_length
            ),

            "minor_axis_length": float(
                region.axis_minor_length
            ),

            # Spatial
            "centroid_x": float(region.centroid[1]),
            "centroid_y": float(region.centroid[0]),

            # Intensity
            "mean_intensity": float(np.mean(pixels)),
            "std_intensity": float(np.std(pixels)),
            "min_intensity": float(np.min(pixels)),
            "max_intensity": float(np.max(pixels)),

            # Texture
            "glcm_contrast": contrast,
            "glcm_correlation": correlation,
            "glcm_energy": energy,
            "glcm_homogeneity": homogeneity,
            "glcm_dissimilarity": dissimilarity
        }

        features.append(feature)

    return features