import cv2
import numpy as np

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label


def separate_nuclei(binary_mask):

    binary_mask = (binary_mask > 0).astype(np.uint8)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    cleaned = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Distance transform
    distance = ndi.distance_transform_edt(cleaned)

    # Detect local maxima
    coordinates = peak_local_max(
        distance,
        min_distance=5,
        labels=cleaned,
        exclude_border=False
    )

    # Create marker image
    markers = np.zeros(
        distance.shape,
        dtype=np.int32
    )

    for index, coordinate in enumerate(coordinates, start=1):

        markers[
            coordinate[0],
            coordinate[1]
        ] = index

    if len(coordinates) == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    markers = ndi.label(markers > 0)[0]

    # Watershed segmentation
    instances = watershed(
        -distance,
        markers,
        mask=cleaned
    )

    # Remove very small objects
    output = np.zeros_like(instances, dtype=np.int32)

    new_id = 1

    for region_id in np.unique(instances):

        if region_id == 0:
            continue

        region = instances == region_id

        if np.sum(region) >= 10:

            output[region] = new_id
            new_id += 1

    return output