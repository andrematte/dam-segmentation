SELECTED_FEATURES_BINARY = [
    "gaussian 7",
    "gaussian 12",
    "gaussian 3",
    "median 7",
    "gndvi",
    "ndvi",
    "ndwi",
    "rededge",
    "nir",
    "green",
    "gaussian 15",
]

SELECTED_FEATURES_MULTICLASS = [
    "rededge",
    "ndwi",
    "green",
    "nir",
    "gaussian 3",
    "red",
    "gray",
    "median 7",
    "asm",
    "gndvi",
    "ndre",
    "gaussian 7",
    "gaussian 12",
    "ndvi",
    "gaussian 15",
    "blue",
]

MASK_COLORS = {
    "uncategorized": [0, 0, 0],  # #000000
    "slope": [6, 214, 160],  # #06d6a0
    "stairs": [7, 59, 76],  # #073b4c
    "drain": [239, 71, 111],  # #ef476f
    "rocks": [181, 23, 158],  # #b5179e
}

SEGMENTATION_CLASSES = {
    "uncategorized": 0,
    "slope": 1,
    "stairs": 2,
    "drain": 3,
    "rocks": 4,
}
