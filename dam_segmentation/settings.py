SELECTED_FEATURES = [
    "Gaussian 15",
    "Gaussian 12",
    "Gaussian 3",
    "Gaussian 7",
    "Gray",
    "RedEdge",
    "NIR",
    "Median 7",
    "Blue",
    "Gabor 36",
    "Red",
    "Gabor 37",
    "Gabor 13",
    "Green",
    "Gabor 18",
    "Gabor 42",
    "Prewitt",
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
