RGB = ["gray", "red", "green", "blue"]
MSPEC = ["rededge", "nir"]
VINDEX = ["ndvi", "gndvi", "ndre", "ndwi"]
FILTERS = [
    "canny",
    "laplacian",
    "roberts",
    "sobel",
    "scharr",
    "prewitt",
    "gaussian 3",
    "gaussian 7",
    "gaussian 12",
    "gaussian 15",
    "median 7",
    "gabor_0",
    "gabor_1",
    "gabor_2",
    "gabor_3",
    "gabor_4",
    "gabor_5",
    "gabor_6",
    "gabor_7",
    "gabor_8",
    "gabor_9",
    "gabor_10",
    "gabor_11",
    "gabor_12",
    "gabor_13",
    "gabor_14",
    "gabor_15",
    "gabor_16",
    "gabor_17",
    "gabor_18",
    "gabor_19",
    "gabor_20",
    "gabor_21",
    "gabor_22",
    "gabor_23",
    "gabor_24",
    "gabor_25",
    "gabor_26",
    "gabor_27",
    "gabor_28",
    "gabor_29",
    "gabor_30",
    "gabor_31",
    "gabor_32",
    "gabor_33",
    "gabor_34",
    "gabor_35",
    "gabor_36",
    "gabor_37",
    "gabor_38",
    "gabor_39",
    "gabor_40",
    "gabor_41",
    "gabor_42",
    "gabor_43",
    "gabor_44",
    "gabor_45",
    "gabor_46",
    "gabor_47",
]

SUBSETS = {
    "SUBSET_1": RGB,
    "SUBSET_2": RGB + MSPEC + VINDEX,
    "SUBSET_3": RGB + FILTERS,
    "SUBSET_4": RGB + MSPEC + VINDEX + FILTERS,
}

MASK_COLORS = {
    "uncategorized": [0, 0, 0],  # #000000
    "slope": [6, 214, 160],  # #06d6a0
    "stairs": [7, 59, 76],  # #073b4c
    "drain": [239, 71, 111],  # #ef476f
}

SEGMENTATION_CLASSES = {
    "uncategorized": 0,
    "slope": 1,
    "stairs": 2,
    "drain": 3,
}
