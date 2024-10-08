# ---------------------------------------------------------------------------- #
#   Experimento 1: Avaliar subsets de features e parâmetros de Random Forest   #
# ---------------------------------------------------------------------------- #
#
# Este experimento avalia o impacto de diferentes subsets de features e parâmetros
# de Random Forest na segmentação de land-cover de imagens de barragens de aterro.
#
# Parâmetros:
# 1) nTrees: número de árvores de decisão;
#
# Categorias:
# 1) RGB; 2) MSPEC; 3) RGB + MSPEC + V-INDEX; 5) RGB + FILT + GLCM;
# 6) RGB + MSPEC + FILT + GLCM; 7) RGB + MSPEC + V-INDEX + FILT + GLCM.
# ---------------------------------------------------------------------------- #

import pandas as pd

from dam_segmentation.train_rf import RandomForestModel
from dam_segmentation.utils import logger_setup

logger = logger_setup(to_file=False)

train_path = "../../data/train/train_data_multiclass_downsampled.parquet"
test_path = "../../data/test/test_data_multiclass.parquet"

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)

# -------------------------- Categorias de Features -------------------------- #
RGB = ["gray", "red", "green", "blue"]
MSPEC = ["rededge", "nir"]
VINDEX = ["ndvi", "gndvi", "ndre", "ndwi"]
# GLCM = [
#     "contrast",
#     "dissimilarity",
#     "homogeneity",
#     "entropy",
#     "correlation",
#     "asm",
# ]
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

result_columns = [
    "name",
    "nTrees",
    "maxSamples",
    "trainTime",
    "accuracy",
    "precision",
    "recall",
    "f1score",
    "kappa",
]
results = pd.DataFrame(columns=result_columns)
results.to_csv("subsets_results_multiclass.csv", index=False)

NTREES = [64, 128]  # [2, 4, 8, 16, 32, 64, 128]
LABELS = [0, 1, 2, 3]
SUBSETS = {
    # "SUBSET_1": RGB,
    # "SUBSET_2": RGB + MSPEC + VINDEX,
    # "SUBSET_3": RGB + FILTERS,
    "SUBSET_4": RGB + MSPEC + VINDEX + FILTERS,
}


for SUBSET in SUBSETS.keys():
    logger.info(f"-> Treinando modelos para o {SUBSET}")
    for ntrees in NTREES:
        model = RandomForestModel(
            n_estimators=ntrees, max_features="sqrt", seed=23, n_jobs=7
        )
        model.load_train_data(train_data[SUBSETS[SUBSET] + ["label"]])
        model.load_test_data(
            test_data[SUBSETS[SUBSET] + ["label"]], labels=LABELS
        )
        model.fit_model()
        model.test_model()
        model.compute_feature_importances()
        result = {
            "name": SUBSET,
            "nTrees": ntrees,
            "maxSamples": model.max_samples,
            "trainTime": model.train_time,
            "accuracy": model.metrics["Accuracy"],
            "precision": model.metrics["Precision"],
            "recall": model.metrics["Recall"],
            "f1score": model.metrics["F1 Score"],
            "kappa": model.metrics["Kappa"],
        }

        results = pd.DataFrame(result, index=[0]).round(4)
        results.to_csv(
            "subsets_results_multiclass.csv",
            mode="a",
            index=False,
            header=False,
        )
