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

train_path = "../../../data/train_data_binary.parquet"
test_path = "../../../data/test_data_binary.parquet"

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)

# -------------------------- Categorias de Features -------------------------- #
RGB = ["gray", "red", "green", "blue"]
MSPEC = ["rededge", "nir"]
VINDEX = ["ndvi", "gndvi", "ndre", "ndwi"]
GLCM = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "entropy",
    "correlation",
    "asm",
]
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
]

result_columns = [
    "name",
    "nTrees",
    "maxSamples",
    "maxDepth",
    "maxFeatures",
    "trainTime",
    "accuracy",
    "precision",
    "recall",
    "f1score",
    "kappa",
]
results = pd.DataFrame(columns=result_columns)
results.to_csv("subsets_results_binary.csv", index=False)

NTREES = [2, 4, 8, 16, 32, 64, 128]
LABELS = [0, 1]
SUBSETS = {
    "SUBSET_1": RGB,
    "SUBSET_2": RGB + MSPEC + VINDEX,
    "SUBSET_3": RGB + FILTERS + GLCM,
    "SUBSET_4": RGB + MSPEC + VINDEX + FILTERS + GLCM,
}


for SUBSET in SUBSETS.keys():
    logger.info(f"-> Treinando modelos para o {SUBSET}")
    for ntrees in NTREES:
        model = RandomForestModel(
            n_estimators=ntrees, max_features="sqrt", seed=23
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
            # "maxDepth": model.max_depth,
            # "maxFeatures": "sqrt",
            "trainTime": model.train_time,
            "accuracy": model.metrics["Accuracy"],
            "precision": model.metrics["Precision"],
            "recall": model.metrics["Recall"],
            "f1score": model.metrics["F1 Score"],
            "kappa": model.metrics["Kappa"],
        }

        results = pd.DataFrame(result, index=[0]).round(4)
        results.to_csv(
            "subsets_results_binary.csv",
            mode="a",
            index=False,
            header=False,
        )
