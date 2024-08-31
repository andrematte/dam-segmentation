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

# train_path = "../../../data/train/binary_balanced_training.parquet"
train_path = "../../../data/train/binary_reduced_training.parquet"
# test_path = "../../../data/test/binary_test.parquet"
test_path = "../../../data/test/binary_reduced_test.parquet"

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

NTREES = [8, 16, 32, 64, 128, 256, 512, 1024]
LABELS = [0, 1]
SUBSETS = {
    "SUBSET_1": RGB,
    "SUBSET_2": RGB + MSPEC,
    "SUBSET_3": RGB + MSPEC + VINDEX,
    "SUBSET_4": RGB + FILTERS + GLCM,
    "SUBSET_5": RGB + MSPEC + FILTERS + GLCM,
    "SUBSET_6": RGB + MSPEC + VINDEX + FILTERS + GLCM,
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
        results = pd.concat(
            [results, pd.DataFrame(result, index=[0]).round(4)],
            ignore_index=True,
        )

results.to_csv(
    "subsets_results_binary.csv",
    index=False,
)
