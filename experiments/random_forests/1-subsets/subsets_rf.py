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

import numpy as np
import pandas as pd

from dam_segmentation.train_rf import RandomForestModel

# train_path = "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/train/binary_balanced_training.parquet"
train_path = "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/train/binary_reduced_training.parquet"
# test_path = "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/test/binary_test.parquet"
test_path = "/Users/andrematte/Developer/Projects/phd/dam_segmentation/data/test/binary_reduced_test.parquet"

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

NTREES = [8, 16, 32, 64, 128, 256, 512, 1024]
LABELS = [0, 1]

result_columns = [
    "name",
    "nTrees",
    "maxFeatures",
    "trainTime",
    "accuracy",
    "precision",
    "recall",
    "f1score",
    "kappa",
]
results = pd.DataFrame(columns=result_columns)

# ------------------------------ Cenário 1: RGB ------------------------------ #
NAME = "CENARIO_1"
S1 = RGB
for ntrees in NTREES:
    model = RandomForestModel(
        n_estimators=ntrees, max_features="sqrt", seed=23
    )
    model.load_train_data(train_data[S1 + ["label"]])
    model.load_test_data(test_data[S1 + ["label"]], labels=LABELS)
    model.fit_model()
    model.test_model()
    model.compute_feature_importances()
    result = {
        "name": NAME,
        "nTrees": ntrees,
        "maxFeatures": "sqrt",
        "trainTime": model.train_time,
        "accuracy": model.metrics["Accuracy"],
        "precision": model.metrics["Precision"],
        "recall": model.metrics["Recall"],
        "f1score": model.metrics["F1 Score"],
        "kappa": model.metrics["Kappa"],
    }
    results = pd.concat(
        [results, pd.DataFrame(result, index=[0]).round(3)], ignore_index=True
    )

# ----------------------------- Cenário 2: MSPEC ----------------------------- #
NAME = "CENARIO_2"
S2 = RGB + MSPEC

for ntrees in NTREES:
    model = RandomForestModel(
        n_estimators=ntrees, max_features="sqrt", seed=23
    )
    model.load_train_data(train_data[S2 + ["label"]])
    model.load_test_data(test_data[S2 + ["label"]], labels=LABELS)
    model.fit_model()
    model.test_model()
    model.compute_feature_importances()
    result = {
        "name": NAME,
        "nTrees": ntrees,
        "maxFeatures": "sqrt",
        "trainTime": model.train_time,
        "accuracy": model.metrics["Accuracy"],
        "precision": model.metrics["Precision"],
        "recall": model.metrics["Recall"],
        "f1score": model.metrics["F1 Score"],
        "kappa": model.metrics["Kappa"],
    }
    results = pd.concat(
        [results, pd.DataFrame(result, index=[0]).round(3)], ignore_index=True
    )

# --------------------- Cenário 3: RGB + MSPEC + V-INDEX --------------------- #
NAME = "CENARIO_3"
S3 = RGB + MSPEC + VINDEX

for ntrees in NTREES:
    model = RandomForestModel(
        n_estimators=ntrees, max_features="sqrt", seed=23
    )
    model.load_train_data(train_data[S3 + ["label"]])
    model.load_test_data(test_data[S3 + ["label"]], labels=LABELS)
    model.fit_model()
    model.test_model()
    model.compute_feature_importances()
    result = {
        "name": NAME,
        "nTrees": ntrees,
        "maxFeatures": "sqrt",
        "trainTime": model.train_time,
        "accuracy": model.metrics["Accuracy"],
        "precision": model.metrics["Precision"],
        "recall": model.metrics["Recall"],
        "f1score": model.metrics["F1 Score"],
        "kappa": model.metrics["Kappa"],
    }
    results = pd.concat(
        [results, pd.DataFrame(result, index=[0]).round(3)], ignore_index=True
    )

# ----------------------- Cenário 4: RGB + FILT + GLCM ----------------------- #
NAME = "CENARIO_4"
S4 = RGB + FILTERS + GLCM

for ntrees in NTREES:
    model = RandomForestModel(
        n_estimators=ntrees, max_features="sqrt", seed=23
    )
    model.load_train_data(train_data[S4 + ["label"]])
    model.load_test_data(test_data[S4 + ["label"]], labels=LABELS)
    model.fit_model()
    model.test_model()
    model.compute_feature_importances()
    result = {
        "name": NAME,
        "nTrees": ntrees,
        "maxFeatures": "sqrt",
        "trainTime": model.train_time,
        "accuracy": model.metrics["Accuracy"],
        "precision": model.metrics["Precision"],
        "recall": model.metrics["Recall"],
        "f1score": model.metrics["F1 Score"],
        "kappa": model.metrics["Kappa"],
    }
    results = pd.concat(
        [results, pd.DataFrame(result, index=[0]).round(3)], ignore_index=True
    )

# ------------------- Cenário 5: RGB + MSPEC + FILT + GLCM ------------------- #
NAME = "CENARIO_5"
S5 = RGB + MSPEC + FILTERS + GLCM

for ntrees in NTREES:
    model = RandomForestModel(
        n_estimators=ntrees, max_features="sqrt", seed=23
    )
    model.load_train_data(train_data[S5 + ["label"]])
    model.load_test_data(test_data[S5 + ["label"]], labels=LABELS)
    model.fit_model()
    model.test_model()
    model.compute_feature_importances()
    result = {
        "name": NAME,
        "nTrees": ntrees,
        "maxFeatures": "sqrt",
        "trainTime": model.train_time,
        "accuracy": model.metrics["Accuracy"],
        "precision": model.metrics["Precision"],
        "recall": model.metrics["Recall"],
        "f1score": model.metrics["F1 Score"],
        "kappa": model.metrics["Kappa"],
    }
    results = pd.concat(
        [results, pd.DataFrame(result, index=[0]).round(3)], ignore_index=True
    )

# -------------- Cenário 6: RGB + MSPEC + V-INDEX + FILT + GLCM -------------- #
NAME = "CENARIO_6"
S6 = RGB + MSPEC + VINDEX + FILTERS + GLCM

for ntrees in NTREES:
    model = RandomForestModel(
        n_estimators=ntrees, max_features="sqrt", seed=23
    )
    model.load_train_data(train_data[S6 + ["label"]])
    model.load_test_data(test_data[S6 + ["label"]], labels=LABELS)
    model.fit_model()
    model.test_model()
    model.compute_feature_importances()
    result = {
        "name": NAME,
        "nTrees": ntrees,
        "maxFeatures": "sqrt",
        "trainTime": model.train_time,
        "accuracy": model.metrics["Accuracy"],
        "precision": model.metrics["Precision"],
        "recall": model.metrics["Recall"],
        "f1score": model.metrics["F1 Score"],
        "kappa": model.metrics["Kappa"],
    }
    results = pd.concat(
        [results, pd.DataFrame(result, index=[0]).round(3)], ignore_index=True
    )

results.to_csv(
    "/Users/andrematte/Developer/Projects/phd/dam_segmentation/experiments/random_forests/1-subsets/results.csv",
    index=False,
)
