# ---------------------------------------------------------------------------- #
#      Experimento 2: Feature Ranking para determinação do Top X utilizado     #
# ---------------------------------------------------------------------------- #
#
# Este experimento avalia o impacto de diferentes conjuntos de X features do
# Subset selecionado.
#
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
LABELS = [0, 1, 2, 3]
SUBSETS = {
    "SUBSET_1": RGB,
    "SUBSET_2": RGB + MSPEC + VINDEX,
    "SUBSET_3": RGB + FILTERS,
    "SUBSET_4": RGB + MSPEC + VINDEX + FILTERS,
}

BEST_SUBSET = SUBSETS["SUBSET_4"]
BEST_NTREES = 128

IMPORTANCE_THRESHOLD = 0.01

result_columns = [
    "name",
    "nTrees",
    "trainTime",
    "accuracy",
    "precision",
    "recall",
    "f1score",
    "kappa",
]
results = pd.DataFrame(columns=result_columns)
results.to_csv("ranking_results_multiclass.csv", index=False)

# ---------- Etapa 1: Treinar modelo em todas as features do subset ---------- #
logger.info("-> Training model with the best subset of features")
model = RandomForestModel(
    n_estimators=BEST_NTREES, max_features="sqrt", seed=23
)
model.load_train_data(train_data[BEST_SUBSET + ["label"]])
model.load_test_data(test_data[BEST_SUBSET + ["label"]], labels=LABELS)
model.fit_model()
model.test_model()
importances = model.compute_feature_importances()
logger.info(f"-> Filtering features with importance > {IMPORTANCE_THRESHOLD}")
importances = importances[importances > IMPORTANCE_THRESHOLD]

importance_df = pd.DataFrame(importances, columns=["Importance"])
importance_df.index.name = "Feature"
importance_df.to_csv("feature_importances_multiclass.csv")

ranked_features = importances.index

# ----------- Etapa 2: Treinar modelos com features Top 1 até Top X ---------- #
for X in range(11, len(ranked_features)):
    logger.info(f"---> Training model with the Top {X + 1} features")

    model = RandomForestModel(
        n_estimators=BEST_NTREES, max_features="sqrt", seed=23
    )
    model.load_train_data(
        train_data[ranked_features[: X + 1].to_list() + ["label"]]
    )
    model.load_test_data(
        test_data[ranked_features[: X + 1].to_list() + ["label"]],
        labels=LABELS,
    )
    model.fit_model()
    model.test_model()

    result = {
        "name": f"Top {X + 1}",
        "nTrees": BEST_NTREES,
        "trainTime": model.train_time,
        "accuracy": model.metrics["Accuracy"],
        "precision": model.metrics["Precision"],
        "recall": model.metrics["Recall"],
        "f1score": model.metrics["F1 Score"],
        "kappa": model.metrics["Kappa"],
    }

    results = pd.DataFrame(result, index=[0]).round(4)
    results.to_csv(
        "ranking_results_multiclass.csv",
        mode="a",
        index=False,
        header=False,
    )
