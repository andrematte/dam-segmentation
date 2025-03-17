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

from dam_segmentation.settings import SUBSETS
from dam_segmentation.train_rf import RandomForestModel
from dam_segmentation.utils import logger_setup

logger = logger_setup(to_file=False)

train_path = "../../../data/dam-segmentation/train_data_multiclass_downsampled.parquet"
test_path = "../../../data/dam-segmentation/val_data_multiclass.parquet"

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)

# -------------------------- Categorias de Features -------------------------- #


result_columns = [
    "name",
    "nTrees",
    "trainTime",
    "oobError",
    "accuracy",
    "balancedAccuracy",
    "precision",
    "recall",
    "f1score",
    "kappa",
    "AUC",
]
results = pd.DataFrame(columns=result_columns)
results.to_csv("subsets_results_multiclass.csv", index=False)

NTREES = [2, 4, 8, 16, 32, 64, 128, 256]
LABELS = [0, 1, 2, 3]


for SUBSET in SUBSETS.keys():
    logger.info(f"-> Treinando modelos para o {SUBSET}")
    for ntrees in NTREES:
        model = RandomForestModel(
            n_estimators=ntrees, max_features="sqrt", seed=12, n_jobs=-1
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
            "trainTime": model.train_time,
            "oobError": model.metrics["OOB Error"],
            "accuracy": model.metrics["Accuracy"],
            "balancedAccuracy": model.metrics["Balanced Accuracy"],
            "precision": model.metrics["Precision"],
            "recall": model.metrics["Recall"],
            "f1score": model.metrics["F1 Score"],
            "kappa": model.metrics["Kappa"],
            "auc": model.metrics["AUC"],
        }

        results = pd.DataFrame(result, index=[0]).round(4)
        results.to_csv(
            "subsets_results_multiclass.csv",
            mode="a",
            index=False,
            header=False,
        )
