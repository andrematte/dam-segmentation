# ---------------------------------------------------------------------------- #
#      Experimento 2: Feature Ranking para determinação do Top X utilizado     #
# ---------------------------------------------------------------------------- #
#
# Este experimento avalia o impacto de diferentes conjuntos de X features do
# Subset selecionado.
#
# ---------------------------------------------------------------------------- #

import pandas as pd

from dam_segmentation.settings import SUBSETS
from dam_segmentation.train_rf import RandomForestModel
from dam_segmentation.utils import logger_setup

logger = logger_setup(to_file=False)

train_path = "../../../data/train_data_multiclass_downsampled.parquet"
test_path = "../../../data/val_data_multiclass.parquet"


train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)


# -------------------------- Categorias de Features -------------------------- #

LABELS = [0, 1, 2, 3]
BEST_SUBSET = SUBSETS["SUBSET_4"]
BEST_NTREES = 128
MAX_FEATURES = "sqrt"

IMPORTANCE_THRESHOLD = 0.00001

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
results.to_csv("ranking_results_multiclass.csv", index=False)

# ---------- Etapa 1: Treinar modelo em todas as features do subset ---------- #
logger.info("-> Training model with the best subset of features")
model = RandomForestModel(
    n_estimators=BEST_NTREES, max_features=MAX_FEATURES, seed=23
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
for X in range(1, len(ranked_features)):
    logger.info(f"---> Training model with the Top {X + 1} features")

    model = RandomForestModel(
        n_estimators=BEST_NTREES, max_features=MAX_FEATURES, seed=23
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
        "ranking_results_multiclass.csv",
        mode="a",
        index=False,
        header=False,
    )
