# ---------------------------------------------------------------------------- #
#     Experimento 3: Treinamento do modelo com o melhor Subset de Features     #
# ---------------------------------------------------------------------------- #
#
# Este experimento avalia o desempenho do modelo treinado com o melhor Subset
# de Features, conforme determinado nos Experimentos 1 e 2.
#
# ---------------------------------------------------------------------------- #

import pandas as pd

from dam_segmentation.train_rf import RandomForestModel
from dam_segmentation.utils import logger_setup

logger = logger_setup(to_file=False)

# train_path = "../../../data/train/binary_balanced_training.parquet"
# test_path = "../../../data/test/binary_test.parquet"

train_path = "../../../data/train/binary_reduced_training.parquet"
test_path = "../../../data/test/binary_reduced_test.parquet"

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)


# -------------------------- Categorias de Features -------------------------- #

LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["Background", "Vegetation", "Stairways", "Drainage"]
NFEATURES = 15
NTREES = 128
SUBSET = "CENARIO_6"

features = pd.read_csv("../2-feature-ranking/feature_importances.csv")[
    "Feature"
].values[:NFEATURES]

result_columns = [
    "name",
    "nTrees",
    # "maxSamples",
    # "maxDepth",
    # "maxFeatures",
    "trainTime",
    "accuracy",
    "precision",
    "recall",
    "f1score",
    "kappa",
]
results = pd.DataFrame(columns=result_columns)

# ------------- Etapa 1: Treinar modelo nas features selecionadas ------------ #
logger.info("-> Training model with the best subset of features")
model = RandomForestModel(n_estimators=NTREES, label_names=LABEL_NAMES)
model.load_train_data(train_data[list(features) + ["label"]])
model.load_test_data(test_data[list(features) + ["label"]], labels=LABELS)
model.fit_model()
model.test_model()

result = {
    "name": f"{SUBSET} - Top {NFEATURES}",
    "nTrees": NTREES,
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
    "best_results_binary.csv",
    index=False,
)

confmat = pd.DataFrame(
    model.metrics["Confusion Matrix"], index=LABEL_NAMES, columns=LABEL_NAMES
)
confmat_norm = pd.DataFrame(
    model.metrics["Normalized Confusion Matrix"],
    index=LABEL_NAMES,
    columns=LABEL_NAMES,
)
report = pd.DataFrame(model.metrics["Report"]).transpose()

confmat.to_csv("confusion_matrix_multiclass.csv")
confmat_norm.to_csv("normalized_confusion_matrix_multiclass.csv")
report.to_csv("classification_report_multiclass.csv")
