# ---------------------------------------------------------------------------- #
#     Experimento 3: Treinamento do modelo com o melhor Subset de Features     #
# ---------------------------------------------------------------------------- #
#
# Este experimento avalia o desempenho do modelo treinado com o melhor Subset
# de Features, conforme determinado nos Experimentos 1 e 2.
#
# ---------------------------------------------------------------------------- #
import joblib
import pandas as pd
from sklearn.metrics import auc, roc_curve

from dam_segmentation.train_rf import RandomForestModel
from dam_segmentation.utils import logger_setup

logger = logger_setup(to_file=False)

train_path = "../../../data/dam-segmentation/train_data_binary_downsampled.parquet"
test_path = "../../../data/dam-segmentation/test_data_binary.parquet"

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)


# -------------------------- Categorias de Features -------------------------- #

LABELS = [0, 1]
LABEL_NAMES = ["Not Slope", "Slope"]
NFEATURES = 17
NTREES = 256
SUBSET = "SUBSET_4"

features = pd.read_csv("../2-feature-ranking/feature_importances_binary.csv")[
    "Feature"
].values[:NFEATURES]

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

# ---------------------------- Calcular ROC Curve ---------------------------- #

fpr, tpr, thresholds = roc_curve(model.y_test, model.pred_test_set_prob)
roc_auc = auc(fpr, tpr)

roc_df = pd.DataFrame(
    {
        "FPR": fpr,
        "TPR": tpr,
        "Thresholds": thresholds,
        "AUC": [roc_auc] * len(fpr),
    }
)

# ----------------------------- Salvar Resultados ---------------------------- #

roc_df.to_csv("roc_binary.csv", index=False)

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

confmat.to_csv("confusion_matrix_binary.csv")
confmat_norm.to_csv("normalized_confusion_matrix_binary.csv")
report.to_csv("classification_report_binary.csv")
joblib.dump(model.model, "rf_final_binary.joblib")
