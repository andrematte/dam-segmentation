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

train_path = "../../../data/train_data_multiclass.parquet"
test_path = "../../../data/test_data_multiclass.parquet"

train_data = pd.read_parquet(train_path)
test_data = pd.read_parquet(test_path)


# -------------------------- Categorias de Features -------------------------- #

LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["Background", "Vegetation", "Stairways", "Drainage"]
NFEATURES = 24
NTREES = 128
SUBSET = "SUBSET_4"

features = pd.read_csv(
    "../2-feature-ranking/feature_importances_multiclass.csv"
)["Feature"].values[:NFEATURES]

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
num_classes = model.pred_test_set_prob.shape[1]
roc_data = []

for i in range(num_classes):
    y_test_binary = (model.y_test == i).astype(int)
    fpr, tpr, thresholds = roc_curve(
        y_test_binary, model.pred_test_set_prob[:, i]
    )
    roc_auc = auc(fpr, tpr)
    class_data = pd.DataFrame(
        {
            "Class": [i] * len(fpr),
            "FPR": fpr,
            "TPR": tpr,
            "Thresholds": thresholds,
            "AUC": [roc_auc] * len(fpr),
        }
    )
    roc_data.append(class_data)

roc_df = pd.concat(roc_data, ignore_index=True)


# ----------------------------- Salvar Resultados ---------------------------- #

roc_df.to_csv("roc_multiclass.csv", index=False)
results.to_csv(
    "best_results_multiclass.csv",
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
joblib.dump(model.model, "rf_final_multiclass.joblib")
