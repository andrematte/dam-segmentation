import time
from datetime import datetime

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from dam_segmentation.utils import logger_setup

logger = logger_setup(to_file=False)


class RandomForestModel:
    """
    Handles training and inference of the Random Forest models for ortophoto segmentation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str = "sqrt",
        seed: int = 23,
    ):
        self.id = datetime.now().strftime(f"%Y%m%d-%H%M%S-{seed}")

        self.seed = seed
        self.n_estimators = n_estimators
        self.max_features = max_features

        logger.info("-> Initializing Random Forest model")
        logger.info(
            f"---> nEstimators: {n_estimators}, maxFeatures: {max_features}, seed: {seed}"
        )

        self.model = self.init_model()

    def init_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.seed,
        )

    def load_train_data(self, dataset: pd.DataFrame):
        logger.info("---> Loading training data")
        self.y_train = dataset["label"]
        self.X_train = dataset.drop("label", axis=1)
        self.features = self.X_train.columns

    def load_test_data(self, dataset: pd.DataFrame, labels: list[int]):
        logger.info("---> Loading test data:")
        self.labels = labels
        self.y_test = dataset["label"]
        self.X_test = dataset.drop("label", axis=1)

    def fit_model(self):
        logger.info("---> Fitting model")

        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        end = time.time()
        self.train_time = end - start

        logger.info(f"---> Training time: {self.train_time:.2f} s")

    def test_model(self):
        logger.info("---> Testing model")

        self.pred_test_set = self.model.predict(self.X_test)
        accuracy = metrics.accuracy_score(self.y_test, self.pred_test_set)
        precision = metrics.precision_score(
            self.y_test, self.pred_test_set, average="macro"
        )
        recall = metrics.recall_score(
            self.y_test, self.pred_test_set, average="macro"
        )
        f1 = metrics.f1_score(self.y_test, self.pred_test_set, average="macro")
        kappa = metrics.cohen_kappa_score(self.y_test, self.pred_test_set)
        report = metrics.classification_report(self.y_test, self.pred_test_set)
        conf_mat = metrics.confusion_matrix(
            self.y_test, self.pred_test_set, labels=self.labels
        )
        norm_conf_mat = metrics.confusion_matrix(
            self.y_test,
            self.pred_test_set,
            normalize="true",
            labels=self.labels,
        )

        self.metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Kappa": kappa,
            "Report": report,
            "Confusion Matrix": conf_mat,
            "Normalized Confusion Matrix": norm_conf_mat,
        }

        logger.info(f"---> Accuracy: {accuracy:.2f}, F1-Score: {f1:.2f}")

    def compute_feature_importances(self, type="mdi"):
        if type == "mdi":
            self.mdi_importance = pd.Series(
                self.model.feature_importances_, index=self.features
            ).sort_values(ascending=False)
            return self.mdi_importance

        elif type == "permutation":
            result = permutation_importance(
                self.model,
                self.X_test,
                self.y_test,
                n_repeats=10,
                random_state=self.seed,
                n_jobs=2,
            )
            self.perm_importance = pd.Series(
                result.importances_mean, index=self.features
            ).sort_values(ascending=False)
            return self.perm_importance
