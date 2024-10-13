import time
from datetime import datetime

import joblib
import numpy as np
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
        max_depth: int = None,
        max_samples: float = 1.0,
        label_names: list[str] = None,
        n_jobs: int = -1,
        seed: int = 23,
        model_path: str = None,
    ):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.id = datetime.now().strftime(f"%Y%m%d-%H%M%S-{seed}")

            self.seed = seed
            self.n_jobs = n_jobs
            self.n_estimators = n_estimators
            self.max_features = max_features
            self.max_depth = max_depth
            self.max_samples = max_samples
            self.label_names = label_names

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
            n_jobs=self.n_jobs,
            max_depth=self.max_depth,
            max_samples=self.max_samples,
            oob_score=True,
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
        self.pred_test_set_prob = self.model.predict_proba(self.X_test)

        if len(np.unique(self.y_test)) > 2:
            logger.info("---> Multiclass classification")
            average = "macro"
            multi_class = "ovo"
        else:
            logger.info("---> Binary classification")
            average = "binary"
            multi_class = "raise"
            self.pred_test_set_prob = self.pred_test_set_prob[:, 1]

        oob_error = 1 - self.model.oob_score_

        accuracy = metrics.accuracy_score(self.y_test, self.pred_test_set)
        bal_accuracy = metrics.balanced_accuracy_score(
            self.y_test, self.pred_test_set
        )
        precision = metrics.precision_score(
            self.y_test, self.pred_test_set, average=average
        )
        recall = metrics.recall_score(
            self.y_test, self.pred_test_set, average=average
        )
        f1 = metrics.f1_score(self.y_test, self.pred_test_set, average=average)
        kappa = metrics.cohen_kappa_score(self.y_test, self.pred_test_set)

        macro_auc = metrics.roc_auc_score(
            self.y_test,
            self.pred_test_set_prob,
            multi_class=multi_class,
            average="macro",
        )

        report = metrics.classification_report(
            self.y_test,
            self.pred_test_set,
            output_dict=True,
            target_names=self.label_names,
        )
        conf_mat = metrics.confusion_matrix(
            self.y_test,
            self.pred_test_set,
            labels=self.labels,
        )
        norm_conf_mat = metrics.confusion_matrix(
            self.y_test,
            self.pred_test_set,
            normalize="true",
            labels=self.labels,
        )

        self.metrics = {
            "OOB Error": oob_error,
            "Accuracy": accuracy,
            "Balanced Accuracy": bal_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Kappa": kappa,
            "Report": report,
            "Confusion Matrix": conf_mat,
            "Normalized Confusion Matrix": norm_conf_mat,
            "AUC": macro_auc,
        }

        logger.info(
            f"---> Balanced Accuracy: {bal_accuracy:.3f}, F1-Score: {f1:.3f}, AUC: {macro_auc:.3f}"
        )

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
