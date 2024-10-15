from glob import glob

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from sklearn.metrics import jaccard_score

from dam_segmentation.feature_extraction import Features
from dam_segmentation.settings import SELECTED_FEATURES_MULTICLASS
from dam_segmentation.utils import mask_to_rgb, rgb_to_mask

selected = [
    "/Users/andrematte/Data/nesa-dataset/test/images/S2_01_SLICED_0054.tiff",
    "/Users/andrematte/Data/nesa-dataset/test/images/S1_01_SLICED_0228.tiff",
    "/Users/andrematte/Data/nesa-dataset/test/images/S1_01_SLICED_0218.tiff",
    "/Users/andrematte/Data/nesa-dataset/test/images/S1_01_SLICED_0172.tiff",
    "/Users/andrematte/Data/nesa-dataset/test/images/S1_01_SLICED_0157.tiff",
    "/Users/andrematte/Data/nesa-dataset/test/images/S1_01_SLICED_0190.tiff",
]

image_paths = glob("/Users/andrematte/Data/nesa-dataset/test/images/*.tiff")
multi_model_path = "/Users/andrematte/Developer/Projects/phd/dam-segmentation/experiments/multiclass-segmentation/3-best-model/rf_final_multiclass.joblib"


LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["Background", "Slope", "Stairways", "Drainage"]


multi_model = joblib.load(multi_model_path)

results = []

for i, path in enumerate(image_paths):
    multi_label = tifffile.imread(path.replace("images", "mask_multiclass"))

    multi_label = rgb_to_mask(multi_label)

    multi_features = Features(path).features[SELECTED_FEATURES_MULTICLASS]

    multi_pred = multi_model.predict(multi_features)

    multi_pred_mask = multi_pred.reshape(256, 256)
    smooth_multi_pred_mask = cv2.medianBlur(multi_pred_mask, ksize=11)

    multi_iou = jaccard_score(
        multi_label.flatten(), multi_pred_mask.flatten(), average="macro"
    )
    multi_iou_smooth = jaccard_score(
        multi_label.flatten(),
        smooth_multi_pred_mask.flatten(),
        average="macro",
    )

    multi_pred_mask = mask_to_rgb(multi_pred_mask.reshape(256, 256, 1))
    multi_smooth_mask = mask_to_rgb(
        smooth_multi_pred_mask.reshape(256, 256, 1)
    )

    result = {
        "image_path": path,
        "multi_iou": multi_iou,
        "multi_iou_smooth": multi_iou_smooth,
        "multi_smooth_diff": multi_iou_smooth - multi_iou,
    }

    results.append(result)

    if path in selected:
        cv2.imwrite(
            f"inferences/{i}_rgb_pred.png",
            cv2.cvtColor(multi_pred_mask, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            f"inferences/{i}_rgb_smooth.png",
            cv2.cvtColor(multi_smooth_mask, cv2.COLOR_RGB2BGR),
        )

        diff = cv2.absdiff(multi_pred_mask, multi_smooth_mask)
        cv2.imwrite(
            f"inferences/{i}_diff.png",
            cv2.cvtColor(diff, cv2.COLOR_RGB2BGR),
        )


df = pd.DataFrame(results)
df.to_csv("test_results_iou.csv", index=False)
