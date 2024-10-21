from glob import glob

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from sklearn.metrics import jaccard_score

from dam_segmentation.feature_extraction import Features
from dam_segmentation.settings import SELECTED_FEATURES_BINARY
from dam_segmentation.utils import mask_to_rgb, rgb_to_mask

train = glob("/Users/andrematte/Data/nesa-dataset/train/images/*.tiff")
val = glob("/Users/andrematte/Data/nesa-dataset/val/images/*.tiff")
test = glob("/Users/andrematte/Data/nesa-dataset/test/images/*.tiff")

folders = {"train": train, "val": val, "test": test}

bin_model_path = "/Users/andrematte/Developer/Projects/phd/dam-segmentation/experiments/binary-segmentation/3-best-model/rf_final_binary.joblib"


LABELS = [0, 1]
LABEL_NAMES = ["Non-Slope", "Slope"]


bin_model = joblib.load(bin_model_path)

results = []

for folder in folders.keys():
    for i, path in enumerate(folders[folder]):
        bin_label = tifffile.imread(path.replace("images", "mask_binary"))

        bin_label = rgb_to_mask(bin_label)

        bin_features = Features(path).features[SELECTED_FEATURES_BINARY]

        bin_pred = bin_model.predict(bin_features)

        bin_pred_mask = bin_pred.reshape(256, 256)
        smooth_bin_pred_mask = cv2.medianBlur(bin_pred_mask, ksize=11)

        bin_iou = jaccard_score(
            bin_label.flatten(), bin_pred_mask.flatten(), average="macro"
        )
        bin_iou_smooth = jaccard_score(
            bin_label.flatten(),
            smooth_bin_pred_mask.flatten(),
            average="macro",
        )

        bin_pred_mask = mask_to_rgb(bin_pred_mask.reshape(256, 256, 1))
        bin_smooth_mask = mask_to_rgb(
            smooth_bin_pred_mask.reshape(256, 256, 1)
        )

        result = {
            "image_path": path,
            "set": folder,
            "bin_iou": bin_iou,
            "bin_iou_smooth": bin_iou_smooth,
            "bin_smooth_diff": bin_iou_smooth - bin_iou,
        }

        results.append(result)

        if folder == "test":
            cv2.imwrite(
                f"inferences/{i}_rgb_pred.png",
                cv2.cvtColor(bin_pred_mask, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                f"inferences/{i}_rgb_smooth.png",
                cv2.cvtColor(bin_smooth_mask, cv2.COLOR_RGB2BGR),
            )

            diff = cv2.absdiff(bin_pred_mask, bin_smooth_mask)
            cv2.imwrite(
                f"inferences/{i}_diff.png",
                cv2.cvtColor(diff, cv2.COLOR_RGB2BGR),
            )


df = pd.DataFrame(results)
df.to_csv("test_results_iou.csv", index=False)
