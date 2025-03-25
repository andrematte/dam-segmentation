# ---------------------------------------------------------------------------- #
#              Experimento 4: Cálculo do IoU no Conjunto de Teste              #
# ---------------------------------------------------------------------------- #

from glob import glob
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ms_image_tool.image import Image
from scipy import ndimage
from sklearn.metrics import jaccard_score

from dam_segmentation.feature_extraction import Features
from dam_segmentation.utils import create_directory, mask_to_rgb

# ------------------------- Carregar Imagens e Modelo ------------------------ #

model_path = "../3-best-model/rf_final_multiclass.joblib"
model = joblib.load(model_path)

test_images = glob("../../../data/dam-segmentation/test/images/*.tif*")
test_labels = glob(
    "../../../data/dam-segmentation/test/mask_multiclass/*.tif*"
)

selected_features = [
    "blue",
    "rededge",
    "nir",
    "green",
    "ndwi",
    "gndvi",
    "red",
    "ndvi",
    "ndre",
    "gray",
    "scharr",
    "prewitt",
    "median 7",
    "sobel",
    "gaussian 12",
    "roberts",
    "gabor_1",
    "gabor_0",
    "gabor_27",
    "gaussian 15",
    "gabor_2",
    "gabor_17",
    "gaussian 7",
    "gaussian 3",
]


def majority_filter(mask, kernel_size=3):
    footprint = np.ones((kernel_size, kernel_size))
    return ndimage.generic_filter(
        mask,
        lambda x: np.bincount(x.astype(int)).argmax(),
        footprint=footprint,
    )


# ---------------------------- Iterar sobre Images --------------------------- #
results = []
for image_path, label_path in zip(test_images, test_labels):
    features = Features(image_path, label_path)

    label = features.features["label"]
    features = features.features[selected_features]
    pred = model.predict(features)

    pred_image = pred.reshape(256, 256)
    pred_image_smooth = pred_image.copy()
    pred_image_smooth = majority_filter(pred_image_smooth, kernel_size=5)
    pred_smooth = pred_image_smooth.flatten()

    macro_iou = jaccard_score(label, pred, average="macro")
    micro_iou = jaccard_score(label, pred, average="micro")
    w_iou = jaccard_score(label, pred, average="weighted")

    smoothed_macro_iou = jaccard_score(label, pred_smooth, average="macro")
    smoothed_micro_iou = jaccard_score(label, pred_smooth, average="micro")
    smoothed_w_iou = jaccard_score(label, pred_smooth, average="weighted")

    results.append(
        {
            "image": Path(image_path).stem,
            "label": Path(label_path).stem,
            "macro_iou": macro_iou.round(3),
            "micro_iou": micro_iou.round(3),
            "w_iou": w_iou.round(3),
            "smoothed_macro_iou": smoothed_macro_iou.round(3),
            "smoothed_micro_iou": smoothed_micro_iou.round(3),
            "smoothed_w_iou": smoothed_w_iou.round(3),
        }
    )

    create_directory("test-results")

    image = Image(image_path)
    rgb = image.get_rgb()
    mask = cv2.imread(label_path, cv2.IMREAD_COLOR_RGB)
    pred = mask_to_rgb(pred_image.reshape(256, 256, 1))
    pred_diff = mask - pred
    pred_smooth = mask_to_rgb(pred_image_smooth.reshape(256, 256, 1))
    pred_smooth_diff = mask - pred_smooth

    plt.imsave(
        f"test-results/{Path(image_path).stem}_rgb.png",
        rgb,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_mask.png",
        mask,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_pred_{macro_iou.round(3)}.png",
        pred,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_pred_diff.png",
        pred_diff,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_pred_smooth_{smoothed_macro_iou.round(3)}.png",
        pred_smooth,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_pred_smooth_diff.png",
        pred_smooth_diff,
    )

results = pd.DataFrame(results)
results.to_csv("iou_test.csv", index=False)
