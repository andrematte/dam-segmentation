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

from dam_segmentation.feature_extraction import Features
from dam_segmentation.utils import create_directory, mask_to_rgb

# ------------------------- Carregar Imagens e Modelo ------------------------ #

model_path = "../3-best-model/rf_final_binary.joblib"
model = joblib.load(model_path)

test_images = glob("../../../data/dam-segmentation/test/images/*.tif*")
test_labels = glob("../../../data/dam-segmentation/test/mask_binary/*.tif*")

selected_features = [
    "ndvi",
    "gndvi",
    "ndwi",
    "rededge",
    "ndre",
    "blue",
    "nir",
    "green",
    "red",
    "gray",
    "gaussian 15",
    "median 7",
    "gaussian 7",
    "gaussian 12",
    "gaussian 3",
    "gabor_2",
    "prewitt",
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
labels = [0, 1]
n_classes = len(labels)
total_inter = np.zeros(n_classes)
total_union = np.zeros(n_classes)
total_inter_smooth = np.zeros(n_classes)
total_union_smooth = np.zeros(n_classes)

for image_path, label_path in zip(test_images, test_labels):
    features = Features(image_path, label_path)

    label = features.features["label"].to_numpy()
    features = features.features[selected_features]
    pred = model.predict(features)

    pred_image = pred.reshape(256, 256)
    pred_image_smooth = pred_image.copy()
    pred_image_smooth = majority_filter(pred_image_smooth, kernel_size=5)
    pred_smooth = pred_image_smooth.flatten()

    for cls in range(n_classes):
        pred_mask = pred == cls
        true_mask = label == cls

        inter = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()

        total_inter[cls] += inter
        total_union[cls] += union

    for cls in range(n_classes):
        pred_mask = pred_smooth == cls
        true_mask = label == cls

        inter = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()

        total_inter_smooth[cls] += inter
        total_union_smooth[cls] += union

    create_directory("test-results")

    image = Image(image_path)
    rgb = image.get_rgb()
    mask = cv2.imread(label_path, cv2.IMREAD_COLOR_RGB)
    pred = mask_to_rgb(pred_image.reshape(256, 256, 1))
    pred_diff = mask - pred
    pred_smooth = mask_to_rgb(pred_image_smooth.reshape(256, 256, 1))
    pred_smooth_diff = mask - pred_smooth

    pred_diff[pred_diff != 0] = 255
    pred_smooth_diff[pred_smooth_diff != 0] = 255

    plt.imsave(
        f"test-results/{Path(image_path).stem}_rgb.png",
        rgb,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_mask.png",
        mask,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_pred.png",
        pred,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_pred_diff.png",
        pred_diff,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_pred_smooth.png",
        pred_smooth,
    )
    plt.imsave(
        f"test-results/{Path(image_path).stem}_pred_smooth_diff.png",
        pred_smooth_diff,
    )

iou_per_class = total_inter / (total_union + 1e-7)
mean_iou = np.nanmean(iou_per_class)

iou_per_class_smooth = total_inter_smooth / (total_union_smooth + 1e-7)
mean_iou_smooth = np.nanmean(iou_per_class_smooth)

print(iou_per_class)
print(iou_per_class_smooth)
print()
print(mean_iou)
print(mean_iou_smooth)

# Create a dataframe to save the results, including the mean IoU
df = pd.DataFrame(
    {
        "Class": labels,
        "IoU": iou_per_class.round(3),
        "IoU Smooth": iou_per_class_smooth.round(3),
    }
)
df["Mean IoU"] = mean_iou.round(3)
df["Mean IoU Smooth"] = mean_iou_smooth.round(3)
df.to_csv("iou_results.csv", index="Class")
