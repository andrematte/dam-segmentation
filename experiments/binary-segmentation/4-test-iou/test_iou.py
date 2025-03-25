# ---------------------------------------------------------------------------- #
#              Experimento 4: Cálculo do IoU no Conjunto de Teste              #
# ---------------------------------------------------------------------------- #

from glob import glob
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import jaccard_score

from dam_segmentation.feature_extraction import Features

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


# ---------------------------- Iterar sobre Images --------------------------- #
results = []
for image_path, label_path in zip(test_images, test_labels):
    features = Features(image_path, label_path)

    label = features.features["label"]
    features = features.features[selected_features]
    pred = model.predict(features)

    macro_iou = jaccard_score(label, pred, average="macro")
    micro_iou = jaccard_score(label, pred, average="micro")
    w_iou = jaccard_score(label, pred, average="weighted")

    results.append(
        {
            "image": Path(image_path).stem,
            "label": Path(label_path).stem,
            "macro_iou": macro_iou.round(3),
            "micro_iou": micro_iou.round(3),
            "w_iou": w_iou.round(3),
        }
    )

results = pd.DataFrame(results)
results.to_csv("iou_test.csv", index=False)
