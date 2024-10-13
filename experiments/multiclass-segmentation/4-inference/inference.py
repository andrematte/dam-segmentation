import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from sklearn.metrics import jaccard_score

from dam_segmentation.feature_extraction import Features
from dam_segmentation.settings import SELECTED_FEATURES_MULTICLASS
from dam_segmentation.utils import mask_to_rgb, rgb_to_mask

paths = [
    # "/Users/andrematte/Data/nesa-dataset/images/DIQUE6C_01_0265.tiff",
    # "/Users/andrematte/Data/nesa-dataset/images/BVSA_01_0530.tiff",
    # "/Users/andrematte/Data/nesa-dataset/images/BVSA_01_0377.tiff",
    "/Users/andrematte/Data/nesa-dataset/test/images/S1_01_SLICED_0218.tiff",
    "/Users/andrematte/Data/nesa-dataset/test/images/S1_01_SLICED_0157.tiff",
    "/Users/andrematte/Data/nesa-dataset/test/images/S1_01_SLICED_0036.tiff",
]

model_path = "/Users/andrematte/Developer/Projects/phd/dam-segmentation/experiments/multiclass-segmentation/3-best-model/rf_final_multiclass.joblib"

input_image = "path"
input_label = ""

LABELS = [0, 1, 2, 3]
LABEL_NAMES = ["Background", "Slope", "Stairways", "Drainage"]

model = joblib.load(model_path)

for i, path in enumerate(paths):
    # bin_label = tifffile.imread(path.replace("images", "mask_binary"))
    label = tifffile.imread(path.replace("images", "mask_multiclass"))
    label_mask = rgb_to_mask(label)

    features = Features(path).features[SELECTED_FEATURES_MULTICLASS]

    results = model.predict(features)

    result_mask = results.reshape(256, 256)
    smooth_mask = cv2.medianBlur(result_mask, ksize=11)

    if len(np.unique(result_mask)) < 3:
        iou_result = jaccard_score(
            label_mask.flatten(), result_mask.flatten()
        )  # , average="macro")
        iou_smooth = jaccard_score(
            label_mask.flatten(), smooth_mask.flatten()
        )  # , average="macro")

    else:
        iou_result = jaccard_score(
            label_mask.flatten(), result_mask.flatten(), average="macro"
        )
        iou_smooth = jaccard_score(
            label_mask.flatten(), smooth_mask.flatten(), average="macro"
        )

    print(path)
    print(f"IOU Result: {iou_result:.3f}, IOU Smooth: {iou_smooth:.3f}")

    result_mask = mask_to_rgb(result_mask.reshape(256, 256, 1))
    smooth_mask = mask_to_rgb(smooth_mask.reshape(256, 256, 1))

    plt.axis("off")

    # plt.imshow(label)
    # plt.savefig(
    #     f"/Users/andrematte/Developer/Projects/phd/dam_segmentation/experiments/random_forests/4-inference/results/{i}_label.png",
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )

    # plt.imshow(result_mask)
    # plt.savefig(
    #     f"/Users/andrematte/Developer/Projects/phd/dam_segmentation/experiments/random_forests/4-inference/results/{i}_result_IoU_{iou_result:.3f}.png",
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )

    # plt.imshow(cv2.absdiff(label, result_mask))
    # plt.savefig(
    #     f"/Users/andrematte/Developer/Projects/phd/dam_segmentation/experiments/random_forests/4-inference/results/{i}_result_diff.png",
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )

    plt.imshow(smooth_mask)
    # plt.savefig(
    #     f"/Users/andrematte/Developer/Projects/phd/dam_segmentation/experiments/random_forests/4-inference/results/{i}_smooth_IoU_{iou_smooth:.3f}.png",
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )

    # plt.imshow(cv2.absdiff(label, smooth_mask))
    # plt.savefig(
    #     f"/Users/andrematte/Developer/Projects/phd/dam_segmentation/experiments/random_forests/4-inference/results/{i}_smooth_diff.png",
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )
    break
