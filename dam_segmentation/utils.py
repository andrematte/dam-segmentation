import logging
import os

import cv2
import numpy as np

from damseg_ml.settings import MASK_COLORS, SEGMENTATION_CLASSES


def logger_setup(to_file=True):
    log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    if to_file:
        log_file = "../../logs.log"
    else:
        log_file = None
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename=log_file)
    logger = logging.getLogger(__name__)
    return logger


def rgb_to_mask(rgb_mask, bgr=False):
    """
    Convert image mask from rgb values to int values.
    """
    # If the original mask is in BGR, converts to RGB first
    if bgr:
        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)

    mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)

    # Change pixels for class values
    for segmentation_class in SEGMENTATION_CLASSES.keys():
        mask[rgb_mask[:, :, 0] == MASK_COLORS[segmentation_class][0]] = (
            SEGMENTATION_CLASSES[segmentation_class]
        )

    # Raises an error if the class assigned does not work
    if not set(np.unique(mask)).issubset(SEGMENTATION_CLASSES.values()):
        raise ValueError("Error in mask conversion from RGB to Labels.")

    return mask


def mask_to_rgb(mask):
    """
    Convert image mask from int values to rgb values.
    """
    height, width = mask.shape[:2]
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for segmentation_class in SEGMENTATION_CLASSES.keys():
        temp_mask = np.all(
            mask == SEGMENTATION_CLASSES[segmentation_class], axis=2
        )
        rgb_mask[temp_mask] = np.array(MASK_COLORS[segmentation_class])

    return rgb_mask


def create_directory(directory):
    """
    Create a new directory if it does not exist.

    Args:
        directory (str): Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory)
