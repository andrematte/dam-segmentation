import itertools
from glob import glob

import cv2
import numpy as np
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from ms_image_tool.image import Image
from scipy import ndimage as nd
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import prewitt, roberts, scharr, sobel

from damseg_ml.utils import logger_setup, rgb_to_mask

logger = logger_setup(to_file=False)


class Features:
    def __init__(
        self,
        image_path: str,
        label_path: str = None,
    ) -> None:
        self.image_path = image_path
        self.label_path = label_path
        self.image = Image(self.image_path)

        logger.info(f"---> File: {self.image_path}")
        logger.info(f"---> Labels: {self.label_path}")
        logger.info("-----> Initiating feature extraction")

        self.features = self.extract_features()

        logger.info("-----> Done!")

    def extract_features(self):
        """
        Creates a dataframe containing all the following features from a multispectral image:
        - Gray Image
        - R, G, B, REDEDGE and NIR spectral bands
        - Vegetation Indexes (NDRE, NDVI, ...)
        - Filtered versions of the above
        """
        df = pd.DataFrame(
            index=np.arange(0, self.image.shape[0] * self.image.shape[1])
        )

        logger.info("-----> Extracting basic features")

        df["gray"] = self.image.get_gray().reshape(-1)
        df["red"] = self.image.red.reshape(-1)
        df["green"] = self.image.green.reshape(-1)
        df["blue"] = self.image.blue.reshape(-1)
        df["rededge"] = self.image.rededge.reshape(-1)
        df["nir"] = self.image.nir.reshape(-1)
        df["ndvi"] = self.image.get_ndvi().reshape(-1)
        df["gndvi"] = self.image.get_gndvi().reshape(-1)
        df["ndre"] = self.image.get_ndre().reshape(-1)
        df["ndwi"] = self.image.get_ndwi().reshape(-1)

        logger.info("-----> Extracting texture features")
        texture_features = extract_glcm_features(self.image.get_gray())

        logger.info("-----> Extracting filter features")
        filter_features = extract_filter_features(self.image.get_gndvi())
        gabor_features, _, _ = extract_gabor_features(self.image.get_gndvi())

        df = (
            df.join(gabor_features)
            .join(filter_features)
            .join(texture_features)
        )

        if self.label_path:
            df["label"] = rgb_to_mask(
                tifffile.imread(self.label_path)
            ).reshape(-1)

        self.names = df.columns

        return df

    def original_shape(self, column="Gray"):
        """
        Reshape image back to original format (width x height).
        """
        return self.features[column].values.reshape(self.image.shape)


def create_dataset(image_dir: list[str], label_dir: list[str]) -> None:
    logger.info("-> Initiating dataset creation")

    image_paths = glob(f"{image_dir}/*.tiff")
    label_paths = glob(f"{label_dir}/*.tiff")
    logger.info(f"-> Images: {len(image_paths)}. Labels: {len(label_paths)}")
    assert len(image_paths) == len(
        label_paths
    ), "Number of images and labels must be the same"

    list_of_dfs = []

    for image, label in zip(image_paths, label_paths):
        features = Features(image, label)
        list_of_dfs.append(features.features)

    if len(list_of_dfs) > 1:
        dataset = pd.concat(list_of_dfs, ignore_index=True)
        return dataset


def extract_filter_features(image: np.ndarray) -> pd.DataFrame:
    """
    Generates a dataframes containing image features acquired from a set of filters.
    """

    filter_features = pd.DataFrame()
    filter_features["canny"] = canny_filter(image).reshape(-1)
    filter_features["laplacian"] = laplacian_filter(image).reshape(-1)
    filter_features["roberts"] = roberts_filter(image).reshape(-1)
    filter_features["sobel"] = sobel_filter(image).reshape(-1)
    filter_features["scharr"] = scharr_filter(image).reshape(-1)
    filter_features["prewitt"] = prewitt_filter(image).reshape(-1)

    filter_features["gaussian 3"] = gaussian_filter(image, 3).reshape(-1)
    filter_features["gaussian 7"] = gaussian_filter(image, 7).reshape(-1)
    filter_features["gaussian 12"] = gaussian_filter(image, 12).reshape(-1)
    filter_features["gaussian 15"] = gaussian_filter(image, 15).reshape(-1)
    filter_features["median 7"] = median_filter(image, 7).reshape(-1)

    return filter_features


def extract_gabor_features(image: np.ndarray) -> pd.DataFrame:
    """
    Generates a series of gabor kernels and apply them to the input image.
    """
    df = pd.DataFrame()

    thetas = [theta / 4.0 * np.pi for theta in [0, 1, 2, 3]]
    sigmas = [1, 3]
    lamdas = np.arange(0, np.pi, np.pi / 4)[1:]
    gammas = [0.05, 0.25]
    kernel_size = 9
    kernels = []

    for i, (theta, sigma, lamda, gamma) in enumerate(
        itertools.product(thetas, sigmas, lamdas, gammas)
    ):
        label = f"gabor_{i}"
        parameters = (
            f"Theta: {theta}, Sigma: {sigma}, Lambda: {lamda}, Gamma: {gamma}"
        )

        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma,
            theta,
            lamda,
            gamma,
            0,
            ktype=cv2.CV_32F,
        )
        kernels.append(kernel)

        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        df[label] = filtered.reshape(-1)

    gabor_parameters = parameters
    gabor_kernels = kernels

    return df, gabor_parameters, gabor_kernels


def calculate_glcm_for_window(window, distances, angles, levels):
    glcm = graycomatrix(
        window,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    feature_vector = {
        "contrast": graycoprops(glcm, "contrast").flatten()[0],
        "dissimilarity": graycoprops(glcm, "dissimilarity").flatten()[0],
        "homogeneity": graycoprops(glcm, "homogeneity").flatten()[0],
        "entropy": graycoprops(glcm, "energy").flatten()[0],
        "correlation": graycoprops(glcm, "correlation").flatten()[0],
        "asm": graycoprops(glcm, "ASM").flatten()[0],
    }
    return feature_vector


def extract_glcm_features(
    image, distances=[1], angles=[0], levels=256, window_size=3, n_jobs=-1
):
    """
    Extract GLCM features for each pixel in the image over a neighborhood defined by window_size.
    Returns a pandas DataFrame where each column is a feature and each row represents a pixel.
    """
    features = []
    half_window = window_size // 2
    padded_image = np.pad(
        image, pad_width=half_window, mode="constant", constant_values=0
    )

    # Use parallel processing to speed up feature extraction
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_glcm_for_window)(
            padded_image[
                i - half_window : i + half_window + 1,
                j - half_window : j + half_window + 1,
            ],
            distances,
            angles,
            levels,
        )
        for i in range(half_window, image.shape[0] + half_window)
        for j in range(half_window, image.shape[1] + half_window)
    )

    # Convert results to a pandas DataFrame
    features = pd.DataFrame(results)

    return features


def canny_filter(
    image: np.ndarray, min: int = 120, max: int = 260
) -> np.ndarray:
    """
    Apply the Canny Edge Detection filter to the input image.
    """
    image = np.uint8(image)
    return cv2.Canny(image, min, max, 3, L2gradient=True)


def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply the Laplacian filter to the input image.
    """
    image = np.uint8(image)
    blur = cv2.GaussianBlur(image, (9, 9), 0)
    return cv2.Laplacian(blur, cv2.CV_8U)


def roberts_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply the Roberts filter to the input image.
    """
    return roberts(image)


def sobel_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply the Sobel filter to the input image.
    """
    return sobel(image)


def scharr_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply the Scharr filter to the input image.
    """
    return scharr(image)


def prewitt_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply the Prewitt filter to the input image.
    """
    return prewitt(image)


def gaussian_filter(image: np.ndarray, sigma: int) -> np.ndarray:
    return nd.gaussian_filter(image, sigma)


def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    return nd.median_filter(image, size)
