import cv2
import itertools
from itertools import chain
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pickle
import random
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import Optional

USER_ID = "16340679"

LABELS = [
    "airplane",
    "airport",
    "baseball_diamond",
    "basketball_court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular_farmland",
    "cloud",
    "commercial_area",
    "dense_residential",
    "desert",
    "forest",
    "freeway",
    "golf_course",
    "ground_track_field",
    "harbor",
    "industrial_area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "medium_residential",
    "mobile_home_park",
    "mountain",
    "overpass",
    "palace",
    "parking_lot",
    "railway",
    "railway_station",
    "rectangular_farmland",
    "river",
    "roundabout",
    "runway",
    "sea_ice",
    "ship",
    "snowberg",
    "sparse_residential",
    "stadium",
    "storage_tank",
    "tennis_court",
    "terrace",
    "thermal_power_station",
    "wetland",
]

LABEL_SETS = [
    [
        19,
        18,
        21,
        42,
        11,
        37,
        25,
        16,
        10,
        36,
    ],
    [
        39,
        33,
        43,
        29,
        16,
        8,
        5,
        32,
        44,
        42,
    ],
    [
        31,
        3,
        6,
        41,
        22,
        45,
        17,
        38,
        25,
        39,
    ],
    [
        7,
        14,
        8,
        35,
        5,
        45,
        9,
        4,
        18,
        15,
    ],
    [
        16,
        36,
        22,
        29,
        4,
        43,
        11,
        23,
        13,
        8,
    ],
    [
        40,
        7,
        19,
        4,
        34,
        23,
        16,
        24,
        13,
        44,
    ],
    [
        45,
        10,
        29,
        31,
        27,
        4,
        7,
        44,
        37,
        23,
    ],
    [
        14,
        40,
        32,
        38,
        21,
        11,
        22,
        18,
        2,
        3,
    ],
    [
        7,
        26,
        19,
        37,
        39,
        1,
        9,
        25,
        43,
        5,
    ],
    [
        21,
        1,
        20,
        37,
        4,
        30,
        17,
        19,
        42,
        18,
    ],
]


class ImageFeatureOperations:
    SIFT = "sift"
    HOG = "hog"
    DSIFT = "dsift"


class FisherVectorGMM:
    def __init__(self, means: np.ndarray, covariances: np.ndarray, priors: np.ndarray):
        self.means = means
        self.covariances = covariances
        self.priors = priors

    @property
    def eignvalues_and_eigenvectors(self) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigvenvectors = np.linalg.eig(self.covariances)
        return eigenvalues, eigvenvectors


class Image:
    def __init__(self, path: str, label: str, rgb_iamge: cv2.Mat):
        self.path = path
        self.label = label
        self.rgb_image = rgb_iamge

        self.name = path.split("/")[-1]
        self.hsv_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        self.flattened_hsv_image = self.hsv_image.reshape(-1, 3)
        self.grayscale_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)

        self.sift_features: np.ndarray[np.float32] = np.array([])
        self.hog_features: np.ndarray[np.float32] = np.array([])
        self.dsift_features: np.ndarray[np.float32] = np.array([])


def load_images(
    root: str, labels: set[str], num_images_per_label: int, from_pickle: bool = False
) -> dict[str, list[Image]]:
    """ """
    if from_pickle:
        print(f"Loading images from pickle...")

        with open("images.pickle", "rb") as f:
            images = pickle.load(f)
        return images

    print(f"Loading images from {root}...")
    folders = os.listdir(root)
    images: dict[str, list[Image]] = {}

    for label in folders:
        if label not in labels:
            continue

        print(f"Loading {label} images...")
        if label not in images:
            images[label] = []

        for i, image in enumerate(
            random.choices(os.listdir(root + label), k=num_images_per_label)
        ):
            img = cv2.imread(f"{root}{label}/{image}")
            images[label].append(
                Image(path=f"{root}{label}/{image}", label=label, rgb_iamge=img)
            )

    with open("images.pickle", "wb") as f:
        pickle.dump(images, f)

    return images


def get_image_features(image: Image, operation: str) -> Image:
    """
    Extracts the specified feature type from the input image and stores the result in the corresponding
    attribute of the Image object.

    Parameters
    ----------
    image : Image
        The image to extract features from.
    operation : str
        The feature type to extract. One of "sift", "hog", "dsift".

    Returns
    -------
    Image
        The input image with the extracted features stored in the corresponding attribute.

    Raises
    ------
    ValueError
        If the specified operation is not supported.
    """

    if operation == ImageFeatureOperations.SIFT:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image.grayscale_image, None)
        image.sift_features = descriptors
        return image

    if operation == ImageFeatureOperations.HOG:
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(image.grayscale_image)
        image.hog_features = hog_features
        return image

    if operation == ImageFeatureOperations.DSIFT:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image.grayscale_image, None)
        image.dsift_features = descriptors
        return image

    raise ValueError(
        f"Operation {operation} not supported, use one of {ImageFeatureOperations.__dict__.keys()}"
    )


def get_fisher_vector_model(
    features: np.ndarray[np.float32], kd: int, number_GMM_components: int
) -> FisherVectorGMM:
    """
    Fits a Fisher Vector model to the input features.

    Parameters
    ----------
    features : np.ndarray[np.float32]
        A numpy array of shape (n, d) containing n features of dimension d.
    kd : int
        The desired lower dimension of the feature.
    number_GMM_components : int
        The number of Gaussian components to use in the GMM.

    Returns
    -------
    FisherVectorGMM
        The Fisher Vector model.
    """

    pca = PCA(n_components=kd)
    f0 = pca.fit_transform(features)

    gmm = GaussianMixture(n_components=number_GMM_components)
    gmm.fit(f0)

    means: np.ndarray = gmm.means_
    covariances: np.ndarray = gmm.covariances_
    priors: np.ndarray = gmm.weights_

    fv_gmm = FisherVectorGMM(means, covariances, priors)
    return fv_gmm


if __name__ == "__main__":
    nwpu_path = os.path.join(os.path.dirname(__file__), "../NWPU-RESISC45/")

    labels = [LABELS[i - 1] for i in LABEL_SETS[int(USER_ID[-1]) - 1]]

    LOAD_IMAGES = False
    NUM_IMAGES_PER_LABEL = 100

    images = load_images(
        nwpu_path,
        labels=labels,
        num_images_per_label=NUM_IMAGES_PER_LABEL,
        from_pickle=not LOAD_IMAGES,
    )

    f_sift: list[np.ndarray[np.float32]] = []
    # f_hog: list[np.ndarray[np.float32]] = []
    f_dsift: list[np.ndarray[np.float32]] = []

    if LOAD_IMAGES:
        for label in images:
            print(f"Extracting features for {label} images...")
            for image in images[label]:
                image = get_image_features(image, ImageFeatureOperations.SIFT)
                # image = get_image_features(image, ImageFeatureOperations.HOG)
                image = get_image_features(image, ImageFeatureOperations.DSIFT)
                f_sift.append(image.sift_features)
                # f_hog.append(image.hog_features)
                f_dsift.append(image.dsift_features)

        with open("images.pickle", "wb") as f:
            print(f"Saving images with features to pickle...")
            pickle.dump(images, f)

        with open("sift_features.pickle", "wb") as f:
            print(f"Saving sift features to pickle...")
            pickle.dump(f_sift, f)

        # with open("hog_features.pickle", "wb") as f:
        #     print(f"Saving hog features to pickle...")
        #     pickle.dump(f_hog, f)

        with open("dsift_features.pickle", "wb") as f:
            print(f"Saving dsift features to pickle...")
            pickle.dump(f_dsift, f)

    else:
        for label in images:
            print(f"Loading features for {label} images...")
            for image in images[label]:
                f_sift.append(image.sift_features)
                # f_hog.append(image.hog_features)
                f_dsift.append(image.dsift_features)

    f_sift_fv_gmm = get_fisher_vector_model(f_sift, 10, 10)
    # f_hog_fv_gmm = get_fisher_vector_model(f_hog[0], 10, 10)
    f_dsift_fv_gmm = get_fisher_vector_model(f_dsift, 10, 10)

    # Visualize the eigen values for SIFT, and Dense SIFT features here as a bar plot or line plot two different plots

    f_sift_eigenvalues = np.sort(f_sift_fv_gmm.eignvalues_and_eigenvectors[0])[::-1]
    f_dsift_eigenvalues = np.sort(f_dsift_fv_gmm.eignvalues_and_eigenvectors[0])[::-1]

    # one figure two plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Eigenvalues for SIFT and Dense SIFT features")
    ax1.plot(f_sift_eigenvalues)
    ax1.set_title("SIFT")
    ax2.plot(f_dsift_eigenvalues)
    ax2.set_title("Dense SIFT")
    plt.show()

    print("DONE")
