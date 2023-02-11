# with the help of ChatGPT and provided examples...

# qA find sift points
# qB ...
# qC ...
# qD ...

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from typing import Optional


class Image:
    def __init__(self, image_path: str):
        self.image_path = image_path

        self.image = cv2.imread(image_path)
        self.name = os.path.basename(image_path)
        self.gray: Optional[cv2.Mat] = None

    def set_gray(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


def sift_matches(
    image_1: Image,
    image_2: Image,
    num_matches: Optional[int] = None,
    from_existing_files: bool = False,
    show_zero_singular_values: bool = False,
    show_USV: bool = False,
    show_null_space: bool = False,
    show_agreeing_points: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate SIFT keypoint matches between two images.

    Parameters
    ----------
    image_1 : Image
        The first image to match keypoints in.
    image_2 : Image
        The second image to match keypoints in.
    num_matches : int, optional
        The number of matches to return. If `None`, all matches are returned.
    from_existing_files : bool, optional
        If `True`, the function will attempt to load the SIFT matches from existing files.
        If `False`, the function will calculate the SIFT matches.
    show_zero_singular_values : bool, optional
        If `True`, the function will show a histogram of the zero singular values.
    show_USV : bool, optional
        If `True`, the function will show the U, S, and V matrices.
    show_null_space : bool, optional
        If `True`, the function will show the null space of the A matrix.
    show_agreeing_points : bool, optional
        If `True`, the function will show the points that agree with the homography.

    Returns
    -------
    X1 : numpy.ndarray
        An (3, numMatches) shaped array containing the matched keypoints in `image_1`.
    X2 : numpy.ndarray
        An (3, numMatches) shaped array containing the matched keypoints in `image_2`.
    image_matches : numpy.ndarray
        An image of the matches between the two images.

    Notes
    -----
    The SIFT keypoint matching algorithm is used to find correspondences between keypoints in two images.
    The function first converts the input images to grayscale and then computes SIFT keypoints and descriptors for each image.
    The function then uses the brute force matcher from OpenCV to match the keypoints between the two images.
    The function then returns the matched keypoints in the two images as (3, numMatches) shaped arrays and an image of the matches.
    """
    if from_existing_files:
        print(f"Loading SIFT matches between {image_1.name} and {image_2.name}...")
        X1 = np.load(f"sift_{image_1.name}_X1.npy")
        X2 = np.load(f"sift_{image_2.name}_X2.npy")
        image_matches = np.load(f"sift_{image_1.name}_matches.npy")
        return X1, X2, image_matches

    print(f"Calculating SIFT matches between {image_1.name} and {image_2.name}...")

    image_1.set_gray()
    image_2.set_gray()

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1.gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2.gray, None)

    bf = cv2.BFMatcher()  # brute force matcher
    matches: tuple = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    num_matches_found = len(matches)
    print(f"Matches found: {num_matches_found}")

    num_matches = num_matches_found if num_matches is None else num_matches

    # everything below this is beyond me how it works, lot of trial and error to
    # just get it to run with no errors. The 'inliers' stuff are good points in
    # the image, but at no point could I get the errot to be < 36 as the example
    # code had.
    # I also didn't do any manual refinement...

    X1 = np.zeros((3, num_matches_found))
    X2 = np.zeros((3, num_matches_found))

    for i, match in enumerate(matches):
        X1[:, i] = (
            keypoints_1[match.queryIdx].pt[0],
            keypoints_1[match.queryIdx].pt[1],
            1,
        )
        X2[:, i] = (
            keypoints_2[match.trainIdx].pt[0],
            keypoints_2[match.trainIdx].pt[1],
            1,
        )

    num_trials = 100
    homographies = []
    scores = []
    inliers = []
    all_zero_singular_values = []
    all_usvs = []

    for i in range(num_trials):
        subset = random.sample(list(range(num_matches_found)), k=4)
        A = []
        for j in subset:
            x1 = X1[:, j]
            x2 = X2[:, j]
            A.append(
                np.kron(
                    x1.T,
                    np.array(
                        [[0, -x2[2], x2[1]], [x2[2], 0, -x2[0]], [-x2[1], x2[0], 0]]
                    ),
                )
            )
        A = np.concatenate(A, axis=0)

        U, S, V = np.linalg.svd(A)
        all_usvs.append((U, S, V))
        zero_singular_values = S[np.isclose(S, 0, atol=1e-6)]
        all_zero_singular_values.append(zero_singular_values)
        homography = V[:, 8].reshape((3, 3))
        homographies.append(homography)

        X2_ = np.matmul(homography, X1)
        du = X2_[0, :] / X2_[2, :] - X2[0, :] / X2[2, :]
        dv = X2_[1, :] / X2_[2, :] - X2[1, :] / X2[2, :]
        inliers_error = du**2 + dv**2
        _inliers = np.where(inliers_error < 6**2)[0]
        _inliers = [min(inliers_error)] if np.shape(_inliers) == (0,) else _inliers
        inliers.append(_inliers)

        score = sum(_inliers)
        scores.append(score)

    best_i = np.argmax(scores)
    best_homography = homographies[best_i]
    best_inliers = inliers[best_i]
    zero_singular_values = all_zero_singular_values[best_i]
    U, S, V = all_usvs[best_i]

    if show_zero_singular_values:
        print(f"Zero singular values: {zero_singular_values}")

    if show_USV:
        print(f"U: {U}")
        print(f"S: {S}")
        print(f"V: {V}")

    if show_null_space:
        rank = np.sum(np.abs(S) > 1e-6)
        null_space = V[:, rank:]
        print(f"Null space: {null_space}")

    if show_agreeing_points:
        X2_ = np.dot(best_homography, X1)
        du = X2_[0, :] / X2[0, :] - X2[0, :] / X2[2, :]
        dv = X2_[1, :] / X2[1, :] - X2[1, :] / X2[2, :]
        inlier_indices = np.where(du**2 + dv**2 < 6**2)[0]
        num_inliers = max(len(inlier_indices), 1)
        print(f"Number of inliers: {num_inliers}")

    # everything below is simply for drawing the matches
    image_matches = cv2.drawMatches(
        img1=image_1.image,
        keypoints1=keypoints_1,
        img2=image_2.image,
        keypoints2=keypoints_2,
        matches1to2=matches[:num_matches],
        outImg=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    for match in matches[:num_matches]:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = keypoints_1[img1_idx].pt
        (x2, y2) = keypoints_2[img2_idx].pt

        # Draw circles around the keypoints
        cv2.circle(
            image_matches, (int(x1), int(y1)), radius=10, color=(0, 255, 0), thickness=2
        )
        cv2.circle(
            image_matches,
            (int(x2) + image_1.image.shape[1], int(y2)),
            radius=10,
            color=(0, 255, 0),
            thickness=2,
        )

    np.save(f"sift_{image_1.name}_X1.npy", X1)
    np.save(f"sift_{image_2.name}_X2.npy", X2)
    np.save(f"sift_{image_1.name}_matches.npy", image_matches)

    return X1, X2, image_matches


if __name__ == "__main__":
    COMPUTE_SIFT_MATCHES = True  # otherwise load from existing files

    directory = os.path.dirname(__file__)
    image_1_path = os.path.join(directory, "sift_1.png")
    image_2_path = os.path.join(directory, "sift_2.png")

    image_1 = Image(image_1_path)
    image_2 = Image(image_2_path)

    X1, X2, image_matches = sift_matches(
        image_1,
        image_2,
        num_matches=8,
        from_existing_files=not COMPUTE_SIFT_MATCHES,
        show_zero_singular_values=True,
        show_USV=True,
        show_null_space=True,
        show_agreeing_points=True,
    )

    # B) Show the SVD of A and what are the zero singular values ? [10 pts]

    plt.figure()
    plt.imshow(image_matches)
    plt.show()
