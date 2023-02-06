# with the help of ChatGPT and provided examples...

# qA find sift points
# qB ...
# qC ...
# qD ...

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
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
    image_1: Image, image_2: Image, num_matches: Optional[int] = None
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

    Returns
    -------
    X1 : numpy.ndarray
        An (3, numMatches) shaped array containing the matched keypoints in `image_1`.
    X2 : numpy.ndarray
        An (3, numMatches) shaped array containing the matched keypoints in `image_2`.

    Notes
    -----
    The SIFT keypoint matching algorithm is used to find correspondences between keypoints in two images.
    The function first converts the input images to grayscale and then computes SIFT keypoints and descriptors for each image.
    The function then uses the brute force matcher from OpenCV to match the keypoints between the two images.
    The function then returns the matched keypoints in the two images as (3, numMatches) shaped arrays and an image of the matches.
    """
    print(f"Calculating SIFT matches between {image_1.name} and {image_2.name}...")

    image_1.set_gray()
    image_2.set_gray()

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1.gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2.gray, None)

    bf = cv2.BFMatcher()  # brute force matcher
    matches = bf.match(descriptors_1, descriptors_2)

    num_matches_found = len(matches)
    print(f"Matches found: {num_matches_found}")

    num_matches = num_matches_found if num_matches is None else num_matches

    X1 = np.zeros((3, num_matches))
    X2 = np.zeros((3, num_matches))

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    for i, match in enumerate(matches[:num_matches]):
        X1[:, i] = keypoints_1[match.queryIdx].pt + (1,)
        X2[:, i] = keypoints_2[match.trainIdx].pt + (1,)

    # everything below is simply for drawing the matches
    img_matches = cv2.drawMatches(
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
            img_matches, (int(x1), int(y1)), radius=10, color=(0, 255, 0), thickness=2
        )
        cv2.circle(
            img_matches,
            (int(x2) + image_1.image.shape[1], int(y2)),
            radius=10,
            color=(0, 255, 0),
            thickness=2,
        )

    return X1, X2, img_matches


if __name__ == "__main__":
    directory = os.path.dirname(__file__)
    image_1_path = os.path.join(directory, "sift_1.png")
    image_2_path = os.path.join(directory, "sift_2.png")

    image_1 = Image(image_1_path)
    image_2 = Image(image_2_path)

    X1, X2, image_matches = sift_matches(image_1, image_2, num_matches=80)

    plt.figure()
    plt.imshow(image_matches)
    plt.show()
