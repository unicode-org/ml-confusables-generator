# Copyright (C) 2020 and later: Google, Inc.

import cv2

def calculate_from_path(metric, path1, path2):
    """Calculate distance between the two images specified by file path.

    Args:
        metric: Function, distance metric to be used.
        path1: Str, path to the first image.
        path2: Str, path to the second image.

    Returns:
        distance: Float, distance between the two images.
    """

    try:
        img1 = cv2.imread(path1)
    except FileNotFoundError:
        print('Image at path1 not found.')
        raise

    try:
        img2 = cv2.imread(path2)
    except FileNotFoundError:
        print('Image at path2 not found.')
        raise

    return metric(img1, img2)
