# Copyright (C) 2020 and later: Google, Inc.

import cv2
import enum
import numpy as np

class ImgFormat(enum.Enum):
    """Enumeration class for image format."""
    RGB = 1 # RGB format, necessary for ClearType
    A8 = 2 # 8-bit grayscale format
    A1 = 3 # black and white grayscale format
    EMBEDDINGS = 4 # Arbitrary sized array for

class Distance:
    """Generator of distance metrics for specific image format.

    Initialization:
        >>> dis = Distance(img_format=ImgFormat.RGB)
    Get available distance metrics for specific image format:
        >>> metrics = dis.get_metrics()
    'metrics' is a dictionary mapping from distance name to distance function：
        >>> metrics.keys() # dict_keys(['manhattan', 'sum_squared', ...])
    Calculate distance between two image arrays:
        >>> metrics['manhattan'](img1, img2)
    """

    def __init__(self, img_format=ImgFormat.RGB):
        """Store image format.

        Args:
            img_format: ImgFormat, format of input.
        """

        self.img_format = img_format

    @property
    def img_format(self):
        """
        Returns:
                self._img_format: ImgFormat, format of input.
        """
        return self._img_format

    @img_format.setter
    def img_format(self, img_format):
        """Raises TypeError if img_format is not a member of ImgFormat class."""
        if img_format not in list(ImgFormat):
            raise TypeError('Expect img_format to be a member of ImgFormat '
                            'class.')
        else:
            self._img_format = img_format

    def get_metrics(self):
        """Return a dictionary of compatible distance names to functions.

        Returns:
            distances: Dict, mapping from names to distance functions.

        Raises:
            NotImplemented: if no functions implemented for current format.
        """
        if self.img_format == ImgFormat.RGB:
            metrics = {
                'manhattan': self._manhattan_distance_rgb,
                'sum_squared': self._sum_squared_distance_rgb,
                'cross_correlation': self._cross_correlation_distance_rgb
            }
        elif self.img_format == ImgFormat.A1 or self.img_format == ImgFormat.A8:
            metrics = {
                'manhattan': self._manhattan_distance_gray,
                'sum_squared': self._sum_squared_distance_gray,
                'cross_correlation': self._cross_correlation_distance_gray
            }
        elif self.img_format == ImgFormat.EMBEDDINGS:
            metrics = {
                'manhattan': self._manhattan_distance_emb,
                'euclidean': self._euclidean_distance_emb
            }
        else:
            raise NotImplemented()

        return metrics

    def _check_image_type_and_shape(self, img1, img2, dimension):
        """Check two input images:
            1. are the same type
            2. have the same dimension
            3. have the same shape.

        Args:
            img1: np.ndarray, image array.
            img2: np.ndarray, image array.

        Raises:
            TypeError: if img1 or img2 are not np.ndarray.
            ValueError: is img1 and img2 has non-compatible shape.
        """
        # Check both images are nd.array
        if (type(img1) != np.ndarray) or (type(img2) != np.ndarray):
            raise TypeError('Expect both images to be of type numpy.ndarray.')
        # Check both images have the same dimension
        if len(img1.shape) != dimension or len(img2.shape) != dimension:
            raise ValueError('Expect 2d array as input.')
        # Check both images have the same shape
        if img1.shape != img2.shape:
            raise ValueError('Cannot calculate distance between two images with'
                             ' different shape.')

    def _manhattan_distance_emb(self, emb1, emb2):
        """Get the average distance between every pair of corresponding
        pixels in the two images.

        Args:
            emb1: np.ndarray, embeddings.
            emb2: np.ndarray, embeddings.

        Raises:
            TypeError: if img1 or img2 are not np.ndarray
            ValueError: is img1 and img2 has non-compatible shape
        """
        # Check image type and shape
        self._check_image_type_and_shape(emb1, emb2, 1)

        dis = np.absolute(emb1 - emb2)
        total_dis = np.sum(dis)
        return total_dis

    def _euclidean_distance_emb(self, emb1, emb2):
        """Get euclidean distance between two representations.

         Args:
            emb1: np.ndarray, embeddings.
            emb2: np.ndarray, embeddings.

        Raises:
            TypeError: if img1 or img2 are not np.ndarray
            ValueError: is img1 and img2 has non-compatible shape
        """
        # Check image type and shape
        self._check_image_type_and_shape(emb1, emb2, 1)

        total_dis = np.linalg.norm(emb1 - emb2)
        return total_dis

    def _manhattan_distance_rgb(self, img1, img2):
        """Get the average distance between every pair of corresponding
        pixels in the two images. Expect both images to be rgb image (the
        shape must be [image_height, image_width, 3]).

        Args:
            img1: np.ndarray, 3d array representing the first image with shape
                [image_height, image_width, 3]
            img2: np.ndarray, 3d array representing the second image

        Returns:
            distance: Float, the manhattan distance between the two images.
        """
        # Check image type and shape
        self._check_image_type_and_shape(img1, img2, 3)

        # Split into 3 channels
        b1, g1, r1 = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
        b2, g2, r2 = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]

        # Calculate manhattan distance in 3 channels
        b_dis = self._manhattan_distance_gray(b1, b2)
        g_dis = self._manhattan_distance_gray(g1, g2)
        r_dis = self._manhattan_distance_gray(r1, r2)

        total_dis = b_dis + g_dis + r_dis
        distance = total_dis / 3
        return distance

    def _manhattan_distance_gray(self, img1, img2):
        """Get the sum of the distance between every pair of corresponding
        pixels in the two images. Expect both images to be grayscale image (the
        shape must be [image_height, image_width]).

        Args:
            img1: np.ndarray, 2d array representing the first image with shape
                [image_height, image_width]
            img2: np.ndarray, 2d array representing the second image

        Returns:
            distance: Float, the manhattan distance between the two images.
        """
        # Check image type and shape
        self._check_image_type_and_shape(img1, img2, 2)

        # Calculate manhattan distance
        total_pxs = img1.shape[0] * img1.shape[1]
        im_dis = np.absolute(img1 - img2)
        total_dis = np.sum(im_dis)

        distance = total_dis / total_pxs
        return distance

    def _sum_squared_distance_rgb(self, img1, img2):
        """Get normalized sum squared difference.
        distance = sum()

        Args:
            img1: np.ndarray, 3d array representing the first image with shape
                [image_height, image_width, 3]
            img2: np.ndarray, 3d array representing the second image

        Returns:
            distance: Float, sum square distance between two images
        """
        # Check image type and shape
        self._check_image_type_and_shape(img1, img2, 3)

        # Calculate sum squared distance
        distance = cv2.matchTemplate(img1, img2, cv2.TM_SQDIFF_NORMED)[0][0]
        return distance

    def _sum_squared_distance_gray(self, img1, img2):
        """Get normalized sum squared difference.

        Args:
            img1: np.ndarray, 2d array representing the first image with shape
                [image_height, image_width]
            img2: np.ndarray, 2d array representing the second image

        Returns:
            distance: Float, sum square distance between two images
        """
        # Check image type and shape
        self._check_image_type_and_shape(img1, img2, 2)

        # Simply replicate and stack up on the 3rd dimension
        img1_rgb = np.stack((img1,)*3, axis=-1)
        img2_rgb = np.stack((img2,)*3, axis=-1)

        # Calculate sum squared distance
        distance = cv2.matchTemplate(img1_rgb, img2_rgb,
                                     cv2.TM_SQDIFF_NORMED)[0][0]
        return distance

    def _cross_correlation_distance_rgb(self, img1, img2):
        """Get normalized cross correlation difference. This distance is based
        on observation that there is high correlation between spatially close
        pixels for natural images. See
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3596837/ for details.

        Args:
            img1: np.ndarray, 3d array representing the first image with shape
                [image_height, image_width, 3]
            img2: np.ndarray, 3d array representing the second image

        Returns:
            distance: Float, cross correlation distance between two images
        """
        # Check image type and shape
        self._check_image_type_and_shape(img1, img2, 3)

        # Calculate sum squared distance
        distance = cv2.matchTemplate(img1, img2, cv2.TM_SQDIFF_NORMED)[0][0]
        return distance

    def _cross_correlation_distance_gray(self, img1, img2):
        """Get normalized cross correlation difference. This distance is based
        on observation that there is high correlation between spatially close
        pixels for natural images. See
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3596837/ for details.

        Args:
            img1: np.ndarray, 2d array representing the first image with shape
                [image_height, image_width]
            img2: np.ndarray, 2d array representing the second image

        Returns:
            distance: Float, cross correlation distance between two images
        """
        # Check image type and shape
        self._check_image_type_and_shape(img1, img2, 2)

        # Simply replicate and stack up on the 3rd dimension
        img1_rgb = np.stack((img1,) * 3, axis=-1)
        img2_rgb = np.stack((img2,) * 3, axis=-1)

        # Calculate sum squared distance
        distance = cv2.matchTemplate(img1_rgb, img2_rgb, cv2.TM_SQDIFF_NORMED)[0][0]
        return distance
