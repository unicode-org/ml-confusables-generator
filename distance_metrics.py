r"""Distance metrics module for confusable detection."""

from PIL import Image
import numpy as np
import enum

class ImgFormat(enum.Enum):
    RGB = 1
    A8 = 2 # 8-bit grayscale format
    A1 = 3 # black and white grayscale format
    EMBEDDINGS = 4 # Arbitrary sized array for

class Distance:
    """Generator of distance metrics.

    To use:

    """

    def __init__(self, img_format=ImgFormat.RGB, embedding_len=None):
        """

        Args:
            img_format: Img_Format, format of input.
            embedding_len: Int, length of the embedding array.
        """

        self.img_format = img_format
        self.embedding_len = embedding_len

    @property
    def img_format(self):
        return self.__img_format

    @property
    def embedding_len(self):
        return self.__embedding_len

    @img_format.setter
    def img_format(self, img_format):
        if img_format not in list(ImgFormat):
            raise TypeError('Expect img_format to be a member of Format class.')
        else:
            self.__img_format = img_format

    @embedding_len.setter
    def embedding_len(self, embedding_len):
        if embedding_len is None:
            self.__embedding_len = embedding_len
        elif not isinstance(embedding_len, int):
            raise TypeError('Expect embedding_len to be an integer.')
        elif embedding_len < 0:
            raise ValueError('Expect embedding_len to be non-negative.')
        else:
            self.__embedding_len = embedding_len

    def get_metrics(self):
        """Return a dictionary of compatible distance names to functions.

        Returns:
            distances: Dict, mapping from names to distance functions
        """
        if self.img_format == ImgFormat.RGB:
            metrics = {
                'naive': self.__naive_distance_rgb
            }
        elif self.img_format == ImgFormat.A1 or self.img_format == ImgFormat.A8:
            metrics = {
                'naive': self.__naive_distance_gray
            }
        else:
            raise NotImplemented()

        return metrics

    def calculate_from_path(self, metric, path1, path2):
        """Calculate distance between the two images specified by file path.

        Args:
            metric: Function, distance metric to be used
            path1: Str, path to the first image
            path2: Str, path to the second image

        Returns:
            distance: Float, distance between the two images
        """

        try:
            img1 = np.asarray(Image.open(path1))
        except FileNotFoundError:
            print('Image at path1 not found.')
            raise

        try:
            img2 = np.asarray(Image.open(path2))
        except FileNotFoundError:
            print('Image at path2 not found.')
            raise

        return metric(img1, img2)

    def __naive_distance_rgb(self, img1, img2, average=True):
        """Get the average distance between every pair of corresponding
        pixels in the two images. Expect both images to be rgb image (the
        shape must be [image_height, image_width, 3]).

        Args:
            img1: 3d numpy array representing the first image with shape
                [image_height, image_width, 3]
            img2: 3d numpy array representing the second image.
            average: Bool, whether or not to average over r, g, b channels

        Returns:
            distance: Int, the pixel to pixel distance between the two images.

        Raises:
            TypeError: if img1 or img2 are not np.ndarray
            ValueError: is img1 and img2 has non-compatible shape
        """
        # Split into 3 channels
        b1, g1, r1 = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
        b2, g2, r2 = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]

        # Calculate distance in 3 channels
        b_dis = self.__naive_distance_gray(b1, b2)
        g_dis = self.__naive_distance_gray(g1, g2)
        r_dis = self.__naive_distance_gray(r1, r2)

        total_dis = b_dis + g_dis + r_dis
        distance = total_dis / 3 if average else total_dis
        return distance

    def __naive_distance_gray(self, img1, img2):
        """Get the sum of the distance between every pair of corresponding
        pixels in the two images. Expect both images to be grayscale image (the
        shape must be [image_height, image_width]).

        Args:
            img1: 2d numpy array representing the first image with shape
                [image_height, image_width]
            img2: 2d numpy array representing the second image.

        Returns:
            distance: Int, the pixel to pixel distance between the two images.

        Raises:
            TypeError: if img1 or img2 are not np.ndarray
            ValueError: is img1 and img2 has non-compatible shape
        """

        if (type(img1) != np.ndarray) or (type(img2) != np.ndarray):
            raise TypeError('Expect both images to be of type numpy.ndarray.')
        if len(img1.shape) != 2 or len(img2.shape) != 2:
            raise ValueError('Expect 2d array as input.')

        if img1.shape != img2.shape:
            raise ValueError('Cannot calculate distance between two images with'
                             ' different shape.')

        # Calculate naive distance
        total_pxs = img1.shape[0] * img1.shape[1]
        im_dis = np.absolute(img1 - img2)
        total_dis = np.sum(im_dis)

        distance = total_dis / total_pxs
        return distance
