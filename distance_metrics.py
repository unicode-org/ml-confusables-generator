from PIL import Image
import numpy as np

def naive_distance(im1, im2):
    """Get the sum of the distance between every pair of corresponding pixels in the two images.
    The two images needs to be grayscale image (the shape must be [image_height, image_width]).

    Args:
        im1: 2d numpy array representing the first image with shape [image_height, image_width]
        im2: 2d numpy array representing the second image.
        
    Returns:
        distance: Int, the pixel to pixel distance between the two images.
    """
    
    if (type(im1) != np.ndarray) or (type(im2) != np.ndarray):
        raise TypeError('Function naive_distance expects numpy.ndarray as input.')
    if len(im1.shape) != 2 or len(im2.shape) != 2:
        raise ValueError('Function naive_distance expect 2d array as input.')

    if im1.shape != im2.shape:
        raise ValueError('Cannot calculate distance between two images with different shape.')

    # Calculate naive distance
    total_pxs = im1.shape[0] * im1.shape[1]
    im_dis = np.absolute(im1 - im2)
    total_dis = np.sum(im_dis)

    distance = total_dis / total_pxs
    return distance



if __name__ == "__main__":
    im1 = np.asarray(Image.open('img_out/Noto_Sans_CJK_SC/63847.png'))
    im1 = im1.mean(axis=2)
    
    im2 = np.asarray(Image.open('img_out/Noto_Sans_CJK_SC/23506.png'))
    im2 = im2.mean(axis=2)
    
    dis = naive_distance(im1, im2)
    import pdb;pdb.set_trace()
