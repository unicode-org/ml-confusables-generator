r"""Distance metrics module for confusable detection."""

from PIL import Image
import numpy as np

def naive_distance(img1, img2):
    """Get the sum of the distance between every pair of corresponding pixels in
    the two images. Expect both images to be grayscale image (the shape must be 
    [image_height, image_width]).

    Args:
        img1: 2d numpy array representing the first image with shape
            [image_height, image_width]
        img2: 2d numpy array representing the second image.
        
    Returns:
        distance: Int, the pixel to pixel distance between the two images.
        
    Raises:
        TypeError: if img1 or img2 are
    """
    
    if (type(img1) != np.ndarray) or (type(img2) != np.ndarray):
        raise TypeError('Expect both images to be of type numpy.ndarray.')
    if len(img1.shape) != 2 or len(img2.shape) != 2:
        raise ValueError('Expect 2d array as input.')

    if img1.shape != img2.shape:
        raise ValueError('Cannot calculate distance between two images with '
                         'different shape.')

    # Calculate naive distance
    total_pxs = img1.shape[0] * img1.shape[1]
    im_dis = np.absolute(img1 - img2)
    total_dis = np.sum(im_dis)

    distance = total_dis / total_pxs
    return distance



if __name__ == "__main__":
    ord_range = [24049, 24050, 24051, 24052, 24053]
    
    for i in ord_range:
        for j in ord_range:
            im1 = np.asarray(Image.open('img_out/Noto_Sans_CJK_SC/'+str(i)+'.png')).mean(axis=2)
            im2 = np.asarray(Image.open('img_out/Noto_Sans_CJK_SC/'+str(j)+'.png')).mean(axis=2)
            print(chr(i)+' '+chr(j)+' '+str(naive_distance(im1, im2)))

    #im1 = np.asarray(Image.open('img_out/Noto_Sans_CJK_SC/63847.png'))
    #im1 = im1.mean(axis=2)
    
    #im2 = np.asarray(Image.open('img_out/Noto_Sans_CJK_SC/23506.png'))
    #im2 = im2.mean(axis=2)
    
    #dis = naive_distance(im1, im2)
    #import pdb;pdb.set_trace()
