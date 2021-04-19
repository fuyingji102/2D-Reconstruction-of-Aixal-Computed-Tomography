from PIL import Image
import numpy as np
from scipy import ndimage


image = Image.open(r'D:\file\CT\CT Reconstruction\phantom_image.tif')
img = np.array(image)

def DiscreteRadonTransform(image, steps):
    channels = len(image[0])
    res = np.zeros((channels, channels), dtype='float64')
    for s in range(steps):
        rotation = ndimage.rotate(image.astype('float64'), -s*180/steps, reshape=False).astype('float64')
        #print(sum(rotation).shape)
        res[:,s] = sum(rotation)
    return res


data = DiscreteRadonTransform(img, len(img[0]))


