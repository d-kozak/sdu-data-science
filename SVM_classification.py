import numpy as np
from skimage import io

classes = {'vegetation': 0, 'building': 1, 'water': 2}
n_classes = len(classes)
palette = np.uint8([[0, 255, 0], [255, 0, 0], [0, 0, 255]])

img = io.imread('https://i.stack.imgur.com/TFOv7.png')
rows, cols, bands = img.shape