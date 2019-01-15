import numpy as np
import matplotlib.pyplot as plt
from skimage import io

classes = {'vegetation': 0, 'building': 1, 'water': 2}
n_classes = len(classes)
palette = np.uint8([[0, 255, 0], [255, 0, 0], [0, 0, 255]])

img = io.imread('https://i.stack.imgur.com/TFOv7.png')
rows, cols, bands = img.shape

from sklearn.cluster import KMeans
X = img.reshape(rows*cols, bands)
kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
unsupervised = kmeans.labels_.reshape(rows, cols)
io.imshow(palette[unsupervised])

plt.imshow(palette[unsupervised], cmap=plt.cm.binary)
plt.show()