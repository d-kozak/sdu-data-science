import numpy as np
import matplotlib.pyplot as plt
from skimage import io

classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4, 'bubbles': 5}
n_classes = len(classes)
palette = np.uint8([[0, 255, 0], [255, 0, 0], [0, 0, 255], [0, 0, 0], [128,128,128], [255,255,255]])

img = io.imread('images/1.jpg')
img2 = io.imread('images/1.jpg')
rows, cols, bands = img.shape

from sklearn.cluster import KMeans
X = img.reshape(rows*cols, bands)
kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)

supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
supervised[140:162, 29:50] = classes['polyp']
supervised[20:35, 170:185] = classes['wall']
#supervised[0:0, 0:0] = classes['dirt']
supervised[0:10, 0:10] = classes['dark']
supervised[165:190, 80:100] = classes['hole']
#supervised[0:0, 0:0] = classes['bubbles']

y = supervised.ravel()
train = np.flatnonzero(supervised < n_classes)
test = np.flatnonzero(supervised == n_classes)

from sklearn.svm import SVC
clf = SVC()
clf.fit(X[train], y[train])
y[test] = clf.predict(X[test])
supervised = y.reshape(rows, cols)
#io.imshow(palette[supervised])
plt.imshow(img2)
plt.show()
plt.imshow(palette[supervised], cmap=plt.cm.binary)
plt.show()

supervised_001 = supervised
