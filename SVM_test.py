import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import imageio
from sklearn.cluster import KMeans
from sklearn.svm import SVC

##18----------------------------------------------------------------------------------------------
classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
n_classes = len(classes)
palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])

img = io.imread('./images/18.jpg')
img2 = img
rows, cols, bands = img.shape


X = img.reshape(rows*cols, bands)
kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)

supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
supervised[205:227, 90:200] = classes['wall']
supervised[165:205, 125:210] = classes['wall']
supervised[110:195, 0:82] = classes['polyp']
supervised[190:205, 12:35] = classes['polyp']
supervised[105:149, 50:120] = classes['polyp']
supervised[50:140, 130:185] = classes['dirt']
supervised[40:75, 15:150] = classes['dirt']
#supervised[40:45, 50:55] = classes['dirt']
#supervised[50:60, 15:25] = classes['hole']
supervised[0:12, 0:12] = classes['dark']

y = supervised.ravel()
train = np.flatnonzero(supervised < n_classes)
test = np.flatnonzero(supervised == n_classes)

clf = SVC()
clf.fit(X[train], y[train])
y[test] = clf.predict(X[test])
supervised = y.reshape(rows, cols)
#io.imshow(palette[supervised])
plt.imshow(img2)
plt.show()
plt.imshow(palette[supervised], cmap=plt.cm.binary)
plt.show()

supervised_18 = supervised
imageio.imwrite('./images/18_SVM.jpg',palette[supervised])