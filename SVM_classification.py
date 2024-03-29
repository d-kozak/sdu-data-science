import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import imageio
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import os



# #1----------------------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/1.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[130:170, 22:55] = classes['polyp']
# supervised[20:95, 0:120] = classes['wall']
# #supervised[0:0, 0:0] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# supervised[165:195, 80:170] = classes['hole']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_1 = supervised
# imageio.imwrite('./images/1_SVM.jpg',palette[supervised])
# #2----------------------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/2.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[115:166, 20:52] = classes['polyp']
# supervised[10:110, 110:150] = classes['wall']
# supervised[125:170, 190:210] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# supervised[93:115, 53:80] = classes['hole']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_2 = supervised
# imageio.imwrite('./images/2_SVM.jpg',palette[supervised])
# # 3----------------------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/3.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[160:200, 145:190] = classes['polyp']
# supervised[00:130, 25:175] = classes['wall']
# #supervised[148:154, 109:146] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# #supervised[90:110, 65:75] = classes['hole']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_3 = supervised
# imageio.imwrite('./images/3_SVM.jpg',palette[supervised])
# #4------------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/4.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[10:50, 80:125] = classes['polyp']
# supervised[125:175, 70:175] = classes['wall']
# #supervised[148:154, 109:136] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# supervised[65:85, 85:120] = classes['hole']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_4 = supervised
# imageio.imwrite('./images/4_SVM.jpg',palette[supervised])
# #5------------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/5.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[165:195, 140:170] = classes['polyp']
# supervised[25:100, 150:165] = classes['wall']
# supervised[140:160, 169:182] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# supervised[65:90, 50:80] = classes['hole']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_5 = supervised
# imageio.imwrite('./images/5_SVM.jpg',palette[supervised])
# #6-----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/6.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:65, 25:175] = classes['wall']
# supervised[135:170, 123:150] = classes['polyp']
# supervised[150:180, 65:105] = classes['polyp']
# supervised[105:120, 148:170] = classes['dirt']
# supervised[130:160, 192:214] = classes['dirt']
# supervised[105:115, 81:90] = classes['dirt']
# supervised[125:165, 10:48] = classes['hole']
# supervised[0:10, 0:10] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_6 = supervised
# imageio.imwrite('./images/6_SVM.jpg',palette[supervised])
# #7-----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/7.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[100:170, 110:190] = classes['wall']
# supervised[110:198, 0:75] = classes['wall']
# #supervised[100:170, 110:190] = classes['wall']
#
# supervised[25:85, 0:52] = classes['polyp']
# supervised[5:25, 18:60] = classes['polyp']
# #supervised[110:125, 148:164] = classes['dirt']
# supervised[210:225, 125:170] = classes['hole']
# supervised[0:10, 0:10] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_7 = supervised
# imageio.imwrite('./images/7_SVM.jpg',palette[supervised])
# #8-----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/8.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[75:150, 115:185] = classes['wall']
# supervised[10:175, 150:200] = classes['wall']
# supervised[60:100, 20:72] = classes['polyp']
# #supervised[110:125, 148:164] = classes['dirt']
# supervised[110:139, 40:100] = classes['hole']
# supervised[0:10, 0:10] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_8 = supervised
# imageio.imwrite('./images/8_SVM.jpg',palette[supervised])
# #9-----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/9.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[100:140, 160:200] = classes['wall']
# supervised[148:185, 135:175] = classes['polyp']
# #supervised[110:125, 148:164] = classes['dirt']
# supervised[40:80, 40:82] = classes['hole']
# supervised[0:10, 0:10] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_9 = supervised
# imageio.imwrite('./images/9_SVM.jpg',palette[supervised])
# #10-----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/10.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[100:160, 0:100] = classes['wall']
# supervised[20:50, 125:169] = classes['polyp']
# #supervised[110:125, 148:164] = classes['dirt']
# #supervised[40:75, 40:80] = classes['hole']
# supervised[0:10, 0:10] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_10 = supervised
# imageio.imwrite('./images/10_SVM.jpg',palette[supervised])
#11----------------------------------------------------------------------------------------------
classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
n_classes = len(classes)
palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])

img = io.imread('images/11.jpg')
img2 = img
rows, cols, bands = img.shape


X = img.reshape(rows*cols, bands)
kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)

supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
supervised[40:190, 0:50] = classes['wall']
supervised[90:160, 65:102] = classes['polyp']
#supervised[110:125, 148:164] = classes['dirt']
supervised[100:175, 120:165] = classes['hole']
supervised[160:210, 60:150] = classes['hole']
supervised[0:10, 0:10] = classes['dark']

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
plt.imshow(palette[supervised])
plt.show()

supervised_11 = supervised
imageio.imwrite('./images/11_SVM.jpg',palette[supervised])
# #12----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/12.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[22:50, 0:20] = classes['wall']
# supervised[180:210, 175:190] = classes['polyp']
# supervised[180:205, 175:200] = classes['polyp']
# supervised[90:110, 83:98] = classes['dirt']
# supervised[40:108, 125:175] = classes['hole']
# supervised[50:53, 212:214] = classes['hole']
# supervised[0:10, 0:10] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_12 = supervised
# imageio.imwrite('./images/12_SVM.jpg',palette[supervised])
# #14----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/13.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[25:130, 175:200] = classes['wall']
# supervised[180:215, 80:150] = classes['wall']
# supervised[27:85, 0:145] = classes['polyp']
# supervised[50:100, 50:130] = classes['polyp']
# supervised[138:155, 70:100] = classes['dirt']
# supervised[125:170, 65:165] = classes['dirt']
# supervised[135:140, 75:80] = classes['dirt']
# supervised[160:172, 25:38] = classes['hole']
# supervised[0:10, 0:10] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_13 = supervised
# imageio.imwrite('./images/13_SVM.jpg',palette[supervised])
# #14----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/14.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[20:95, 85:180] = classes['wall']
# supervised[49:100, 0:10] = classes['wall']
# supervised[25:120, 40:80] = classes['wall']
# supervised[130:205, 28:113] = classes['polyp']
# #supervised[133:155, 90:100] = classes['dirt']
# supervised[135:185, 135:220] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_14 = supervised
# imageio.imwrite('./images/14_SVM.jpg',palette[supervised])
# #15----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/15.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[170:200, 150:185] = classes['wall']
# supervised[30:79, 20:60] = classes['wall']
# supervised[150:190, 18:92] = classes['polyp']
# #supervised[133:155, 90:100] = classes['dirt']
# supervised[90:125, 130:175] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_15 = supervised
# imageio.imwrite('./images/15_SVM.jpg',palette[supervised])
# #16----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/16.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[40:180, 0:100] = classes['wall']
# supervised[50:85, 105:145] = classes['polyp']
# supervised[60:90, 140:160] = classes['dirt']
# #supervised[90:125, 130:175] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_16 = supervised
# imageio.imwrite('./images/16_SVM.jpg',palette[supervised])
# #17----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('./images/17.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:24, 30:135] = classes['wall']
# supervised[110:185, 190:227] = classes['wall']
# supervised[115:160, 60:110] = classes['polyp']
# supervised[20:85, 150:195] = classes['dirt']
# supervised[37:60, 30:110] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_17 = supervised
# imageio.imwrite('./images/17_SVM.jpg',palette[supervised])
#
# #18----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('./images/18.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[205:227, 90:200] = classes['wall']
# supervised[165:205, 125:210] = classes['wall']
# supervised[110:195, 0:82] = classes['polyp']
# supervised[190:205, 12:35] = classes['polyp']
# supervised[105:149, 50:120] = classes['polyp']
# supervised[50:140, 130:185] = classes['dirt']
# supervised[40:75, 15:150] = classes['dirt']
# #supervised[40:45, 50:55] = classes['dirt']
# #supervised[50:60, 15:25] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_18 = supervised
# imageio.imwrite('./images/18_SVM.jpg',palette[supervised])
# #19----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/19.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[50:165, 0:100] = classes['wall']
# supervised[75:130, 155:215] = classes['polyp']
# supervised[60:80, 175:225] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# supervised[140:175, 140:155] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_19 = supervised
# imageio.imwrite('./images/19_SVM.jpg',palette[supervised])
#
# #20----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/20.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[70:180, 50:200] = classes['wall']
# supervised[0:60, 75:130] = classes['polyp']
# supervised[0:29, 65:145] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# #supervised[140:168, 140:155] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_20 = supervised
# imageio.imwrite('./images/20_SVM.jpg',palette[supervised])
# #21----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/21.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[20:50, 10:150] = classes['wall']
# supervised[100:150, 185:227] = classes['wall']
# supervised[50:100, 180:224] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# supervised[75:110, 110:145] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_21 = supervised
# imageio.imwrite('./images/21_SVM.jpg',palette[supervised])
# #22---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/22.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[50:150, 0:60] = classes['wall']
# supervised[190:227, 100:200] = classes['wall']
# supervised[25:50, 20:120] = classes['wall']
# supervised[53:123, 145:225] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# supervised[50:110, 115:135] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_22 = supervised
# imageio.imwrite('./images/22_SVM.jpg',palette[supervised])
# #23---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/23.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:60, 25:175] = classes['wall']
# supervised[0:100, 175:200] = classes['hole']
# supervised[105:162, 40:100] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# supervised[100:190, 0:25] = classes['hole']
# supervised[30:190, 180:225] = classes['hole']
# supervised[75:90, 165:205] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_23 = supervised
# imageio.imwrite('./images/23_SVM.jpg',palette[supervised])
# #24---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('./images/24.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:40, 25:135] = classes['wall']
# supervised[34:90, 35:110] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# supervised[100:190, 0:200] = classes['hole']
# supervised[100:190, 210:225] = classes['hole']
# supervised[175:1227, 35:190] = classes['hole']
# supervised[75:90, 165:205] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_24 = supervised
# imageio.imwrite('./images/24_SVM.jpg',palette[supervised])
# #25---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/25.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[25:120, 25:175] = classes['wall']
# supervised[85:225, 85:190] = classes['wall']
# supervised[145:225, 22:82] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# #supervised[100:190, 0:25] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_25 = supervised
# imageio.imwrite('./images/25_SVM.jpg',palette[supervised])
# #26---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/26.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:50, 50:100] = classes['wall']
# #supervised[160:200, 37:70] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# #supervised[100:190, 0:25] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_26 = supervised
# imageio.imwrite('./images/26_SVM.jpg',palette[supervised])
# #27---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/27.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[150:220, 10:150] = classes['wall']
# supervised[10:220, 95:130] = classes['wall']
# supervised[100:190, 150:222] = classes['polyp']
# supervised[60:125, 10:60] = classes['dirt']
# #supervised[100:190, 0:25] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_27 = supervised
# imageio.imwrite('./images/27_SVM.jpg',palette[supervised])
# #29---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/29.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[50:170, 0:100] = classes['wall']
# supervised[120:190, 80:220] = classes['wall']
# supervised[68:110, 105:185] = classes['polyp']
# #supervised[90:110, 30:60] = classes['dirt']
# #supervised[100:190, 0:25] = classes['hole']
# supervised[0:14, 0:15] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_29 = supervised
# imageio.imwrite('./images/29_SVM.jpg',palette[supervised])
# #30---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/30.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[40:190, 0:125] = classes['wall']
# supervised[40:120, 100:220] = classes['wall']
# supervised[145:200, 157:210] = classes['polyp']
# supervised[132:170, 132:170] = classes['polyp']
# #supervised[90:110, 30:60] = classes['dirt']
# #supervised[100:190, 0:25] = classes['hole']
# supervised[0:15, 0:15] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_30 = supervised
# imageio.imwrite('./images/30_SVM.jpg',palette[supervised])
# #31---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/31.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[50:150, 0:20] = classes['wall']
# supervised[85:150, 160:227] = classes['wall']
# supervised[58:76, 175:208] = classes['polyp']
# #supervised[90:110, 30:60] = classes['dirt']
# supervised[90:115, 102:120] = classes['hole']
# supervised[0:20, 0:20] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_31 = supervised
# imageio.imwrite('./images/31_SVM.jpg',palette[supervised])
# #32---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/32.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[155:227, 140:200] = classes['wall']
# supervised[40:125, 0:40] = classes['wall']
# supervised[140:195, 10:80] = classes['polyp']
# #supervised[90:110, 30:60] = classes['dirt']
# supervised[0:140, 80:165] = classes['hole']
# supervised[0:15, 0:15] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_32 = supervised
# imageio.imwrite('./images/32_SVM.jpg',palette[supervised])
# # #33---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/33.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[40:125, 0:227] = classes['wall']
# supervised[150:200, 150:200] = classes['wall']
# supervised[130:195, 70:112] = classes['polyp']
# #supervised[90:110, 30:60] = classes['dirt']
# #supervised[75:115, 60:120] = classes['hole']
# supervised[0:15, 0:15] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_33 = supervised
# imageio.imwrite('./images/33_SVM.jpg',palette[supervised])
# #34-------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/34.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[20:80, 140:220] = classes['wall']
# supervised[80:115, 175:210] = classes['polyp']
# supervised[80:125, 10:69] = classes['dirt']
# supervised[150:200, 0:20] = classes['hole']
# supervised[200:220, 75:102] = classes['hole']
# supervised[0:15, 0:15] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_34 = supervised
# imageio.imwrite('./images/34_SVM.jpg',palette[supervised])
# #35--------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/35.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[125:200, 0:125] = classes['wall']
# supervised[205:227, 50:168] = classes['wall']
# supervised[140:190, 120:227] = classes['wall']
# supervised[20:85, 0:85] = classes['polyp']
# supervised[0:30, 24:92] = classes['polyp']
# supervised[35:45, 125:150] = classes['dirt']
# supervised[25:35, 180:190] = classes['dirt']
# supervised[195:210, 175:180] = classes['hole']
# supervised[0:15, 0:15] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_35 = supervised
# imageio.imwrite('./images/35_SVM.jpg',palette[supervised])
# #36---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/36.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:25, 50:155] = classes['wall']
# supervised[194:225, 160:200] = classes['polyp']
# supervised[90:115, 85:115] = classes['dirt']
# #supervised[195:210, 175:180] = classes['hole']
# supervised[0:15, 0:15] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_36 = supervised
# imageio.imwrite('./images/36_SVM.jpg',palette[supervised])
# #37---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/37.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[50:150, 0:100] = classes['wall']
# supervised[10:50, 50:165] = classes['wall']
# supervised[150:227, 25:80] = classes['wall']
# supervised[195:223, 150:188] = classes['polyp']
# #supervised[90:115, 85:115] = classes['dirt']
# #supervised[195:210, 175:180] = classes['hole']
# supervised[0:15, 0:15] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_37 = supervised
# imageio.imwrite('./images/37_SVM.jpg',palette[supervised])
# #38---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/38.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[75:120, 50:87] = classes['polyp']
# supervised[60:125, 45:98] = classes['polyp']
# supervised[0:227, 115:175] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# #supervised[195:210, 175:180] = classes['hole']
# supervised[0:10, 0:10] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_38 = supervised
# imageio.imwrite('./images/38_SVM.jpg',palette[supervised])
# #39---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/39.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# #supervised[75:120, 50:87] = classes['polyp']
# supervised[70:150, 130:165] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# #supervised[195:210, 175:180] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_39 = supervised
# imageio.imwrite('./images/39_SVM.jpg',palette[supervised])
# #40---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/40.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# #supervised[75:120, 50:87] = classes['polyp']
# supervised[70:150, 130:165] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# supervised[110:175, 0:45] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_40 = supervised
# imageio.imwrite('./images/40_SVM.jpg',palette[supervised])
# #41---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/41.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[28:90, 115:150] = classes['polyp']
# supervised[0:65, 130:201] = classes['polyp']
# supervised[100:179, 0:227] = classes['wall']
# supervised[150:227, 50:150] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# #supervised[110:175, 0:45] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_41 = supervised
# imageio.imwrite('./images/41_SVM.jpg',palette[supervised])
# #42---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/42.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[120:170, 110:210] = classes['polyp']
# supervised[15:110, 150:210] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# supervised[40:180, 0:80] = classes['hole']
# supervised[175:227, 30:160] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_42 = supervised
# imageio.imwrite('./images/42_SVM.jpg',palette[supervised])
# #43---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/43.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:105, 150:195] = classes['polyp']
# supervised[75:120, 125:165] = classes['polyp']
# supervised[10:70, 170:215] = classes['polyp']
# supervised[62:75, 135:155] = classes['polyp']
# supervised[15:75, 0:130] = classes['wall']
# supervised[50:210, 0:50] = classes['wall']
# supervised[145:227, 145:200] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# supervised[140:175, 75:130] = classes['hole']
# supervised[105:130, 95:118] = classes['hole']
#
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_43 = supervised
# imageio.imwrite('./images/43_SVM.jpg',palette[supervised])
# #44---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/44.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[105:160, 75:160] = classes['polyp']
# supervised[200:227, 30:190] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# supervised[25:80, 30:190] = classes['hole']
# supervised[75:190, 15:65] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_44 = supervised
# imageio.imwrite('./images/44_SVM.jpg',palette[supervised])
# #45---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/45.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[135:202, 85:175] = classes['polyp']
# supervised[25:120, 0:200] = classes['wall']
# supervised[80:145, 180:225] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# supervised[175:210, 40:80] = classes['hole']
# supervised[210:227, 46:180] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_45 = supervised
# imageio.imwrite('./images/45_SVM.jpg',palette[supervised])
# #46---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/46.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[150:190, 65:110] = classes['polyp']
# supervised[140:190, 145:225] = classes['wall']
# supervised[115:140, 110:130] = classes['dirt']
# supervised[100:125, 55:65] = classes['dirt']
# supervised[25:50, 35:150] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_46 = supervised
# imageio.imwrite('./images/46_SVM.jpg',palette[supervised])
# #47---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/47.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[155:200, 90:132] = classes['polyp']
# supervised[5:120, 100:190] = classes['wall']
# supervised[0:20, 55:100] = classes['wall']
# supervised[90:160, 180:225] = classes['wall']
# supervised[100:160, 0:60] = classes['dirt']
# #supervised[25:50, 35:150] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_47 = supervised
# imageio.imwrite('./images/47_SVM.jpg',palette[supervised])
# #48---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/48.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[205:227, 110:135] = classes['polyp']
# supervised[0:150, 100:200] = classes['wall']
# supervised[150:190, 140:210] = classes['wall']
# supervised[45:140, 0:45] = classes['dirt']
# #supervised[25:50, 35:150] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_48 = supervised
# imageio.imwrite('./images/48_SVM.jpg',palette[supervised])
# #49---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/49.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[80:115, 115:160] = classes['polyp']
# supervised[140:200, 100:175] = classes['wall']
# supervised[95:115, 77:95] = classes['dirt']
# #supervised[25:50, 35:150] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_49 = supervised
# imageio.imwrite('./images/49_SVM.jpg',palette[supervised])
# #50---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/50.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[165:180, 40:70] = classes['polyp']
# supervised[30:100, 0:120] = classes['wall']
# #supervised[95:115, 77:95] = classes['dirt']
# supervised[145:195, 80:125] = classes['hole']
# supervised[10:75, 165:190] = classes['hole']
# supervised[125:140, 140:150] = classes['hole']
# supervised[0:12, 0:12] = classes['dark']
#
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
#
# clf = SVC()
# clf.fit(X[train], y[train])
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# #io.imshow(palette[supervised])
# plt.imshow(img2)
# plt.show()
# plt.imshow(palette[supervised], cmap=plt.cm.binary)
# plt.show()
#
# supervised_50 = supervised
# imageio.imwrite('./images/50_SVM.jpg',palette[supervised])
#
#os.system("ground_truth_generator.py")

print("end")
