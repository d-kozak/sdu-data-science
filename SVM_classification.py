import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import imageio
from sklearn.cluster import KMeans
from sklearn.svm import SVC

plt.close('all')
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
# supervised[140:162, 29:50] = classes['polyp']
# supervised[20:35, 170:185] = classes['wall']
# #supervised[0:0, 0:0] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# supervised[165:180, 80:100] = classes['hole']
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
# supervised[125:140, 25:50] = classes['polyp']
# supervised[90:110, 120:140] = classes['wall']
# supervised[140:160, 190:195] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# supervised[90:110, 65:75] = classes['hole']
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
#3----------------------------------------------------------------------------------------------------------
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
# supervised[160:185, 150:170] = classes['polyp']
# supervised[10:30, 40:150] = classes['wall']
# supervised[148:154, 109:136] = classes['dirt']
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
#4------------------------------------------------------------------------------------------------
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
# supervised[25:45, 85:115] = classes['polyp']
# supervised[100:120, 0:35] = classes['wall']
# #supervised[148:154, 109:136] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# supervised[75:85, 100:120] = classes['hole']
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
# supervised[170:185, 140:153] = classes['polyp']
# supervised[25:50, 150:165] = classes['wall']
# #supervised[148:154, 109:136] = classes['dirt']
# supervised[0:10, 0:10] = classes['dark']
# supervised[62:70, 50:65] = classes['hole']
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
# supervised[10:20, 50:150] = classes['wall']
# supervised[135:155, 125:145] = classes['polyp']
# supervised[110:125, 148:164] = classes['dirt']
# supervised[150:165, 10:48] = classes['hole']
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
# supervised[100:180, 110:190] = classes['wall']
# supervised[25:70, 0:35] = classes['polyp']
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
# supervised[75:125, 75:175] = classes['wall']
# supervised[65:90, 25:60] = classes['polyp']
# #supervised[110:125, 148:164] = classes['dirt']
# supervised[110:139, 40:95] = classes['hole']
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
# supervised[150:175, 140:175] = classes['polyp']
# #supervised[110:125, 148:164] = classes['dirt']
# supervised[40:75, 40:80] = classes['hole']
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
# supervised[20:50, 125:165] = classes['polyp']
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
# #11----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/11.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[40:100, 0:50] = classes['wall']
# supervised[90:150, 75:90] = classes['polyp']
# #supervised[110:125, 148:164] = classes['dirt']
# supervised[100:175, 120:165] = classes['hole']
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
# supervised_11 = supervised
# imageio.imwrite('./images/11_SVM.jpg',palette[supervised])
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
# supervised[75:130, 180:221] = classes['wall']
# supervised[30:70, 35:75] = classes['polyp']
# supervised[133:155, 90:100] = classes['dirt']
# supervised[125:133, 112:120] = classes['dirt']
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
# supervised[50:60, 20:30] = classes['wall']
# supervised[20:60, 125:175] = classes['wall']
# supervised[140:175, 50:95] = classes['polyp']
# #supervised[133:155, 90:100] = classes['dirt']
# supervised[160:185, 140:175] = classes['hole']
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
# supervised[30:75, 20:60] = classes['wall']
# supervised[150:190, 40:92] = classes['polyp']
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
# supervised[25:70, 25:75] = classes['wall']
# supervised[75:85, 115:130] = classes['polyp']
# supervised[65:85, 145:160] = classes['dirt']
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
# img = io.imread('images/17.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:20, 100:135] = classes['wall']
# supervised[110:155, 70:81] = classes['polyp']
# supervised[60:90, 150:165] = classes['dirt']
# supervised[40:60, 30:60] = classes['hole']
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
# #18----------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/18.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[190:225, 125:160] = classes['wall']
# supervised[120:145, 0:40] = classes['polyp']
# supervised[60:90, 150:160] = classes['dirt']
# supervised[40:45, 50:55] = classes['dirt']
# supervised[50:60, 15:25] = classes['hole']
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
# supervised[145:180, 0:40] = classes['wall']
# supervised[60:91, 175:200] = classes['polyp']
# supervised[103:107, 12:18] = classes['dirt']
# supervised[140:168, 140:155] = classes['hole']
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
# supervised[100:180, 50:150] = classes['wall']
# supervised[0:50, 80:130] = classes['polyp']
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
# supervised[25:50, 40:150] = classes['wall']
# supervised[60:80, 190:220] = classes['polyp']
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
# supervised[75:115, 165:210] = classes['polyp']
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
# supervised[0:50, 50:100] = classes['wall']
# supervised[115:130, 50:90] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# supervised[100:190, 0:25] = classes['hole']
# supervised[100:190, 210:225] = classes['hole']
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
# img = io.imread('images/24.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:50, 50:100] = classes['wall']
# supervised[60:90, 50:90] = classes['polyp']
# #supervised[103:107, 12:18] = classes['dirt']
# supervised[100:190, 0:25] = classes['hole']
# supervised[100:190, 210:225] = classes['hole']
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
# supervised[0:50, 50:100] = classes['wall']
# supervised[160:200, 37:70] = classes['polyp']
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
# supervised[80:140, 90:130] = classes['wall']
# supervised[115:150, 175:200] = classes['polyp']
# supervised[90:110, 30:60] = classes['dirt']
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
# #28---------------------------------------------------------------------------------------------
# classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
# n_classes = len(classes)
# palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])
#
# img = io.imread('images/28.jpg')
# img2 = img
# rows, cols, bands = img.shape
#
#
# X = img.reshape(rows*cols, bands)
# kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
#
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[50:100, 0:50] = classes['wall']
# supervised[175:210, 160:200] = classes['polyp']
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
# supervised_28 = supervised
# imageio.imwrite('./images/28_SVM.jpg',palette[supervised])
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
# supervised[50:150, 0:50] = classes['wall']
# supervised[70:100, 126:165] = classes['polyp']
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
# supervised[50:150, 0:50] = classes['wall']
# supervised[155:190, 160:178] = classes['polyp']
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
# supervised[60:74, 185:195] = classes['polyp']
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
# supervised[175:220, 155:200] = classes['wall']
# supervised[150:180, 30:70] = classes['polyp']
# #supervised[90:110, 30:60] = classes['dirt']
# supervised[75:115, 60:120] = classes['hole']
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
# supervised[50:100, 75:125] = classes['wall']
# supervised[150:200, 150:200] = classes['wall']
# supervised[140:190, 90:110] = classes['polyp']
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
# supervised[40:60, 140:210] = classes['wall']
# supervised[90:110, 180:205] = classes['polyp']
# supervised[165:173, 145:150] = classes['dirt']
# supervised[85:102, 35:65] = classes['dirt']
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
# supervised[150:180, 60:125] = classes['wall']
# supervised[20:60, 15:50] = classes['polyp']
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
# supervised[190:220, 160:195] = classes['polyp']
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
# supervised[195:220, 150:185] = classes['polyp']
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
# supervised[70:150, 130:165] = classes['wall']
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
# supervised[10:65, 130:200] = classes['polyp']
# supervised[70:150, 130:165] = classes['wall']
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
# supervised[125:165, 110:165] = classes['polyp']
# supervised[15:60, 20:210] = classes['wall']
# #supervised[90:115, 85:115] = classes['dirt']
# supervised[160:205, 25:100] = classes['hole']
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
# #supervised[125:165, 110:165] = classes['polyp']
# supervised[20:60, 10:100] = classes['wall']
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
# supervised[125:160, 80:128] = classes['polyp']
# supervised[215:227, 50:190] = classes['wall']
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
classes = {'wall': 0, 'polyp': 1, 'dirt': 2, 'dark': 3, 'hole': 4}
n_classes = len(classes)
palette = np.uint8([[234, 164, 128], [234, 112, 78], [234, 207, 79], [0, 0, 0], [67,52,31]])

img = io.imread('images/45.jpg')
img2 = img
rows, cols, bands = img.shape


X = img.reshape(rows*cols, bands)
kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)

supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
supervised[140:160, 110:130] = classes['polyp']
supervised[50:140, 0:75] = classes['wall']
#supervised[90:115, 85:115] = classes['dirt']
#supervised[25:80, 30:190] = classes['hole']
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

supervised_45 = supervised
imageio.imwrite('./images/45_SVM.jpg',palette[supervised])
#46---------------------------------------------------------------------------------------------