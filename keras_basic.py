import keras
import matplotlib.pyplot as plt
import numpy as np

# a simple data set of hand written digits
mnist = keras.datasets.mnist

# split it into training and test
# data are labeled
(input_train, labels_train), (input_test, labels_test) = mnist.load_data()

# normalize input to get values from 0 to 1
input_train = keras.utils.normalize(input_train, axis=1)
input_test = keras.utils.normalize(input_test, axis=1)

# build the neural net
model = keras.Sequential()

# flatten 2D image array into 1D array 
model.add(keras.layers.Flatten())

# core of the nn
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))

# 10 values, because we want to classify 10 categories
# softmax to get probabilities as output
model.add(keras.layers.Dense(10, activation='softmax'))

# prepare the nn
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train it using labeled training data
model.fit(input_train, labels_train, epochs=3)

# evaluate it on testing data
val_loss, val_acc = model.evaluate(input_test, labels_test)
print(val_loss)
print(val_acc)

# just persist and load :)
model.save('nums.model')
new_model = keras.models.load_model('nums.model')

# make predictions, input is a list
predictions = new_model.predict(input_test)

# check out the first number
first_num = input_test[0]
first_prediction = predictions[0]
print(first_prediction)  # list of probabilities
guess = np.argmax(first_prediction)  # to get the index with highest probability

# what is the number?
print(guess)

# plot it :)
plt.imshow(first_num, cmap=plt.cm.binary)
plt.show()
