import keras
import numpy as np

import sys

from description_parser import parseDescriptions

desc = parseDescriptions('description_template.txt')

images = [item.data for item in desc]
labels = [item.type for item in desc]

train_images = np.array(images[:40])
train_labels = np.array(labels[:40])

test_images = np.array(images[41:])
test_labels = np.array(labels[41:])

model = keras.Sequential()

# todo no matter how many layers are added, the acc is always around 0.666... why?
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(4, activation='relu'))
model.add(keras.layers.Dense(2, activation='relu'))

model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train it using labeled training data
model.fit(train_images, train_labels, epochs=10)

val_loss, val_acc = model.evaluate(test_images, test_labels)
print('Loss value ' + str(val_loss))
print('Accuracy ' + str(val_acc))
