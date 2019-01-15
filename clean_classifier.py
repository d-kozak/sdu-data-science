import keras
import numpy as np

from description_parser import parseDescriptions

desc = parseDescriptions('description_template.txt')

images = [item.data for item in desc]
labels = [item.type for item in desc]

train_images = np.array(images[:30])
train_labels = np.array(labels[:30])

test_images = np.array(images[31:])
test_labels = np.array(labels[31:])

model = keras.Sequential()

# todo no matter how many layers are added, the acc is always around 0.666... why?
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train it using labeled training data
model.fit(train_images, train_labels, epochs=10)

val_loss, val_acc = model.evaluate(test_images, test_labels)
print(val_loss)
print(val_acc)
