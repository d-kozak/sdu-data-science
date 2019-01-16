import os

import imageio
import keras
import numpy as np

import random

from description_parser import parseDescriptions, file_prefix_from_file_name

database_folder = './images/database'
database_images = os.listdir(database_folder)

descriptions = parseDescriptions('description_template.txt')

input_data = []

for description in descriptions:
    filename, image_type = description
    file_prefix = file_prefix_from_file_name(filename)
    image_names = list(filter(lambda name: name.startswith(file_prefix) or name == filename, database_images))
    for image_name in image_names:
        data = imageio.imread(os.path.join(database_folder, image_name))
        input_data.append(
            {
                'name': image_name,
                'image_type': image_type,
                'data': data
            }
        )

random.shuffle(input_data)

images = [elem['data'] for elem in input_data]
labels = [elem['image_type'] for elem in input_data]

size = len(images)
train_part = int(size * 0.7)

train_images = np.array(images[:train_part])
train_labels = np.array(labels[:train_part])

test_images = np.array(images[train_part + 1:])
test_labels = np.array(labels[train_part + 1:])

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
