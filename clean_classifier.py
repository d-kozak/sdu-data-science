import os

import imageio
import keras
import numpy as np

from description_parser import parseDescriptions

database_folder = './images/database'
database_images = os.listdir(database_folder)

descriptions = parseDescriptions('description_template.txt')

output = []
for description in descriptions:
    filename = description.filename
    image_type = description.type
    file_prefix = filename.split('.')[0] + "_"
    images = list(filter(lambda name: name.startswith(file_prefix) or name == filename, database_images))
    print(file_prefix)
    print(images)
    print("----")
    for image in images:
        data = imageio.imread(database_folder + '/' + image)
        output.append((image, image_type, data))

print(output)

images = [x[2] for x in output]
labels = [x[1] for x in output]

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
