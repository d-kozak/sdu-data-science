import os
import random

import imageio
import keras
import numpy as np

database_folder = './images/database'


def prepare_input_data():
    output = []
    images = list(
        map(lambda image_name: imageio.imread(os.path.join(database_folder, image_name)), os.listdir(database_folder)))

    for image in images:
        output.append(
            {
                'name': 'foo',
                'image_type': np.zeros(shape=(223, 223, 1), dtype=np.uint8),  # np.zeros((227, 227, 3), dtype=np.uint8),
                'data': image
            }
        )
    return output


def split_input_data(input_data):
    images = [elem['data'] for elem in input_data]
    labels = [elem['image_type'] for elem in input_data]

    size = len(images)
    train_part = int(size * 0.7)

    train_images = np.array(images[:train_part])
    train_labels = np.array(labels[:train_part])

    test_images = np.array(images[train_part + 1:])
    test_labels = np.array(labels[train_part + 1:])
    return (train_images, train_labels), (test_images, test_labels)


tmp = keras.layers.Dense(1)


def build_neural_network():
    global tmp
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(227, 227, 3)))
    model.add(keras.layers.Conv2D(8, (3, 3)))
    model.add(keras.layers.Dense(10))
    model.add(tmp)
    # model.add(keras.layers.Convolution2D(32, (3, 3), input_shape=(227, 227, 3)))
    # model.add(keras.layers.Flatten())

    # model.add(keras.layers.Reshape(target_shape=(227, 227, 3)))

    return model


def evaluate_model(model, test_images, test_labels, train_images, train_labels):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('shape: ')
    print(tmp.output_shape)
    model.fit(train_images, train_labels, epochs=10)
    return model.evaluate(test_images, test_labels)


input_data = prepare_input_data()
random.shuffle(input_data)

((train_images, train_labels), (test_images, test_labels)) = split_input_data(input_data)

model = build_neural_network()

val_loss, val_acc = evaluate_model(model, test_images, test_labels, train_images, train_labels)
print('Loss value ' + str(val_loss))
print('Accuracy ' + str(val_acc))
