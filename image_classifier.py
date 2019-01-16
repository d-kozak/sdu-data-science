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
                'image_type': np.zeros((227, 227, 3), dtype=np.uint8),
                'data': image
            }
        )
    return output


def split_input_data(input_data):
    for elem in input_data:
        assert len(elem['data']) == len(elem['image_type'])
        assert len(elem['data'][0]) == len(elem['image_type'][0])
        assert len(elem['data'][0][0]) == len(elem['image_type'][0][0])

    images = [elem['data'] for elem in input_data]
    labels = [elem['image_type'] for elem in input_data]

    size = len(images)
    train_part = int(size * 0.7)

    train_images = np.array(images[:train_part])
    train_labels = np.array(labels[:train_part])

    test_images = np.array(images[train_part + 1:])
    test_labels = np.array(labels[train_part + 1:])
    return (train_images, train_labels), (test_images, test_labels)


def build_neural_network():
    model = keras.Sequential()

    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Reshape(target_shape=(227, 227, 3)))
    # model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(227, 227, 3)))
    model.add(keras.layers.Convolution2D(32, (3, 3), input_shape=(227, 227, 3)))
    model.add(keras.layers.Reshape(target_shape=(227, 227, 3)))

    return model


def evaluate_model(model, test_images, test_labels, train_images, train_labels):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    return model.evaluate(test_images, test_labels)


input_data = prepare_input_data()
random.shuffle(input_data)

((train_images, train_labels), (test_images, test_labels)) = split_input_data(input_data)

model = build_neural_network()

val_loss, val_acc = evaluate_model(model, test_images, test_labels, train_images, train_labels)
print('Loss value ' + str(val_loss))
print('Accuracy ' + str(val_acc))
