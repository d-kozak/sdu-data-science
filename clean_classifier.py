import os

import imageio
import keras
import numpy as np

from description_parser import parseDescriptions, file_prefix_from_file_name
from utils import run_neural_network

database_folder = './images/database'
database_images = os.listdir(database_folder)


def prepare_input_data():
    output = []
    for description in parseDescriptions('description_template.txt'):
        filename, image_type = description
        file_prefix = file_prefix_from_file_name(filename)
        image_names = list(filter(lambda name: name.startswith(file_prefix) or name == filename, database_images))
        for image_name in image_names:
            data = imageio.imread(os.path.join(database_folder, image_name))
            output.append(
                {
                    'name': image_name,
                    'image_type': image_type,
                    'data': data
                }
            )
    return output


def build_neural_network():
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.Dense(12, activation='relu'))
    model.add(keras.layers.Dense(12, activation='tanh'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    return model


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


def evaluate_model(model, test_images, test_labels, train_images, train_labels):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)
    return model.evaluate(test_images, test_labels)


def main():
    run_neural_network(prepare_input_data, split_input_data, build_neural_network, evaluate_model)


if __name__ == '__main__':
    main()
