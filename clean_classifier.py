import os

import imageio
import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten
import numpy as np

from description_parser import parseDescriptions, file_prefix_from_file_name
from utils import run_neural_network

database_folder = './images/database'
database_images = os.listdir(database_folder)


def prepare_input_data():
    """
    Loads data from description_template.txt file, which contains labels for the images, and the associated images described in the file.
    :return: List of dictionaries[name=name of file,image_type==category(0 for clean, 1 for dirty),data==data]
    """
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
    """
    Builds the neural network using keras sequential API.
    """
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(227, 227, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation = 'softmax'))
    return model

def evaluate_model(model, test_images, test_labels, train_images, train_labels):
    """
    Compiles, fits and trains the network
    :param model: keras model
    :param test_images: list of test images
    :param test_labels: list of test labels
    :param train_images: list of train images
    :param train_labels: list of train labels
    :return: tuple(float,float) loss,accuracy
    """
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=8)
    return model.evaluate(test_images, test_labels)


def main():
    """
    main function
    """
    run_neural_network(prepare_input_data, build_neural_network, evaluate_model)


if __name__ == '__main__':
    main()
