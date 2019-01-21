import os

import imageio
import numpy as np
import scipy

def load_images_from_folder(folder_name):
    return list(
        map(lambda image_name: (
            image_name, imageio.imread(os.path.join(folder_name, image_name)) / 255),
            os.listdir(folder_name)))


def prepare_input_data(database_folder='./images/database', ground_truth_folder='./images/ground_truth_augmented'):
    def remove_svm_from_name(input):
        name, data = input
        return name.replace('_SVM', ''), data

    output = []
    input_images = load_images_from_folder(database_folder)
    ground_truth = dict(map(remove_svm_from_name, load_images_from_folder(ground_truth_folder)))

    for (image_name, image_data) in input_images:
        image_output = ground_truth[image_name]

        image_output = scipy.misc.imresize(image_output, (110,110, 3)) / 255

        output.append(
            {
                'name': image_name,
                'output': image_output,
                'input': image_data
            }
        )
    return output


def split_input_data(input_data):
    images = [elem['input'] for elem in input_data]
    labels = [elem['output'] for elem in input_data]

    size = len(images)
    train_part = int(size * 0.7)

    train_images = np.array(images[:train_part])
    train_labels = np.array(labels[:train_part])

    test_images = np.array(images[train_part + 1:])
    test_labels = np.array(labels[train_part + 1:])
    return (train_images, train_labels), (test_images, test_labels)
