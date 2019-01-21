import os

import imageio


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
        if image_output is None:
            raise RuntimeError('Could not find image ' + image_name)

        output.append(
            {
                'name': image_name,
                'output': image_output,
                'input': image_data
            }
        )
    return output
