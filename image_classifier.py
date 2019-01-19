import os
import random

import imageio
import keras
import matplotlib.pyplot as plt
import numpy as np

database_folder = './images/database'
ground_truth_folder = './images/ground_truth'


def prepare_input_data():
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


def load_images_from_folder(folder_name):
    #image_name, keras.utils.normalize(imageio.imread(os.path.join(folder_name, image_name)), axis=1)),
    return list(
        map(lambda image_name: (
            image_name, imageio.imread(os.path.join(folder_name, image_name))),
            os.listdir(folder_name)))


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


def build_neural_network():
    model = keras.Sequential()
    model.add(keras.layers.ZeroPadding2D(padding=(1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.ZeroPadding2D(padding=(1, 1)))
    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.ZeroPadding2D(padding=(1, 1)))
    model.add(keras.layers.Conv2D(16, (3, 3)))
    model.add(keras.layers.ZeroPadding2D(padding=(1, 1)))
    model.add(keras.layers.Conv2D(3, (3, 3)))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Conv2D(3, (3, 3)))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.UpSampling2D(size=(4, 4)))
    # model.add(keras.layers.Deconv2D(3, (8, 8)))
    return model


def evaluate_model(model, test_images, test_labels, train_images, train_labels):
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=3)
    return model.evaluate(test_images, test_labels)


def scale_image(image):
    max = np.max(image)
    min = np.abs(np.min(image))
    return (image + min) / (max + min)


input_data = prepare_input_data()
random.shuffle(input_data)

((train_images, train_labels), (test_images, test_labels)) = split_input_data(input_data)

model = build_neural_network()

val_loss, val_acc = evaluate_model(model, test_images, test_labels, train_images, train_labels)
print('Loss value ' + str(val_loss))
print('Accuracy ' + str(val_acc))

model.save('model')

predictions = model.predict(train_images)

for i in range(len(predictions)):
    img_in = train_images[i]
    img_label = train_labels[i]

    img_out = predictions[i]
    img_out = scale_image(img_out)

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(img_in)
    f.add_subplot(1, 3, 2)
    plt.imshow(img_label)
    f.add_subplot(1, 3, 3)
    plt.imshow(img_out)
    plt.show(block=True)
