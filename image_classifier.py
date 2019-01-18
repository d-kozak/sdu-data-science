import os
import random

import imageio
import keras
import matplotlib.pyplot as plt
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
                'image_type': np.zeros(shape=(223, 223, 3), dtype=np.uint8),  # np.zeros((227, 227, 3), dtype=np.uint8),
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


def build_neural_network():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(227, 227, 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(3, (3, 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.UpSampling2D(size=(4, 4)))
    model.add(keras.layers.Deconv2D(3, (4, 4)))
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

predictions = model.predict(train_images)

for i in range(len(predictions)):
    img_in = train_images[i]
    img_out = predictions[i]

    img_out = scale_image(img_out)

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(img_in)
    f.add_subplot(1, 2, 2)
    plt.imshow(img_out)
    plt.show(block=True)
