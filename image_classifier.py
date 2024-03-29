import os
import random

import keras
import matplotlib.pyplot as plt

from image_classifier_utils import prepare_input_data, split_input_data


def build_neural_network():
    """
    Builds the neural network using keras sequential API.
    """
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(4, (3, 3), activation='sigmoid', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(50))
    model.add(keras.layers.Dense(50))
    model.add(keras.layers.Dense(50))

    model.add(keras.layers.Dense(36300))
    model.add(keras.layers.Reshape(target_shape=(110, 110, 3)))
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
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error, metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=100)
    return model.evaluate(test_images, test_labels)


input_data = prepare_input_data()
random.shuffle(input_data)

((train_images, train_labels), (test_images, test_labels)) = split_input_data(input_data)

model = build_neural_network()

val_loss, val_acc = evaluate_model(model, test_images, test_labels, train_images, train_labels)
print('Loss value ' + str(val_loss))
print('Accuracy ' + str(val_acc))

model.summary()
model.save('model')

predictions = model.predict(test_images)

output_folder = './images/output/'

try:
    os.mkdir(output_folder)
except FileExistsError:
    pass

for i in range(len(predictions)):
    img_in = test_images[i]
    img_label = test_labels[i]

    img_out = predictions[i]

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(img_in)
    f.add_subplot(1, 3, 2)
    plt.imshow(img_label)
    f.add_subplot(1, 3, 3)
    plt.imshow(img_out)

    plt.savefig(os.path.join(output_folder, str(i) + '.png'))
    plt.close(f)
