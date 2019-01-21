import numpy as np
import random
import matplotlib.pyplot as plt
import os
import  imageio
from keras.models import load_model

from image_classifier_utils import prepare_input_data, split_input_data

def scale_image(image):
    max = np.max(image)
    min = np.abs(np.min(image))
    scaled = (image + min) / (max + min)
    return scaled.astype(np.int)

output_folder = './images/output/'

model = load_model('model')

input_data = prepare_input_data()
random.shuffle(input_data)

((train_images, train_labels), (test_images, test_labels)) = split_input_data(input_data)

# predictions = model.predict(test_images)

for (i, (input, ground_truth)) in enumerate(zip(test_images, test_labels)):

    output = model.predict(np.array([input]))[0]
    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(input)
    f.add_subplot(1, 3, 2)
    plt.imshow(ground_truth)
    f.add_subplot(1, 3, 3)
    plt.imshow(output)
    plt.savefig(os.path.join(output_folder, str(i) + '_fig.png'))
    imageio.imsave(os.path.join(output_folder,str(i) + '_out.png'),output)
    plt.close(f)
