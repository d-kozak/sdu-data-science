import os

import imageio
import numpy as np

from utils import is_in_range

input_folder = './images/'
output_folder = './images/ground_truth_new'

try:
    os.mkdir(output_folder)
except FileExistsError:
    pass

output_polyp = np.array([255, 0, 0])
output_wall = np.array([0, 0, 255])
output_else = np.array([0, 255, 0])

polyp_low = np.array([200, 82, 48])
polyp_high = np.array([255, 142, 108])

wall_low = np.array([200, 134, 98])
wall_high = np.array([255, 194, 158])


def process_cell(cell):
    def change(cell, input):
        for i in range(3):
            cell[i] = input[i]

    if is_in_range(cell, polyp_low, polyp_high):
        change(cell, output_polyp)
    elif is_in_range(cell, wall_low, wall_high):
        change(cell, output_wall)
    else:
        change(cell, output_else)


def change_colors(image):
    for row in image:
        for cell in row:
            process_cell(cell)
    return image


images = list(filter(lambda name: '_SVM' in name, os.listdir(input_folder)))
for index, image_name in enumerate(images):
    print('%s/%s %s ' % (str(index), str(len(images)), image_name))
    image = imageio.imread(os.path.join(input_folder, image_name))
    changed = change_colors(image)
    image_name = image_name.split('.')[0] + '.png'
    imageio.imsave(os.path.join(output_folder, image_name), changed)
