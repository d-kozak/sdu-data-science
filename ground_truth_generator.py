import os, os.path
from PIL import Image, ImageOps

# This script performs image augmentation for input images found in ./images folder. It does not augment the SVM images
# It creates rotated and flipped versions of the original image.

images_directory = './images/ground_truth_new'
images_names_array = [name for name in os.listdir(images_directory)
                      if os.path.isfile(os.path.join(images_directory, name)) and
                      len(name.split('_')) == 2]

print('\nThere are ' + str(len(images_names_array)) + ' images as a source for ground-truth.\n')

# check (and create if necessary) output folder for ground-truth images
output_directory = "./images/ground_truth_augmented"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for image_name in images_names_array:
    name_without_extension, extension = image_name.split('.')
    image = Image.open(os.path.join(images_directory, image_name))
    extension = 'png'
    image.save(output_directory + '/' + name_without_extension + '.' + extension)

    flipped_image = ImageOps.mirror(image)
    flipped_image.save(output_directory + '/' + name_without_extension + '_flipped.' + extension)

    for angle in range(90, 271, 90):
        rotated_image = image.rotate(angle)
        rotated_image.save(output_directory + '/' + name_without_extension + '_' + str(angle) + '.' + extension)

        flipped_and_rotated_image = ImageOps.mirror(rotated_image)
        flipped_and_rotated_image.save(output_directory + '/' + name_without_extension + '_' + str(angle) + '_flipped.' + extension)

    image.close()
