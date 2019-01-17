import os, os.path
from PIL import Image, ImageOps

images_directory = './images'
images_names_array = [name for name in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, name))]

print('\nThere are ' + str(len(images_names_array)) + ' images as a source for database.\n')

# check (and create if necessary) output folder for database images
output_directory = "./images/database"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for image_name in images_names_array:
    name_without_extension, extension = image_name.split('.')
    image = Image.open(os.path.join(images_directory, image_name))

    image.save(output_directory + '/' + name_without_extension + '.' + extension)

    flipped_image = ImageOps.mirror(image)
    flipped_image.save(output_directory + '/' + name_without_extension + '_flipped.' + extension)

    for angle in range(90, 271, 90):
        rotated_image = image.rotate(angle)
        rotated_image.save(output_directory + '/' + name_without_extension + '_' + str(angle) + '.' + extension)

        flipped_and_rotated_image = ImageOps.mirror(rotated_image)
        flipped_and_rotated_image.save(output_directory + '/' + name_without_extension + '_' + str(angle) + '_flipped.' + extension)

    image.close()
