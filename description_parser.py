from scipy import misc

CLEAN = 0
DIRTY = 1


# picture size 257 * 257 * 3 == 154 587
class ImageData():
    def __init__(self, filename, type):
        self.type = type
        self.filename = filename
        self.data = misc.imread('images/' + filename) \
            .flatten()

    def __str__(self):
        return '(' + self.filename + ',' + str(self.type) + ')'

    def __repr__(self):
        return str(self)


def parseDescriptions(filename):
    desciptions = []
    with open(filename, 'r') as file:
        for line in file:
            (filename, type) = line.split()
            desciptions.append(ImageData(filename, CLEAN if type == 'clean' else DIRTY))
    return desciptions


desc = parseDescriptions('description_template.txt')
