CLEAN = 0
DIRTY = 1


# picture size 227 * 227 * 3 == 154 587
class ImageData():
    def __init__(self, filename, type):
        self.type = type
        self.filename = filename

    def __str__(self):
        return '(' + self.filename + ',' + str(self.type) + ')'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        for item in [self.filename, self.type]:
            yield item


def file_prefix_from_file_name(filename):
    return filename.split('.')[0] + "_"

def parseDescriptions(filename):
    desciptions = []
    with open(filename, 'r') as file:
        for line in file:
            (filename, type) = line.split()
            desciptions.append(ImageData(filename, CLEAN if type == 'clean' else DIRTY))
    return desciptions


desc = parseDescriptions('description_template.txt')
