import random

import numpy as np
from sklearn.model_selection import StratifiedKFold


def run_neural_network(prepare_input_data, build_neural_network, evaluate_model):
    """
    Performs cross validation for the clean classifier, using 5 splits.
    :param prepare_input_data: callback to prepare input data
    :param build_neural_network: callback to build the neural network
    :param evaluate_model: callback to prepare and evaluate the model
    :return:
    """
    input_data = prepare_input_data()
    random.shuffle(input_data)

    images = [elem['data'] for elem in input_data]
    labels = [elem['image_type'] for elem in input_data]
    images = np.array(images)
    labels = np.array(labels)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    cvscores = []
    cvlosses = []
    i = 0

    for train, test in kfold.split(images, labels):
        i += 1
        print("cross validation: ", i)
        model = build_neural_network()

        val_loss, val_acc = evaluate_model(model, images[test], labels[test], images[train], labels[train])
        print('Loss value ' + str(val_loss))
        print('Accuracy ' + str(val_acc))

        cvscores.append(val_acc * 100)
        cvlosses.append(val_loss)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f (+/- %.2f)" % (np.mean(cvlosses), np.std(cvlosses)))

    return model


def is_in_range(cell, lower_bound, upper_bound):
    return ((lower_bound <= cell).all()) and ((cell < upper_bound).all())


assert is_in_range(
    np.array([1, 1, 1]),
    np.array([0, 0, 0]),
    np.array([2, 2, 2])
)

assert is_in_range(
    np.array([0, 0, 0]),
    np.array([0, 0, 0]),
    np.array([2, 2, 2])
)

assert not is_in_range(
    np.array([2, 2, 2]),
    np.array([0, 0, 0]),
    np.array([2, 2, 2])
)

assert is_in_range(
    np.array([1, 2, 3]),
    np.array([1, 2, 1]),
    np.array([2, 3, 4])
)

assert is_in_range(
    np.array([5, 4, 2]),
    np.array([5, 3, 2]),
    np.array([6, 5, 3])
)

assert is_in_range(
    np.array([5, 6, 8]),
    np.array([1, 2, 3]),
    np.array([11, 13, 10])
)

assert not is_in_range(
    np.array([5, 6, 8]),
    np.array([6, 6, 8]),
    np.array([11, 13, 10])
)
