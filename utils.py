import random
from sklearn.model_selection import StratifiedKFold
import numpy as np

def run_neural_network(prepare_input_data, split_input_data, build_neural_network, evaluate_model):
    input_data = prepare_input_data()
    random.shuffle(input_data)

#    ((train_images, train_labels), (test_images, test_labels)) = split_input_data(input_data)

    images = [elem['data'] for elem in input_data]
    labels = [elem['image_type'] for elem in input_data]
    images = np.array(images[:len(images)])
    labels = np.array(labels[:len(labels)])

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
