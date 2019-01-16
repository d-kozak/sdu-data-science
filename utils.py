import random


def run_neural_network(prepare_input_data, split_input_data, build_neural_network, evaluate_model):
    input_data = prepare_input_data()
    random.shuffle(input_data)

    ((train_images, train_labels), (test_images, test_labels)) = split_input_data(input_data)

    model = build_neural_network()

    val_loss, val_acc = evaluate_model(model, test_images, test_labels, train_images, train_labels)
    print('Loss value ' + str(val_loss))
    print('Accuracy ' + str(val_acc))
    return model, val_loss, val_acc
