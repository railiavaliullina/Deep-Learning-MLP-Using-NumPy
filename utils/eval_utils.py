import numpy as np


def get_accuracy(input, dataset_cfg, net, on_batch=False):
    images, labels = input if on_batch else (input.dataset.images, input.dataset.labels)
    nb_images = images.shape[0]
    labels = labels if on_batch else np.squeeze(np.eye(nb_images, dataset_cfg['classes'])[labels.reshape(nb_images, -1)], 1)
    images = np.stack(images).reshape((nb_images, -1))
    accuracy = np.mean(np.argmax(labels, axis=1) == np.argmax(net(images)[-1], axis=1)) * 100
    return accuracy


def evaluate(net, train_dl, test_dl, dataset_cfg, cfg):
    net.train = False
    # evaluate on train data
    if cfg['train']['evaluate_on_train_data']:
        acc_train = get_accuracy(train_dl, dataset_cfg, net)
        print(f'Accuracy on train data before training: {acc_train}')
    else:
        acc_train = None
    # evaluate on test data
    acc_test = get_accuracy(test_dl, dataset_cfg, net)
    print(f'Accuracy on test data before training: {acc_test}')
    net.train = True
    return acc_train, acc_test

