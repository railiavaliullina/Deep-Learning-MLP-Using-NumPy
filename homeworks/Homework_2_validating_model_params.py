import numpy as np
import warnings
import time
from copy import deepcopy

from configs.mnist_config import cfg as mnist_cfg
from configs.cifar_config import cfg as cifar_cfg
from configs.config import cfg as initial_cfg
from utils.data_utils import get_data
from models.BaseModel import BaseModelClass
from utils.eval_utils import evaluate
from utils.log_utils import log_metrics, log_params, start_logging, end_logging
warnings.filterwarnings('ignore')


def run(cfg):
    start_time = time.time()
    # initialize model
    net = BaseModelClass(cfg)
    start_epoch = 0

    if cfg['train']['load_model']:
        e = cfg['train']['epoch_to_load']
        net.load_weights(cfg['train']['checkpoints_dir'] + f'checkpoint_{e}')
        start_epoch = e + 1

    # save to mlflow experiment name and experiment params
    start_logging(cfg)
    log_params(cfg)

    # evaluate on train and test data before training
    if cfg['train']['evaluate_before_training']:
        acc_train, acc_test = evaluate(net, train_dl, test_dl, dataset_cfg, cfg)

    losses, global_step = [], 0
    nb_iters_per_epoch = len(train_dl.dataset) // train_dl.batch_size

    # training loop
    for e in range(start_epoch, start_epoch + cfg["dataloader"]["nb_epochs"]):
        print(f'Epoch: {e}')

        batch_generator_ = train_dl.batch_generator()
        for i, batch in enumerate(batch_generator_):
            images, labels = batch
            images = np.stack(images).reshape((cfg['dataloader']['batch_size']['train'], -1))
            loss = net.make_step(images, labels)
            losses.append(loss)

            if i % 100 == 0:
                mean_loss = np.mean(losses)
                print(f'Loss at step {global_step}: {mean_loss}')
            global_step += 1

            # log loss
            log_metrics(['loss'], [loss], global_step, cfg)

        # log mean loss per epoch
        log_metrics(['mean_loss'], [np.mean(losses[:-nb_iters_per_epoch]), ], e, cfg)

        # save model
        if cfg['train']['save_model'] and e % cfg['train']['save_frequency'] == 0:
            net.dump_model(cfg['train']['checkpoints_dir'] + f'checkpoint_{e}')

        # evaluate on train and test data
        acc_train, acc_test = evaluate(net, train_dl, test_dl, dataset_cfg, cfg)

        # log accuracies
        log_metrics(['train_accuracy', 'test_accuracy'], [acc_train, acc_test], e, cfg)

    end_logging(cfg)

    # save last model
    if cfg['train']['save_model']:
        e = start_epoch + cfg["dataloader"]["nb_epochs"]
        net.dump_model(cfg['train']['checkpoints_dir'] + f'checkpoint_{e}')

    print(f'Total time: {round((time.time() - start_time) / 60, 3)} min')


if __name__ == '__main__':
    dataset_cfg = cifar_cfg if initial_cfg["dataset"] == 'cifar' else mnist_cfg

    # get train, test data
    print(f'Current dataset: {initial_cfg["dataset"]}')
    train_dl = get_data(initial_cfg, dataset_type='train')
    test_dl = get_data(initial_cfg, dataset_type='test')

    experiments_names = initial_cfg['models_validation']['hyperparams_validation']['experiments_names']
    exp_id = 0

    for function_name, validation_func in \
            initial_cfg['models_validation']['hyperparams_validation']['validation_functions'].items():
        print(f'\nValidation type: {function_name}')
        initial_cfg_ = deepcopy(initial_cfg)
        cfgs = validation_func(initial_cfg_)
        for name, cfg in cfgs.items():
            cfg_ = deepcopy(cfg)
            # print(f'\n{name}')
            cfg_['train']['experiment_name'] = experiments_names[exp_id]
            cfg_['train']['checkpoints_dir'] += str(experiments_names[exp_id]) + '/'
            print('\nValidating: ', experiments_names[exp_id])
            run(cfg_)
            exp_id += 1
