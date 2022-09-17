from data.datasets.MnistDataset import MnistDataset
from data.datasets.CifarDataset import CifarDataset
from data.dataloaders.Dataloader import DataLoader
from configs.mnist_config import cfg as mnist_cfg
from configs.cifar_config import cfg as cifar_cfg


def get_data(cfg, dataset_type):
    Dataset = CifarDataset if cfg["dataset"] == 'cifar' else MnistDataset
    dataset_cfg = cifar_cfg if cfg["dataset"] == 'cifar' else mnist_cfg

    dataset = Dataset(data_path=dataset_cfg['path'][dataset_type],
                      dataset_type=dataset_type,
                      transforms=dataset_cfg['transform_parameters'][dataset_type],
                      nrof_classes=dataset_cfg['classes'])

    dl = DataLoader(dataset,
                    nrof_classes=dataset_cfg['classes'],
                    dataset_type=dataset_type,
                    shuffle=cfg['dataloader']['shuffle'][dataset_type],
                    batch_size=cfg['dataloader']['batch_size'][dataset_type],
                    sample_type=cfg['dataloader']['sample_type'],
                    epoch_size=cfg['dataloader']['epoch_size'],
                    probabilities=cfg['dataloader']['probabilities'])
    return dl
