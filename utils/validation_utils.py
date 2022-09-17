from netlib.Layer import InitType
from netlib.Optim import OptimType
from netlib.ActivationFunction import ActivationFunction
from copy import deepcopy

from configs.mnist_config import cfg as mnist_cfg
from configs.cifar_config import cfg as cifar_cfg


def validate_weights_init(cfg):
    cfgs = {}
    for init_type in enumerate(InitType):
        cfg_ = deepcopy(cfg)
        cfg_['model']['fc_1']['init_type'] = init_type[1]
        cfgs[str(init_type[1])] = cfg_
    return cfgs


def validate_optim_type(cfg):
    cfgs = {}
    for optim_type in enumerate(OptimType):
        cfg_ = deepcopy(cfg)
        cfg_['train']['optim_type'] = optim_type[1]
        cfgs[str(optim_type[1])] = cfg_
    return cfgs


def validate_activation_functions(cfg):
    cfgs = {}
    for act_func in enumerate(ActivationFunction):
        cfg_ = deepcopy(cfg)
        cfg_['model']['layers'][1] = act_func[1]
        cfgs[str(act_func[1])] = cfg_
    return cfgs


def validate_use_bias(cfg):
    cfgs = {}
    for use_bias in [True, False]:
        cfg_ = deepcopy(cfg)
        cfg_['model']['fc_1']['use_bias'] = use_bias
        cfgs[str(use_bias)] = cfg_
    return cfgs


def validate_nrof_layers(cfg):
    dataset_cfg = cifar_cfg if cfg["dataset"] == 'cifar' else mnist_cfg
    cfgs = {}
    n_layers = [['fc_1', ActivationFunction.ReLU, 'fc_2'],
                ['fc_1', ActivationFunction.ReLU, 'fc_2', ActivationFunction.ReLU, 'fc_3']]

    layers = ['fc_1', 'fc_2', 'fc_2', 'fc_3']
    layers_dim = ['layer_output_dim', 'layer_input_dim', 'layer_output_dim', 'layer_input_dim']
    layers_sizes = 64
    for l, layer in enumerate(layers):
        cfg['model'][layer][layers_dim[l]] = layers_sizes

    n_layers_str = [1, 2]
    for i, nrof_layers in enumerate(n_layers):
        if n_layers_str[i] == 1:
            cfg['model']['fc_2']['layer_output_dim'] = dataset_cfg['classes']
        else:
            cfg['model']['fc_2']['layer_output_dim'] = layers_sizes
        cfg_ = deepcopy(cfg)
        cfg_['model']['layers'] = nrof_layers
        cfgs[str(n_layers_str[i])] = cfg_
    return cfgs


def validate_layers_sizes(cfg):
    cfgs = {}
    sizes = [64, 128]
    for size in sizes:
        cfg_ = deepcopy(cfg)
        cfg_['model']['fc_1']['layer_output_dim'] = size
        cfg_['model']['fc_2']['layer_input_dim'] = size
        cfgs[str(size)] = cfg_
    return cfgs
