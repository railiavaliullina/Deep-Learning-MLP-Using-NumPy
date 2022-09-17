from data.dataloaders.Sampler import SampleType
from netlib.Layer import InitType
from netlib.Optim import OptimType
from netlib.ActivationFunction import ActivationFunction
from utils.validation_utils import validate_weights_init, validate_optim_type, validate_activation_functions, \
    validate_use_bias, validate_nrof_layers, validate_layers_sizes

cfg = {
    "dataset":
        "mnist",  # ["mnist", "cifar"]

    "dataloader": {
        "nb_epochs": 30,
        "sample_type": SampleType.DEFAULT,  # ['DEFAULT', 'UPSAMPLE', 'DOWNSAMPLE', 'PROBABILITIES']
        "epoch_size": None,
        "probabilities": [0.1, 0.1, 0.02, 0.08, 0.3, 0.05, 0.05, 0.2, 0.03, 0.07],  # [probabilities array, None]

        "shuffle": {
            "train": True,
            "test": False},

        "batch_size": {
            "train": 32,
            "test": 32
        },
        "show_batch":
            {
                "nrof_batches_to_save": 3,
                "fig_size": (8, 8),
                "show_fig": True,
                "save_fig": True,
                "path_to_save": 'batches_images/',
            }
    },

    "model": {
        'layers': ['fc_1', ActivationFunction.ReLU, 'fc_2'],  # с одним скрытым слоем
        # 'layers': ['fc_1', ActivationFunction.Linear, 'fc_2', ActivationFunction.Linear, 'fc_3'],  # с двумя скрытыми слоями

        'fc_1':
            {
                'layer_input_dim': 784,
                'layer_output_dim': 128,
                'use_bias': True,
                "init_type": InitType.HeNormal,  # [Normal, Uniform, HeNormal, HeUniform, XavierNormal, XavierUniform,
                # GlorotNormal, GlorotUniform]
                'scale': 0.1,  # for InitType.Normal
                'l': 0.1,  # for InitType.Uniform
                # 'act_function': ActivationFunction.Linear
            },
        'fc_2':
            {
                'layer_input_dim': 128,
                'layer_output_dim': 10,
                'use_bias': True,
                "init_type": InitType.HeNormal,
                'scale': 0.1,  # for InitType.Normal
                'l': 0.1,  # for InitType.Uniform
            },
        'fc_3':
            {
                'layer_input_dim': 128,
                'layer_output_dim': 10,
                'use_bias': True,
                "init_type": InitType.HeNormal,
                'scale': 0.1,  # for InitType.Normal
                'l': 0.1,  # for InitType.Uniform
            },
    },

    "train":
        {
            'optim_type': OptimType.SGD,  # [SGD, MomentumSGD]
            'learning_rate': 0.1,
            'momentum': 0.9,
            'log_metrics': True,
            'experiment_name': 'mlp without aug, best_model_with_Linear_activation',  # для добавления в mlflow
            'checkpoints_dir': 'checkpoints/models_validation/',
            'save_model': True,
            'load_model': False,
            'epoch_to_load': 20,
            'save_frequency': 50,
            'evaluate_on_train_data': True,
            'evaluate_before_training': False,
        },

    "overfitting_on_batch":  # параметры для задачи оверфиттинга на одном батче
        {
            "nb_iters": 1000,
            "hyperparams_validation":
                {  # словарь функций валидации, которые нужно запустить при оверфиттинге на батче
                    'validation_functions': {'validate_nrof_layers': validate_nrof_layers,
                                             'validate_weights_init': validate_weights_init,
                                             'validate_use_bias': validate_use_bias,
                                             'validate_optim_type': validate_optim_type,
                                             'validate_activation_functions': validate_activation_functions,
                                             'validate_layers_sizes': validate_layers_sizes}
                }
        },
    "models_validation":
        {
            "hyperparams_validation":
                {  # словарь функций валидации, которые нужно запустить при валидации параметров модели
                    'validation_functions': {'validate_nrof_layers': validate_nrof_layers,
                                             'validate_activation_functions': validate_activation_functions,
                                             'validate_layers_sizes': validate_layers_sizes},
                    'experiments_names': ['nrof_layers(1)', 'nrof_layers(2)', 'act_functions(ReLU)',
                                          'act_functions(Sigmoid)', 'act_functions(Tanh)', 'act_functions(Linear)',
                                          'layers_sizes(64)', 'layers_sizes(128)']
                }
        }
}
