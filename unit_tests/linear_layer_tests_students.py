import unittest
import numpy as np
from torch.nn import functional
from torch import Tensor
from torch.autograd import gradcheck, Variable
from netlib.Layer import Linear
import torch
import torch.nn as nn

from configs.config import cfg
from netlib.Layer import InitType
from netlib.ActivationFunction import LinearA, Sigmoid, ReLU, Tanh


class TestMLP(unittest.TestCase):
    def test1(self):
        b_size, input_dim, output_dim = 16, 784, 10
        test_input = np.random.normal(size=(b_size, input_dim))

        fc1 = Linear(input_shape=784,
                     output_shape=10,
                     use_bias=False,
                     initialization_type=InitType.Normal,
                     regularization_type='',
                     weight_decay='',
                     layer_name='fc_1',
                     cfg=cfg)

        # forward pass check
        result = fc1(test_input)
        torch_result_input = Variable(Tensor(test_input), requires_grad=True)
        torch_result_W = Variable(Tensor(fc1.W), requires_grad=True).t()
        torch_result = functional.linear(torch_result_input, torch_result_W)

        self.assertEqual(np.array_equal(result.shape, torch_result.detach().numpy().shape), True)
        self.assertEqual(np.allclose(result, torch_result.detach().numpy(), atol=1e-5), True)

        # backward pass check
        grad, _ = fc1.backward(test_input, np.ones((b_size, output_dim)))
        torch_result.backward(torch.ones((b_size, output_dim)))
        torch_grad = torch_result_input.grad
        self.assertEqual(np.allclose(grad, torch_grad.numpy(), atol=1e-6), True)

    def test_fc_size(self):
        """
        в цикле с разным размером полносвязного слоя проверить значения выхода полносвязного слоя и градиентов
        """
        b_size = 16
        fc_shapes = [(784, 64), (64, 128), (128, 10)]
        for shape in fc_shapes:
            test_input = np.random.normal(size=(b_size, shape[0]))
            fc = Linear(input_shape=shape[0],
                        output_shape=shape[1],
                        use_bias=False,
                        initialization_type=InitType.Normal,
                        regularization_type='',
                        weight_decay='',
                        layer_name='fc_1',
                        cfg=cfg)

            # forward pass check
            result = fc(test_input)
            torch_result_input = Variable(Tensor(test_input), requires_grad=True)
            torch_result_W = Variable(Tensor(fc.W), requires_grad=True).t()
            torch_result = functional.linear(torch_result_input, torch_result_W)

            self.assertEqual(np.array_equal(result.shape, torch_result.detach().numpy().shape), True)
            self.assertEqual(np.allclose(result, torch_result.detach().numpy(), atol=1e-5), True)

            # backward pass check
            grad, _ = fc.backward(test_input, np.ones((b_size, shape[1])))
            torch_result.backward(torch.ones((b_size, shape[1])))
            torch_grad = torch_result_input.grad
            self.assertEqual(np.allclose(grad, torch_grad.numpy(), atol=1e-6), True)

    def test_activation_function(self):
        """
        в цикле по разным функциям активации проверить значения выхода функции активации и градиентов
        """
        activation_functions = [Sigmoid(), ReLU(), Tanh(), LinearA()]
        torch_activation_functions = [nn.Sigmoid(), nn.ReLU(), nn.Tanh(), nn.Identity()]
        b_size, input_dim, output_dim = 16, 64, 64
        test_input = np.random.normal(size=(b_size, input_dim))

        for i, act_func in enumerate(activation_functions):
            # forward pass check
            result = act_func(test_input)
            torch_result_input = Variable(Tensor(test_input), requires_grad=True)
            torch_result = torch_activation_functions[i](torch_result_input)
            self.assertEqual(np.allclose(result, torch_result.detach().numpy(), atol=1e-6), True)

            # backward pass check
            grad, _ = act_func.backward(test_input, torch.ones((b_size, output_dim)))
            torch_result.backward(torch.ones((b_size, output_dim)))
            torch_grad = torch_result_input.grad
            self.assertEqual(np.allclose(grad, torch_grad.numpy(), atol=1e-6), True)


if __name__ == "__main__":
    unittest.main()
