# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_TORCH:
    import torch
    import torch.nn as nn

    from pymor.core.base import BasicObject
    from pymor.models.basic import StationaryModel
    from pymor.models.interface import Model
    from pymor.vectorarrays.numpy import NumpyVectorSpace


    class StationaryNeuralNetworkModel(StationaryModel):

        def __init__(self, neural_network, operator, rhs, output_functional=None, products=None,
                error_estimator=None, visualizer=None, name=None):

            super().__init__(operator, rhs, output_functional=output_functional, products=products,
                    error_estimator=error_estimator, visualizer=visualizer, name=name)

            self.__auto_init(locals())

            # make sure that the operators source matches the neural network
            assert self.solution_space == NumpyVectorSpace(neural_network.output_dimension)

        def _compute_solution(self, mu=None, **kwargs):

            # convert the parameter `mu` into a form that is usable in PyTorch
            converted_input = torch.from_numpy(mu.to_numpy()).double()
            # obtain (reduced) coordinates by forward pass of the parameter values through the neural network
            U = self.neural_network(converted_input).data.numpy()
            # convert plain numpy array to element of the actual solution space
            return self.solution_space.make_array(U)


    class NeuralNetworkModel(Model):
        """Class for models of stationary problems that use artificial neural networks.

        This class implements a |Model| that uses a neural network for solving.

        Parameters
        ----------
        neural_network
            The neural network that approximates the mapping from parameter space
            to solution space. Should be an instance of
            :class:`~pymor.models.neural_network.FullyConnectedNN` with input size that
            matches the (total) number of parameters and output size equal to the
            dimension of the reduced space.
        parameters
            |Parameters| of the reduced order model (the same as used in the full-order
            model).
        output_functional
            |Operator| mapping a given solution to the model output. In many applications,
            this will be a |Functional|, i.e. an |Operator| mapping to scalars.
            This is not required, however.
        products
            A dict of inner product |Operators| defined on the discrete space the
            problem is posed on. For each product with key `'x'` a corresponding
            attribute `x_product`, as well as a norm method `x_norm` is added to
            the model.
        error_estimator
            An error estimator for the problem. This can be any object with
            an `estimate_error(U, mu, m)` method. If `error_estimator` is
            not `None`, an `estimate_error(U, mu)` method is added to the
            model which will call `error_estimator.estimate_error(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, m, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the
            visualizer's `visualize` method.
        name
            Name of the model.
        """

        def __init__(self, neural_network, parameters={}, output_functional=None,
                     products=None, error_estimator=None, visualizer=None, name=None):

            super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

            self.__auto_init(locals())
            self.solution_space = NumpyVectorSpace(neural_network.output_dimension)
            self.linear = output_functional is None or output_functional.linear
            if output_functional is not None:
                self.output_space = output_functional.range

        def _compute_solution(self, mu=None, **kwargs):

            # convert the parameter `mu` into a form that is usable in PyTorch
            converted_input = torch.from_numpy(mu.to_numpy()).double()
            # obtain (reduced) coordinates by forward pass of the parameter values through the neural network
            U = self.neural_network(converted_input).data.numpy()
            # convert plain numpy array to element of the actual solution space
            U = self.solution_space.make_array(U)

            return U

    class NeuralNetworkInstationaryModel(Model):
        """Class for models of instationary problems that use artificial neural networks.

        This class implements a |Model| that uses a neural network for solving.

        Parameters
        ----------
        T
            The final time T.
        Nt
            The number of time steps.
        neural_network
            The neural network that approximates the mapping from parameter space
            to solution space. Should be an instance of
            :class:`~pymor.models.neural_network.FullyConnectedNN` with input size that
            matches the (total) number of parameters and output size equal to the
            dimension of the reduced space.
        parameters
            |Parameters| of the reduced order model (the same as used in the full-order
            model).
        output_functional
            |Operator| mapping a given solution to the model output. In many applications,
            this will be a |Functional|, i.e. an |Operator| mapping to scalars.
            This is not required, however.
        products
            A dict of inner product |Operators| defined on the discrete space the
            problem is posed on. For each product with key `'x'` a corresponding
            attribute `x_product`, as well as a norm method `x_norm` is added to
            the model.
        estimator
            An error estimator for the problem. This can be any object with
            an `estimate(U, mu, m)` method. If `estimator` is
            not `None`, an `estimate(U, mu)` method is added to the
            model which will call `estimator.estimate(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, m, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the
            visualizer's `visualize` method.
        name
            Name of the model.
        """

        def __init__(self, T, Nt, neural_network, parameters={}, output_functional=None,
                     products=None, estimator=None, visualizer=None, name=None):

            super().__init__(products=products, estimator=estimator, visualizer=visualizer, name=name)

            self.__auto_init(locals())
            self.solution_space = NumpyVectorSpace(neural_network.output_dimension)
            self.linear = output_functional is None or output_functional.linear
            if output_functional is not None:
                self.output_space = output_functional.range

        def _compute_solution(self, mu=None, **kwargs):

            U = self.solution_space.empty(reserve=self.Nt+1)
            dt = self.T / self.Nt
            t = 0.

            # iterate over time steps
            for i in range(self.Nt + 1):
                mu = mu.with_(t=t)
                # convert the parameter `mu` into a form that is usable in PyTorch
                converted_input = torch.from_numpy(mu.to_numpy()).double()
                # obtain (reduced) coordinates by forward pass of the parameter values through the neural network
                result_neural_network = self.neural_network(converted_input).data.numpy()
                # convert plain numpy array to element of the actual solution space
                U.append(self.solution_space.make_array(result_neural_network))
                t += dt

            return U


    class NeuralNetworkOutputModel(Model):
        """Class for models of stationary problems that use artificial neural networks.

        This class implements a |Model| that uses a neural network for solving.

        Parameters
        ----------
        neural_network
            The neural network that approximates the mapping from parameter space
            to solution space. Should be an instance of
            :class:`~pymor.models.neural_network.FullyConnectedNN` with input size that
            matches the (total) number of parameters and output size equal to the
            dimension of the reduced space.
        parameters
            |Parameters| of the reduced order model (the same as used in the full-order
            model).
        output_functional
            |Operator| mapping a given solution to the model output. In many applications,
            this will be a |Functional|, i.e. an |Operator| mapping to scalars.
            This is not required, however.
        products
            A dict of inner product |Operators| defined on the discrete space the
            problem is posed on. For each product with key `'x'` a corresponding
            attribute `x_product`, as well as a norm method `x_norm` is added to
            the model.
        error_estimator
            An error estimator for the problem. This can be any object with
            an `estimate_error(U, mu, m)` method. If `error_estimator` is
            not `None`, an `estimate_error(U, mu)` method is added to the
            model which will call `error_estimator.estimate_error(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, m, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the
            visualizer's `visualize` method.
        name
            Name of the model.
        """

        def __init__(self, neural_network, parameters={}, output_functional=None,
                     products=None, error_estimator=None, visualizer=None, name=None):

            super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

            self.__auto_init(locals())
            self.solution_space = NumpyVectorSpace(neural_network.output_dimension)
            self.linear = output_functional is None or output_functional.linear
            if output_functional is not None:
                self.output_space = output_functional.range


        def _compute_solution(self, mu=None, **kwargs):
            return None


        def _compute_output(self, solution, mu=None, **kwargs):

            # convert the parameter `mu` into a form that is usable in PyTorch
            converted_input = torch.from_numpy(mu.to_numpy()).double()
            # obtain (reduced) coordinates by forward pass of the parameter values through the neural network
            U = self.neural_network(converted_input).data.numpy()
            # convert plain numpy array to element of the actual solution space
            U = self.solution_space.make_array(U)

            return U
