# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_TORCH:
    from numbers import Number

    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils as utils

    from pymor.algorithms.pod import pod
    from pymor.core.base import BasicObject
    from pymor.core.exceptions import NeuralNetworkTrainingFailed
    from pymor.algorithms.neural_network import (
            FullyConnectedNN, EarlyStoppingScheduler, CustomDataset, certified_ann_training)
    from pymor.models.neural_network import (
            StationaryNeuralNetworkModel, StationaryNeuralNetworkOutputModel,
            NeuralNetworkModel, NeuralNetworkInstationaryModel)
    from pymor.parameters.base import Mu
    from pymor.reductors.basic import StationaryRBReductor
    from pymor.vectorarrays.interface import VectorArray


    class StationaryNeuralNetworkRBReductor(StationaryRBReductor):
        '''
        assumes RB to be orthogonal w.r.t. product
        '''

        def __init__(self, fom, training_data, validation_data,
                RB=None, product=None, check_orthonormality=None, check_tol=None,
                hidden_layers='[(N+P)*3, (N+P)*3]', ann_mse='like_basis',
                max_restarts=10, torch_seed=None, ann_train_params=None):

            super().__init__(fom, RB=RB, product=product,
                    check_orthonormality=check_orthonormality, check_tol=check_tol)

            # check that training and validation data is a tuple (mus, snapshots) of a list/tuple
            # of mus and a VectorArray of snaphots, where snapshots[i] == fom.solve(mu[i]) is assumed
            for data in training_data, validation_data:
                assert isinstance(data, (list, tuple)) and len(data) == 2
                assert isinstance(data[0], (list, tuple))
                assert isinstance(data[1], VectorArray)
                assert len(data[0]) == len(data[1])
                assert all(isinstance(mu, Mu) for mu in data[0])
                assert data[1].space == fom.solution_space

            assert isinstance(ann_mse, Number) or (isinstance(ann_mse, str) and ann_mse == 'like_basis') or not ann_mse

            # if applicable, set a common seed for the PyTorch initialization
            # of weights and biases and further PyTorch methods for all training runs ...
            ann_train_params = ann_train_params or {}
            assert isinstance(ann_train_params, dict)
            if not torch_seed and 'seed' in ann_train_params:
                torch_seed = ann_train_params['seed']
            ann_train_params.pop('seed', None) # has to be removed, otherwise each training uses the same
            if torch_seed:
                torch.manual_seed(torch_seed)

            self.__auto_init(locals())

        def build_rom(self, projected_operators, error_estimator):
            assert not error_estimator, 'Did not think about this yet!'

            RB = self.bases['RB']

            # determine the numbers of neurons in the hidden layers
            if isinstance(self.hidden_layers, str):
                hidden_layers = eval(self.hidden_layers, {'N': len(RB), 'P': self.fom.parameters.dim})
            else:
                hidden_layers = self.hidden_layers
            assert isinstance(hidden_layers, (list, tuple)) and len(hidden_layers) == 2
            assert all(isinstance(l, Number) for l in hidden_layers)
            layers = [len(self.fom.parameters),] + hidden_layers + [len(RB),]

            # perform orthogonal projection of training and validation snapshots onto current basis
            def data_to_tensor(data):
                assert len(data[0]) == len(data[1])
                orth_projs = RB.inner(data[1], product=self.product)
                return [(torch.DoubleTensor(data[0][i].to_numpy()), torch.DoubleTensor(orth_projs[:, i]))
                        for i in np.arange(len(data[0]))]
            training_data = data_to_tensor(self.training_data)
            validation_data = data_to_tensor(self.validation_data)

            # compute desired loss
            if isinstance(self.ann_mse, str) and self.ann_mse == 'like_basis':
                # compute orthogonal projection error of training snapshots onto current basis
                V = self.training_data[1]
                v = RB.inner(V, product=self.product)
                V_proj = RB.lincomb(v.T)
                ann_mse = np.max((V - V_proj).norm(product=self.product))
            else:
                ann_mse = self.ann_mse

            # run the actual training of the neural network
            self.neural_network, self.losses = certified_ann_training(
                    training_data, validation_data, layers, target_loss=ann_mse, max_restarts=self.max_restarts,
                    **self.ann_train_params)

            # create model
            return StationaryNeuralNetworkModel(self.neural_network, error_estimator=None, **projected_operators)


    class StationaryNeuralNetworkOutputReductor(BasicObject):

        # TODO: allow for transformation/back_transformation of outputs (and inputs?)!
        def __init__(self, fom, training_data, validation_data,
                     hidden_layers='[P*3, P*3]', ann_mse=None, max_restarts=10, torch_seed=None, ann_train_params=None):

            assert fom.parameters and len(np.sum(fom.parameters.values())) > 0

            # check that training and validation data is a tuple (mus, outputs) of a list/tuple
            # of mus and a VectorArray of outputs, where outputs[i] == fom.output(mu[i]) is assumed
            for data in training_data, validation_data:
                assert isinstance(data, (list, tuple)) and len(data) == 2
                assert isinstance(data[0], (list, tuple))
                assert isinstance(data[1], VectorArray)
                assert len(data[0]) == len(data[1])
                assert all(isinstance(mu, Mu) for mu in data[0])
                assert data[1].space == fom.output_space

            assert isinstance(ann_mse, Number) or not ann_mse

            # if applicable, set a common seed for the PyTorch initialization
            # of weights and biases and further PyTorch methods for all training runs ...
            ann_train_params = ann_train_params or {}
            assert isinstance(ann_train_params, dict)
            if not torch_seed and 'seed' in ann_train_params:
                torch_seed = ann_train_params['seed']
            ann_train_params.pop('seed', None) # has to be removed, otherwise each training uses the same
            if torch_seed:
                torch.manual_seed(torch_seed)

            # determine the numbers of neurons in the hidden layers
            if isinstance(hidden_layers, str):
                hidden_layers = eval(hidden_layers, {'P': fom.parameters.dim})
            else:
                hidden_layers = self.hidden_layers
            assert isinstance(hidden_layers, (list, tuple)) and len(hidden_layers) == 2
            assert all(isinstance(l, Number) for l in hidden_layers)
            self.layers = [len(fom.parameters),] + hidden_layers + [fom.output_space.dim,]

            self.__auto_init(locals())

        def reduce(self):

            if hasattr(self, 'rom'):
                return self.rom

            def data_to_tensor(data):
                assert len(data[0]) == len(data[1])
                return [(torch.DoubleTensor(mu.to_numpy()), torch.DoubleTensor(output.to_numpy()))
                        for mu, output in zip(data[0], data[1])]
            training_data = data_to_tensor(self.training_data)
            validation_data = data_to_tensor(self.validation_data)

            # run the actual training of the neural network
            self.neural_network, self.losses = certified_ann_training(
                    training_data, validation_data, self.layers, target_loss=self.ann_mse,
                    max_restarts=self.max_restarts, **self.ann_train_params)

            # create model
            self.rom = StationaryNeuralNetworkOutputModel(
                    self.fom.parameters, self.neural_network, linear=self.fom.linear, name=f'{self.fom.name}_reduced')
            return self.rom


    class NeuralNetworkReductor(BasicObject):
        """Reduced Basis reductor relying on artificial neural networks.

        This is a reductor that constructs a reduced basis using proper
        orthogonal decomposition and trains a neural network that approximates
        the mapping from parameter space to coefficients of the full-order
        solution in the reduced basis.
        The approach is described in [HU18]_.

        Parameters
        ----------
        fom
            The full-order |Model| to reduce.
        training_set
            Set of |parameter values| to use for POD and training of the
            neural network.
        validation_set
            Set of |parameter values| to use for validation in the training
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
        basis_size
            Desired size of the reduced basis. If `None`, rtol, atol or l2_err must
            be provided.
        rtol
            Relative tolerance the basis should guarantee on the training set.
        atol
            Absolute tolerance the basis should guarantee on the training set.
        l2_err
            L2-approximation error the basis should not exceed on the training
            set.
        pod_params
            Dict of additional parameters for the POD-method.
        ann_mse
            If `'like_basis'`, the mean squared error of the neural network on
            the training set should not exceed the error of projecting onto the basis.
            If `None`, the neural network with smallest validation error is
            used to build the ROM.
            If a tolerance is prescribed, the mean squared error of the neural
            network on the training set should not exceed this threshold.
            Training is interrupted if a neural network that undercuts the
            error tolerance is found.
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     basis_size=None, rtol=0., atol=0., l2_err=0., pod_params=None,
                     ann_mse='like_basis'):
            assert 0 < validation_ratio < 1 or validation_set
            self.__auto_init(locals())


        def reduce(self, hidden_layers='[(N+P)*3, (N+P)*3]', activation_function=torch.tanh,
                   optimizer=optim.LBFGS, epochs=1000, batch_size=20, learning_rate=1.,
                   restarts=10, seed=0):
            """Reduce by training artificial neural networks.

            Parameters
            ----------
            hidden_layers
                Number of neurons in the hidden layers. Can either be fixed or
                a Python expression string depending on the reduced basis size
                `N` and the total dimension of the |Parameters| `P`.
            activation_function
                Activation function to use between the hidden layers.
            optimizer
                Algorithm to use as optimizer during training.
            epochs
                Maximum number of epochs for training.
            batch_size
                Batch size to use if optimizer allows mini-batching.
            learning_rate
                Step size to use in each optimization step.
            restarts
                Number of restarts of the training algorithm. Since the training
                results highly depend on the initial starting point, i.e. the
                initial weights and biases, it is advisable to train multiple
                neural networks by starting with different initial values and
                choose that one performing best on the validation set.
            seed
                Seed to use for various functions in PyTorch. Using a fixed seed,
                it is possible to reproduce former results.

            Returns
            -------
            rom
                Reduced-order |NeuralNetworkModel|.
            """
            assert restarts > 0
            assert epochs > 0
            assert batch_size > 0
            assert learning_rate > 0.

            # set a seed for the PyTorch initialization of weights and biases and further PyTorch methods
            torch.manual_seed(seed)

            # build a reduced basis using POD and compute training data
            if not hasattr(self, 'reduced_basis'):
                self.build_basis()

            # determine the numbers of neurons in the hidden layers
            if isinstance(hidden_layers, str):
                hidden_layers = eval(hidden_layers, {'N': len(self.reduced_basis), 'P': self.fom.parameters.dim})
            # input and output size of the neural network are prescribed by the dimension of the parameter space
            # and the reduced basis size
            assert isinstance(hidden_layers, list)
            layers = self._compute_layers_sizes(hidden_layers)

            # compute validation data
            if not hasattr(self, 'validation_data'):
                with self.logger.block('Computing validation snapshots ...'):

                    if self.validation_set:
                        U = self.fom.solution_space.empty()
                        for mu in self.validation_set:
                            U.append(self.fom.solve(mu))
                        self.validation_data = self._compute_samples(self.validation_set, U)
                    else:
                        number_validation_snapshots = int(len(self.training_data)*self.validation_ratio)
                        self.validation_data = self.training_data[0:number_validation_snapshots]
                        self.training_data = self.training_data[number_validation_snapshots+1:]

            # run the actual training of the neural network
            with self.logger.block(f'Performing {restarts} restarts for training ...'):

                for run in range(restarts):
                    neural_network, current_losses = self._train(layers, activation_function, optimizer,
                                                       epochs, batch_size, learning_rate)
                    if not hasattr(self, 'losses') or current_losses['val'] < self.losses['val']:
                        self.losses = current_losses
                        self.neural_network = neural_network

                        # check if neural network is sufficient to guarantee certain error bounds
                        with self.logger.block('Checking tolerances for error of neural network ...'):

                            if isinstance(self.ann_mse, Number) and self.losses['full'] <= self.ann_mse:
                                self.logger.info(f'Aborting training after {run} restarts ...')
                                return self._build_rom()
                            elif self.ann_mse == 'like_basis' and self.losses['full'] <= self.mse_basis:
                                self.logger.info(f'Aborting training after {run} restarts ...')
                                return self._build_rom()


            # check if neural network is sufficient to guarantee certain error bounds
            with self.logger.block('Checking tolerances for error of neural network ...'):

                if isinstance(self.ann_mse, Number) and self.losses['full'] > self.ann_mse:
                    raise NeuralNetworkTrainingFailed('Could not train a neural network that '
                                                      'guarantees prescribed tolerance!')
                elif self.ann_mse == 'like_basis' and self.losses['full'] > self.mse_basis:
                    raise NeuralNetworkTrainingFailed('Could not train a neural network with an error as small as the '
                                                      'reduced basis error! Maybe you can try a different neural '
                                                      'network architecture or change the value of `ann_mse`.')
                elif self.ann_mse is None:
                    self.logger.info('Using neural network with smallest validation error ...')
                    self.logger.info(f'Finished training with a validation loss of {self.losses["val"]} ...')
                    return self._build_rom()
                else:
                    raise ValueError('Unknown value for mean squared error of neural network')


        def _compute_layers_sizes(self, hidden_layers):
            """Compute the number of neurons in the layers of the neural network."""
            return [len(self.fom.parameters),] + hidden_layers + [len(self.reduced_basis),]


        def _build_rom(self):
            """Construct the reduced order model."""
            with self.logger.block('Building ROM ...'):
                rom = NeuralNetworkModel(self.neural_network, self.fom.parameters,
                                         name=f'{self.fom.name}_reduced')

            return rom

        def _train(self, layers, activation_function, optimizer, epochs, batch_size, learning_rate):
            """Perform a single training iteration and return the resulting neural network."""
            assert hasattr(self, 'training_data')
            assert hasattr(self, 'validation_data')

            # LBFGS-optimizer does not support mini-batching, so the batch size needs to be adjusted
            if optimizer == optim.LBFGS:
                batch_size = max(len(self.training_data), len(self.validation_data))

            with self.logger.block('Training the neural network ...'):

                # initialize the neural network
                neural_network = FullyConnectedNN(layers,
                                                  activation_function=activation_function).double()

                # initialize the optimizer
                optimizer = optimizer(neural_network.parameters(),
                                      lr=learning_rate)

                loss_function = nn.MSELoss()
                early_stopping_scheduler = EarlyStoppingScheduler(len(self.training_data) + len(self.validation_data))

                # create the training and validation sets as well as the respective data loaders
                training_dataset = CustomDataset(self.training_data)
                validation_dataset = CustomDataset(self.validation_data)
                phases = ['train', 'val']
                training_loader = utils.data.DataLoader(training_dataset,
                                                        batch_size=batch_size)
                validation_loader = utils.data.DataLoader(validation_dataset,
                                                          batch_size=batch_size)
                dataloaders = {'train':  training_loader, 'val': validation_loader}

                self.logger.info('Starting optimization procedure ...')

                # perform optimization procedure
                for epoch in range(epochs):
                    losses = {'full': 0.}

                    # alternate between training and validation phase
                    for phase in phases:
                        if phase == 'train':
                            neural_network.train()
                        else:
                            neural_network.eval()

                        running_loss = 0.0

                        # iterate over batches
                        for batch in dataloaders[phase]:
                            inputs = batch[0]
                            targets = batch[1]

                            with torch.set_grad_enabled(phase == 'train'):
                                def closure():
                                    if torch.is_grad_enabled():
                                        optimizer.zero_grad()
                                    outputs = neural_network(inputs)
                                    loss = loss_function(outputs, targets)
                                    if loss.requires_grad:
                                        loss.backward()
                                    return loss

                                # perform optimization step
                                if phase == 'train':
                                    optimizer.step(closure)

                                # compute loss of current batch
                                loss = closure()

                            # update overall absolute loss
                            running_loss += loss.item() * len(batch[0])

                        # compute average loss
                        epoch_loss = running_loss / len(dataloaders[phase].dataset)

                        losses[phase] = epoch_loss

                        losses['full'] += running_loss

                        # check for early stopping
                        if phase == 'val' and early_stopping_scheduler(losses, neural_network):
                            if not self.logging_disabled:
                                self.logger.info(f'Early stopping training process after {epoch + 1} epochs ...')
                                self.logger.info('Minimum validation loss: '
                                                 f'{early_stopping_scheduler.best_losses["val"]}')
                            return early_stopping_scheduler.best_neural_network, early_stopping_scheduler.best_losses

            return early_stopping_scheduler.best_neural_network, early_stopping_scheduler.best_losses

        def build_basis(self):
            """Compute a reduced basis using proper orthogonal decomposition."""
            with self.logger.block('Building reduced basis ...'):

                # compute snapshots for POD and training of neural networks
                with self.logger.block('Computing training snapshots ...'):
                    U = self.fom.solution_space.empty()
                    for mu in self.training_set:
                        U.append(self.fom.solve(mu))

                # compute reduced basis via POD
                reduced_basis, svals = pod(U, modes=self.basis_size, rtol=self.rtol / 2.,
                                           atol=self.atol / 2., l2_err=self.l2_err / 2.,
                                           **(self.pod_params or {}))

                self.reduced_basis = reduced_basis

                # determine the coefficients of the full-order solutions in the reduced basis to obtain the
                # training data; convert everything into tensors that are compatible with PyTorch
                self.training_data = self._compute_samples(self.training_set, U)

            # compute mean square loss
            self.mse_basis = (sum(U.norm2()) - sum(svals**2)) / len(U)


        def _compute_samples(self, parameters, solutions):
            """Transform parameters and corresponding solutions to tensors."""
            samples = []
            for mu, u in zip(parameters, solutions):
                mu_tensor = torch.DoubleTensor(mu.to_numpy())
                u_tensor = torch.DoubleTensor(self.reduced_basis.inner(u)[:,0])
                samples.append((mu_tensor, u_tensor))
            return samples

        def reconstruct(self, u):
            """Reconstruct high-dimensional vector from reduced vector `u`."""
            assert hasattr(self, 'reduced_basis')
            return self.reduced_basis.lincomb(u.to_numpy())


    class NeuralNetworkInstationaryReductor(NeuralNetworkReductor):
        """Reduced Basis reductor for instationary problems relying on
        artificial neural networks.

        This is a reductor that constructs a reduced basis using proper
        orthogonal decomposition and trains a neural network that approximates
        the mapping from parameter and time space to coefficients of the
        full-order solution in the reduced basis.
        The approach is described in [WHR19]_.

        Parameters
        ----------
        fom
            The full-order |Model| to reduce.
        Nt
            The number of time steps.
        training_set
            Set of |parameter values| to use for POD and training of the
            neural network.
        validation_set
            Set of |parameter values| to use for validation in the training
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
        basis_size
            Desired size of the reduced basis. If `None`, rtol, atol or l2_err must
            be provided.
        rtol
            Relative tolerance the basis should guarantee on the training set.
        atol
            Absolute tolerance the basis should guarantee on the training set.
        l2_err
            L2-approximation error the basis should not exceed on the training
            set.
        pod_params
            Dict of additional parameters for the POD-method.
        ann_mse
            If `'like_basis'`, the mean squared error of the neural network on
            the training set should not exceed the error of projecting onto the basis.
            If `None`, the neural network with smallest validation error is
            used to build the ROM.
            If a tolerance is prescribed, the mean squared error of the neural
            network on the training set should not exceed this threshold.
            Training is interrupted if a neural network that undercuts the
            error tolerance is found.
        """

        def __init__(self, fom, Nt, training_set, validation_set=None, validation_ratio=0.1,
                     basis_size=None, rtol=0., atol=0., l2_err=0., pod_params=None,
                     ann_mse='like_basis'):
            assert isinstance(Nt, int)
            assert 0 < validation_ratio < 1 or validation_set
            self.__auto_init(locals())


        def _compute_layers_sizes(self, hidden_layers):
            """Compute the number of neurons in the layers of the neural network
            (make sure to increase the input dimension to account for the time)."""
            return [len(self.fom.parameters) + 1,] + hidden_layers + [len(self.reduced_basis),]


        def _build_rom(self):
            """Construct the reduced order model."""
            with self.logger.block('Building ROM ...'):
                rom = NeuralNetworkInstationaryModel(self.fom.T, self.Nt, self.neural_network,
                                                     self.fom.parameters, name=f'{self.fom.name}_reduced')

            return rom


        def _compute_samples(self, parameters, solutions):
            """Transform parameters and corresponding solutions to tensors
            (make sure to include the time instances in the inputs)."""
            samples = []
            parameters_with_time = []
            dt = self.fom.T / self.Nt

            for mu in parameters:
                t = 0.
                for i in range(self.Nt + 1):
                    parameters_with_time.append(mu.with_(t=t))
                    t += dt

            for mu, u in zip(parameters_with_time, solutions):
                mu_tensor = torch.DoubleTensor(mu.to_numpy())
                u_tensor = torch.DoubleTensor(self.reduced_basis.inner(u)[:,0])
                samples.append((mu_tensor, u_tensor))

            return samples


    class NeuralNetworkOutputReductor(NeuralNetworkReductor):
        """Reduced Basis reductor for stationary problems relying on
        artificial neural networks.

        This is a reductor that trains a neural network that approximates
        the mapping from parameter space to output space.

        Parameters
        ----------
        fom
            The full-order |Model| to reduce.
        training_set
            Set of |parameter values| to use for POD and training of the
            neural network.
        validation_set
            Set of |parameter values| to use for validation in the training
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1):
            assert 0 < validation_ratio < 1 or validation_set
            self.ann_mse = None
            self.__auto_init(locals())


        def build_basis(self):
            U = self.fom.solution_space.empty()
            for mu in self.training_set:
                U.append(self.fom.solve(mu))
            self.training_data = self._compute_samples(self.training_set, U)
            self.reduced_basis = ()


        def _compute_layers_sizes(self, hidden_layers):
            """Compute the number of neurons in the layers of the neural network
            (make sure to increase the input dimension to account for the time)."""
            return [len(self.fom.parameters),] + hidden_layers + [self.fom.output_space.dim,]


        def _build_rom(self):
            """Construct the reduced order model."""
            with self.logger.block('Building ROM ...'):
                rom = NeuralNetworkOutputModel(self.neural_network, self.fom.parameters,
                                               output_functional=self.fom.output_functional,
                                               name=f'{self.fom.name}_output_reduced')

            return rom


        def _compute_samples(self, parameters, solutions):
            """Transform parameters and corresponding solutions to tensors
            (make sure to include the time instances in the inputs)."""
            samples = []
            for mu, u in zip(parameters, solutions):
                mu_tensor = torch.DoubleTensor(mu.to_numpy())
                u_tensor = torch.DoubleTensor(self.fom.output_functional.apply(u, mu).to_numpy()[:,0])
                samples.append((mu_tensor, u_tensor))
            return samples

