# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.projection import project, project_to_subbasis
from pymor.algorithms.greedy import WeakGreedySurrogate
from pymor.core.interfaces import ImmutableInterface
from pymor.models.basic import StationaryModel
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interfaces import RemoteObjectInterface
from pymor.reductors.basic import StationaryRBReductor
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.reductors.residual import ResidualReductor


class SymmetricCoercivePrimalDualRBReductor(StationaryRBReductor):
    """Primal/dual Reduced Basis reductor for |StationaryModels| with symmetric coercive linear operator.

    In addition to :class:`~pymor.reductors.coercive.CoerciveRBReductor`, a dual rom is assembled to
    provide improved output error estimates as well as a corrected reduced output following the primal-dual
    RB approach [Haa2017]_ (section 2.4). For the reduction of the residuals we use
    :class:`~pymor.reductors.residual.ResidualReductor` for improved numerical stability
    [BEOR14]_.

    Parameters
    ----------
    fom
        The (primal) |Model| which is to be reduced.
    RB
        |VectorArray| containing the (primal) reduced basis on which to project.
    dual_RB
        |VectorArray| containing the dual reduced basis on which to project.
    product
        Inner product for the orthonormalization of `RB` and `dual_RB`, the projection of the
        |Operators| given by `vector_ranged_operators` and for the computation of
        Riesz representatives of the residuals. If `None`, the Euclidean product is used.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound for the coercivity
        constant of the given problem. Note that the computed error estimate is only
        guaranteed to be an upper bound for the error when an appropriate coercivity
        estimate is specified.
    """
    
    def __init__(self, fom, RB=None, dual_RB=None, product=None, coercivity_estimator=None,
            check_orthonormality=None, check_tol=None):
        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality, check_tol=check_tol)

        self.residual_reductor = ResidualReductor(self.bases['RB'], self.fom.operator, self.fom.rhs,
                                                  product=product, riesz_representatives=True)
        assert fom.output_functional and not fom.output_function
        assert fom.output_functional.linear
        dual_fom = fom.with_(operator=fom.operator ,# simply keep operator due to symmetry
                             rhs=-1*fom.output_functional.H,
                             products=None,
                             output_functional=None)
        self.dual = CoerciveRBReductor(dual_fom, product=product, coercivity_estimator=coercivity_estimator,
                check_orthonormality=check_orthonormality, check_tol=check_tol)
        self.bases['dual RB'] = self.dual.bases['RB']
        self.__auto_init(locals())
        self._output_correction_operator = (None, None, None)
        self._output_correction_rhs = (None, None)

    def project_operators(self):
        projected_operators = super().project_operators()
        for kk, op in self.dual.project_operators().items():
            projected_operators[f'dual_{kk}'] = op
        if not self._output_correction_operator or \
                self._output_correction_operator[0] != len(self.bases['RB']) or \
                self._output_correction_operator[1] != len(self.bases['dual RB']):
            self._output_correction_operator = (len(self.bases['RB']), len(self.bases['dual RB']), project(
                self.fom.operator,
                range_basis=self.bases['dual RB'], source_basis=self.bases['RB']))
        projected_operators['output_correction_operator'] = self._output_correction_operator[-1]
        if not self._output_correction_rhs or \
                self._output_correction_rhs[0] != len(self.bases['dual RB']):
            self._output_correction_rhs = (len(self.bases['dual RB']), project(
                self.fom.rhs,
                range_basis=self.bases['dual RB'],
                source_basis=None))
        projected_operators['output_correction_rhs'] = self._output_correction_rhs[-1]
        return projected_operators
                                                                                                                        
    def project_operators_to_subbasis(self, dims):                                                                      
        projected_operators = super().project_operators_to_subbasis(dims)
        for kk, op in self.dual.project_operators_to_subbasis({'RB': dims['dual RB']}).items():
            projected_operators[f'dual_{kk}'] = op
        if not self._output_correction_operator or not self._output_correction_rhs:
            self.project_operators()
        projected_operators['output_correction_operator'] = project_to_subbasis(
            self._output_correction_operator[-1],
            dim_range=dims['dual RB'], dim_source=dims['RB'])
        projected_operators['output_correction_rhs'] = project_to_subbasis(
            self._output_correction_rhs[-1],
            dim_range=dims['dual RB'])
        return projected_operators
        
    def assemble_estimator(self):
        dual_rom = self.dual.reduce()
        residual = self.residual_reductor.reduce()
        dual_residual = self.dual.residual_reductor.reduce()
        return SymmetricCoercivePrimalDualRBEstimator(
            dual_rom,
            residual, tuple(self.residual_reductor.residual_range_dims),
            dual_residual, tuple(self.dual.residual_reductor.residual_range_dims),
            self.coercivity_estimator)

    def assemble_estimator_for_subbasis(self, dims):
        residual_range_dims = tuple(self.residual_reductor.residual_range_dims)[:dims['RB'] + 1]
        residual = self.residual_reductor.reduce().projected_to_subbasis(residual_range_dims[-1], dims['RB'])
        dual_residual_range_dims = tuple(self.dual.residual_reductor.residual_range_dims)[:dims['dual RB'] + 1]
        dual_residual = self.dual.residual_reductor.reduce().projected_to_subbasis(
                dual_residual_range_dims[-1], dims['dual RB'])
        dual_rom = self.dual.reduce(dims={'RB': dims['dual RB']})
        return SymmetricCoercivePrimalDualRBEstimator(
            dual_rom,
            residual, tuple(self.residual_reductor.residual_range_dims),
            dual_residual, tuple(self.dual.residual_reductor.residual_range_dims),
            self.coercivity_estimator)

    def build_rom(self, projected_operators, estimator):
        projected_dual_operators = []
        for kk in projected_operators.keys():
            if len(kk) > 5 and kk[:5] == 'dual_':
                projected_dual_operators.append(kk)
        projected_dual_operators = {key[5:]: projected_operators.pop(key) for key in projected_dual_operators}
        output_correction_operator = projected_operators.pop('output_correction_operator')
        output_correction_rhs = projected_operators.pop('output_correction_rhs')
        rom = StationaryModel(parameter_space=self.fom.parameter_space, estimator=estimator, **projected_operators)
        dual_rom = estimator.dual_rom
        output_functional = projected_operators['output_functional']

        def corrected_output(U, mu):
            uncorrected_output = output_functional.apply(U, mu=mu)
            Q = dual_rom.solve(mu)
            result = uncorrected_output.to_numpy() \
                    - output_correction_rhs.H.apply(Q, mu=mu).to_numpy() \
                    + output_correction_operator.apply2(Q, U, mu=mu)
            return rom.output_space.from_numpy(result)

        return rom.with_(output_function=corrected_output)


class SymmetricCoercivePrimalDualRBEstimator(ImmutableInterface):                                                                          
    """Instantiated by :class:`SymmetricCoercivePrimalDualRBReductor`.

    Not to be used directly.
    """

    def __init__(self, dual_rom,
                 residual, residual_range_dims,
                 dual_residual, dual_residual_range_dims,
                 coercivity_estimator):
        self.__auto_init(locals())

    def estimate(self, U, mu, m):
        est = self.residual.apply(U, mu=mu).l2_norm()
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)
        return est
                                                                                                                        
    def output_error(self, mu, m):
        U = m.solve(mu=mu)
        est = self.residual.apply(U, mu=mu).l2_norm()
        Q = self.dual_rom.solve(mu=mu)
        est *= self.dual_residual.apply(Q, mu=mu).l2_norm()
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)
        return est


class PrimalDualRBSurrogate(WeakGreedySurrogate):
    """Surrogate for the :func:`~pymor.algorithms.greedy.weak_greedy` error.

    See :class:`SymmetricCoercivePrimalDualRBReductor` for more information.

    Given a `fom`, a `training_set`, a `product` and corresponding `coercivity_estimator`,
    this can be used as in ::

        reductor = SymmetricCoercivePrimalDualRBReductor(
            fom, product=product, coercivity_estimator=coercivity_estimator)
        rb_surrogate = PrimalDualRBSurrogate(reductor)

        greedy_data = weak_greedy(rb_surrogate, training_set)

        greedy_data['rom'] = reductor.reduce()
        greedy_data['dual_rom'] = reductor.dual.reduce()
    """

    def __init__(self, reductor, use_estimator='OUTPUT', extension_params=None, pool=None):
        assert use_estimator in ('OUTPUT', 'STATE', 'DUAL')
        pool = pool or dummy_pool
        self.__auto_init(locals())
        self.rom = None
        self.dual_rom = None

    def evaluate(self, mus, return_all_values=False):
        if self.rom is None or self.dual_rom is None:
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()
                self.dual_rom = self.reductor.dual.reduce()

        if not isinstance(mus, RemoteObjectInterface):
            mus = self.pool.scatter_list(mus)

        result = self.pool.apply(_primal_dual_rb_surrogate_evaluate,
                                 rom=self.rom,
                                 dual_rom=self.dual_rom,
                                 use_estimator=self.use_estimator,
                                 mus=mus,
                                 return_all_values=return_all_values)
        if return_all_values:
            return np.hstack(result)
        else:
            errs, max_err_mus = list(zip(*result))
            max_err_ind = np.argmax(errs)
            return errs[max_err_ind], max_err_mus[max_err_ind]

    def extend(self, mu):
        with self.logger.block(f'Computing (primal) solution snapshot for mu = {mu} ...'):
            U = self.reductor.fom.solve(mu)
        with self.logger.block('Extending (primal) basis with (primal) solution snapshot ...'):
            extension_params = self.extension_params
            if len(U) > 1 and extension_params is None:
                extension_params = {'method': 'pod'}
            self.reductor.extend_basis(U, copy_U=False, **(extension_params or {}))
        with self.logger.block(f'Computing dual solution snapshot for mu = {mu} ...'):
            Q = self.reductor.dual.fom.solve(mu)
        with self.logger.block('Extending dual basis with dual solution snapshot ...'):
            extension_params = self.extension_params
            if len(Q) > 1 and extension_params is None:
                extension_params = {'method': 'pod'}
            self.reductor.dual.extend_basis(Q, copy_U=False, **(extension_params or {}))
        with self.logger.block('Reducing ...'):
            self.rom = self.reductor.reduce()
            self.dual_rom = self.reductor.dual.reduce()


def _primal_dual_rb_surrogate_evaluate(rom, dual_rom, use_estimator, mus, return_all_values):
    if not mus:
        if return_all_values:
            return []
        else:
            return -1., None

    assert use_estimator in ('OUTPUT', 'STATE', 'DUAL')
    if use_estimator == 'OUTPUT':
        errors = [rom.output_error(mu) for mu in mus]
    elif use_estimator == 'STATE':
        errors = [rom.estimate(rom.solve(mu), mu) for mu in mus]
    else: # 'DUAL'
        errors = [dual_rom.estimate(dual_rom.solve(mu), mu) for mu in mus]
    # most error_norms will return an array of length 1 instead of a number, so we extract the numbers
    # if necessary
    errors = [x[0] if hasattr(x, '__len__') else x for x in errors]
    if return_all_values:
        return errors
    else:
        max_err_ind = np.argmax(errors)
        return errors[max_err_ind], mus[max_err_ind]

