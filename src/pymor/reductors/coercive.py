# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.core.base import ImmutableObject
from pymor.models.basic import StationaryModel, StationaryPrimalDualModel
from pymor.operators.constructions import LincombOperator, induced_norm
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interface import RemoteObject
from pymor.reductors.basic import StationaryRBReductor, StationaryPGRBReductor
from pymor.reductors.residual import ResidualReductor
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class CoerciveRBReductor(StationaryRBReductor):
    """Reduced Basis reductor for |StationaryModels| with coercive linear operator.

    The only addition to :class:`~pymor.reductors.basic.StationaryRBReductor` is an error
    estimator which evaluates the dual norm of the residual with respect to a given inner
    product and which provides the output estimates as in the primal RB approach
    [Haa2017]_ (section 2.3). For the reduction of the residual we use
    :class:`~pymor.reductors.residual.ResidualReductor` for improved numerical stability
    [BEOR14]_.

    Parameters
    ----------
    fom
        The |Model| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    product
        Inner product for the orthonormalization of `RB`, the projection of the
        |Operators| given by `vector_ranged_operators` and for the computation of
        Riesz representatives of the residual. If `None`, the Euclidean product is used.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound for the coercivity
        constant of the given problem. Note that the computed error estimate is only
        guaranteed to be an upper bound for the error when an appropriate coercivity
        estimate is specified.
    rhs_continuity_estimator
        A bound for the continuity constant of the right hand side of the given problem
        (similar to `coercivity_estimator`), used for the output estimation in the
        general case (i.e. when `compliant == False`).
    compliant
        Boolean value to specify whether the output functional of the model coincides
        with the models right hand side (`True`, compliant case) or not (`False`).
        In the compliant case, Proposition 2.19 in [Haa2017]_ is used while Proposition 2.24
        in [Haa2017]_ is used in the general case.
    """

    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None, rhs_continuity_estimator=None,
                 compliant=False, check_orthonormality=None, check_tol=None):
        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol)
        self.residual_reductor = ResidualReductor(self.bases['RB'], self.fom.operator, self.fom.rhs,
                                                  product=product, riesz_representatives=True)
        self.__auto_init(locals())

    def assemble_estimator(self):
        residual = self.residual_reductor.reduce()
        estimator = CoerciveRBEstimator(residual, tuple(self.residual_reductor.residual_range_dims),
                                        self.coercivity_estimator, self.compliant, self.rhs_continuity_estimator)
        return estimator

    def assemble_estimator_for_subbasis(self, dims):
        return self._last_rom.estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class CoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`CoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, residual, residual_range_dims, coercivity_estimator, compliant, rhs_continuity_estimator):
        self.__auto_init(locals())

    def estimate(self, U, mu, m):
        est = self.residual.apply(U, mu=mu).l2_norm()
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)
        return est

    def output_error(self, mu, m):
        U = m.solve(mu=mu)
        est = self.residual.apply(U, mu=mu).l2_norm()
        if self.compliant and self.coercivity_estimator:
            return est**2 / self.coercivity_estimator(mu)
        elif self.compliant:
            return est**2
        elif self.rhs_continuity_estimator:
            return est * self.rhs_continuity_estimator(mu)
        else:
            return est

    def restricted_to_subbasis(self, dim, m):
        if self.residual_range_dims:
            residual_range_dims = self.residual_range_dims[:dim + 1]
            residual = self.residual.projected_to_subbasis(residual_range_dims[-1], dim)
            return CoerciveRBEstimator(residual, residual_range_dims, self.coercivity_estimator, self.compliant,
                    self.rhs_continuity_estimator)
        else:
            self.logger.warning('Cannot efficiently reduce to subbasis')
            return CoerciveRBEstimator(self.residual.projected_to_subbasis(None, dim), None,
                                       self.coercivity_estimator, self.compliant, self.rhs_continuity_estimator)


class SimpleCoerciveRBReductor(StationaryRBReductor):
    """Reductor for linear |StationaryModels| with affinely decomposed operator and rhs.

    .. note::
       The reductor :class:`CoerciveRBReductor` can be used for arbitrary coercive
       |StationaryModels| and offers an improved error estimator
       with better numerical stability.

    The only addition is to :class:`~pymor.reductors.basic.StationaryRBReductor` is an error
    estimator, which evaluates the norm of the residual with respect to a given inner product
    and which provides the output estimates as in the primal RB approach [Haa2017]_ (section 2.3).

    Parameters
    ----------
    fom
        The |Model| which is to be reduced.
    RB
        |VectorArray| containing the reduced basis on which to project.
    product
        Inner product for the orthonormalization of `RB`, the projection of the
        |Operators| given by `vector_ranged_operators` and for the computation of
        Riesz representatives of the residual. If `None`, the Euclidean product is used.
    coercivity_estimator
        `None` or a |Parameterfunctional| returning a lower bound for the coercivity
        constant of the given problem. Note that the computed error estimate is only
        guaranteed to be an upper bound for the error when an appropriate coercivity
        estimate is specified.
    rhs_continuity_estimator
        A bound for the continuity constant of the right hand side of the given problem
        (similar to `coercivity_estimator`), used for the output estimation in the
        general case (i.e. when `compliant == False`).
    compliant
        Boolean value to specify whether the output functional of the model coincides
        with the models right hand side (`True`, compliant case) or not (`False`).
        In the compliant case, Proposition 2.19 in [Haa2017]_ is used while Proposition 2.24
        in [Haa2017]_ is used in the general case.
    """

    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None, rhs_continuity_estimator=None,
                 compliant=False, check_orthonormality=None, check_tol=None):
        assert fom.operator.linear and fom.rhs.linear
        assert isinstance(fom.operator, LincombOperator)
        assert all(not op.parametric for op in fom.operator.operators)
        if fom.rhs.parametric:
            assert isinstance(fom.rhs, LincombOperator)
            assert all(not op.parametric for op in fom.rhs.operators)

        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol)
        self.residual_reductor = ResidualReductor(self.bases['RB'], self.fom.operator, self.fom.rhs,
                                                  product=product)
        self.extends = None
        self.__auto_init(locals())

    def assemble_estimator(self):
        fom, RB, extends = self.fom, self.bases['RB'], self.extends
        if extends:
            old_RB_size = extends[0]
            old_data = extends[1]
        else:
            old_RB_size = 0

        # compute data for estimator
        space = fom.operator.source

        # compute the Riesz representative of (U, .)_L2 with respect to product
        def riesz_representative(U):
            if self.products['RB'] is None:
                return U.copy()
            else:
                return self.products['RB'].apply_inverse(U)

        def append_vector(U, R, RR):
            RR.append(riesz_representative(U), remove_from_other=True)
            R.append(U, remove_from_other=True)

        # compute all components of the residual
        if extends:
            R_R, RR_R = old_data['R_R'], old_data['RR_R']
        elif not fom.rhs.parametric:
            R_R = space.empty(reserve=1)
            RR_R = space.empty(reserve=1)
            append_vector(fom.rhs.as_range_array(), R_R, RR_R)
        else:
            R_R = space.empty(reserve=len(fom.rhs.operators))
            RR_R = space.empty(reserve=len(fom.rhs.operators))
            for op in fom.rhs.operators:
                append_vector(op.as_range_array(), R_R, RR_R)

        if len(RB) == 0:
            R_Os = [space.empty()]
            RR_Os = [space.empty()]
        elif not fom.operator.parametric:
            R_Os = [space.empty(reserve=len(RB))]
            RR_Os = [space.empty(reserve=len(RB))]
            for i in range(len(RB)):
                append_vector(-fom.operator.apply(RB[i]), R_Os[0], RR_Os[0])
        else:
            R_Os = [space.empty(reserve=len(RB)) for _ in range(len(fom.operator.operators))]
            RR_Os = [space.empty(reserve=len(RB)) for _ in range(len(fom.operator.operators))]
            if old_RB_size > 0:
                for op, R_O, RR_O, old_R_O, old_RR_O in zip(fom.operator.operators, R_Os, RR_Os,
                                                            old_data['R_Os'], old_data['RR_Os']):
                    R_O.append(old_R_O)
                    RR_O.append(old_RR_O)
            for op, R_O, RR_O in zip(fom.operator.operators, R_Os, RR_Os):
                for i in range(old_RB_size, len(RB)):
                    append_vector(-op.apply(RB[i]), R_O, RR_O)

        # compute Gram matrix of the residuals
        R_RR = RR_R.dot(R_R)
        R_RO = np.hstack([RR_R.dot(R_O) for R_O in R_Os])
        R_OO = np.vstack([np.hstack([RR_O.dot(R_O) for R_O in R_Os]) for RR_O in RR_Os])

        estimator_matrix = np.empty((len(R_RR) + len(R_OO),) * 2)
        estimator_matrix[:len(R_RR), :len(R_RR)] = R_RR
        estimator_matrix[len(R_RR):, len(R_RR):] = R_OO
        estimator_matrix[:len(R_RR), len(R_RR):] = R_RO
        estimator_matrix[len(R_RR):, :len(R_RR)] = R_RO.T

        estimator_matrix = NumpyMatrixOperator(estimator_matrix)

        estimator = SimpleCoerciveRBEstimator(estimator_matrix, self.coercivity_estimator,
                self.rhs_continuity_estimator, self.compliant)
        self.extends = (len(RB), dict(R_R=R_R, RR_R=RR_R, R_Os=R_Os, RR_Os=RR_Os))

        return estimator

    def assemble_estimator_for_subbasis(self, dims):
        return self._last_rom.estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


class SimpleCoerciveRBEstimator(ImmutableObject):
    """Instantiated by :class:`SimpleCoerciveRBReductor`.

    Not to be used directly.
    """

    def __init__(self, estimator_matrix, coercivity_estimator, rhs_continuty_estimator, compliant):
        self.__auto_init(locals())
        self.norm = induced_norm(estimator_matrix)

    def _estimate_residual(self, U, mu, m):
        if len(U) > 1:
            raise NotImplementedError
        if not m.rhs.parametric:
            CR = np.ones(1)
        else:
            CR = np.array(m.rhs.evaluate_coefficients(mu))

        if not m.operator.parametric:
            CO = np.ones(1)
        else:
            CO = np.array(m.operator.evaluate_coefficients(mu))

        C = np.hstack((CR, np.dot(CO[..., np.newaxis], U.to_numpy()).ravel()))

        return self.norm(NumpyVectorSpace.make_array(C))

    def estimate(self, U, mu, m):
        est = self._estimate_residual(U, mu, m)

        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)

        return est

    def output_error(self, mu, m):
        U = m.solve(mu=mu)
        est = self._estimate_residual(U, mu, m)
        if self.compliant and self.coercivity_estimator:
            return est**2 / self.coercivity_estimator(mu)
        elif self.compliant:
            return est**2
        elif self.rhs_continuity_estimator:
            return est * self.rhs_continuity_estimator(mu)
        else:
            return est


    def restricted_to_subbasis(self, dim, m):
        cr = 1 if not m.rhs.parametric else len(m.rhs.operators)
        co = 1 if not m.operator.parametric else len(m.operator.operators)
        old_dim = m.operator.source.dim

        indices = np.concatenate((np.arange(cr),
                                 ((np.arange(co)*old_dim)[..., np.newaxis] + np.arange(dim)).ravel() + cr))
        matrix = self.estimator_matrix.matrix[indices, :][:, indices]

        return SimpleCoerciveRBEstimator(NumpyMatrixOperator(matrix), self.coercivity_estimator,
                rhs_continuty_estimator, compliant)


class CoercivePrimalDualRBReductor(ImmutableObject):
    """Primal/dual Reduced Basis reductor for |StationaryModels| with dual.

    In addition to :class:`~pymor.reductors.coercive.CoerciveRBReductor`, a dual rom is assembled to
    provide improved output error estimates as well as a corrected reduced output following the primal-dual
    RB approach [Haa2017]_ (section 2.4). For the reduction of the residuals we use
    :class:`~pymor.reductors.residual.ResidualReductor` for improved numerical stability
    [BEOR14]_.

    Parameters
    ----------
    fom
        The (primal/dual) |Model| which is to be reduced.
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

    def __init__(self, fom, primal_RB=None, dual_RB=None, primal_product=None, dual_product=None, coercivity_estimator=None,
            check_orthonormality=None, check_tol=None):
        assert isinstance(fom, StationaryPrimalDualModel)
        self.fom = fom
        self.primal = CoerciveRBReductor(fom.primal, RB=primal_RB, product=primal_product,
                coercivity_estimator=coercivity_estimator, check_orthonormality=check_orthonormality,
                check_tol=check_tol)
        self.dual = CoerciveRBReductor(fom.dual, RB=dual_RB, product=dual_product,
                coercivity_estimator=coercivity_estimator, check_orthonormality=check_orthonormality,
                check_tol=check_tol)
        self.bases = {'primal_RB': self.primal.bases['RB'], 'dual_RB': self.dual.bases['RB']}
        self.coercivity_estimator=coercivity_estimator
        self._last_rom = None
        self.__auto_init(locals())

    def reduce(self, dims=None):
        if dims is None:
            dims = {k: len(v) for k, v in self.bases.items()}
        if isinstance(dims, Number):
            dims = {k: dims for k in self.bases}
        if set(dims.keys()) != set(self.bases.keys()):
            raise ValueError(f'Must specify dimensions for {set(self.bases.keys())}')
        for k, d in dims.items():
            if d < 0:
                raise ValueError(f'Reduced state dimension must be larger than zero {k}')
            if d > len(self.bases[k]):
                raise ValueError(f'Specified reduced state dimension larger than reduced basis {k}')

        if self._last_rom is None or any(dims[b] > self._last_rom_dims[b] for b in dims):
            # build estimator and rom
            primal_rom = self.primal.reduce()
            primal_rom.disable_logging()
            dual_rom = self.dual.reduce()
            dual_rom.disable_logging()
            estimator = CoercivePrimalDualRBEstimator(
                    primal_rom.estimator.with_(coercivity_estimator=None),
                    dual_rom.estimator.with_(coercivity_estimator=None),
                    self.coercivity_estimator)
            self._output_correction_reductor = StationaryPGRBReductor(
                StationaryModel(self.fom.primal.operator, self.fom.primal.rhs),
                range_RB=self.bases['dual_RB'], source_RB=self.bases['primal_RB'],
                check_orthonormality=False)
            output_correction = self._output_correction_reductor.reduce()
            self._last_rom = StationaryPrimalDualModel(primal_rom, dual_rom, estimator=estimator,
                    output_correction_op=output_correction.operator, output_correction_rhs=output_correction.rhs)
            self._last_rom.disable_logging()
            self._last_rom_dims = {k: len(v) for k, v in self.bases.items()}

        if dims == self._last_rom_dims:
            return self._last_rom
        else:
            # build restricted estimator and rom
            primal_rom = self.primal.reduce(dims={'RB': dims['primal_RB']})
            primal_rom.disable_logging()
            dual_rom = self.dual.reduce(dims={'RB': dims['dual_RB']})
            dual_rom.disable_logging()
            estimator = CoercivePrimalDualRBEstimator(
                    primal_rom.estimator.with_(coercivity_estimator=None),
                    dual_rom.estimator.with_(coercivity_estimator=None),
                    self.coercivity_estimator)
            output_correction = self._output_correction_reductor.reduce(
                    dims={'range_RB': dims['dual_RB'], 'source_RB': dims['primal_RB']})
            rom = StationaryPrimalDualModel(primal_rom, dual_rom, estimator=estimator,
                    output_correction_op=output_correction.operator, output_correction_rhs=output_correction.rhs)
            rom.disable_logging()
            return rom

    def reconstruct(self, u, basis=None):
        assert not basis
        assert isinstance(u, BlockVectorArray)
        u, q = u._blocks
        return BlockVectorArray([self.primal.reconstruct(u), self.dual.reconstruct(q)], self.fom.solution_space)


class CoercivePrimalDualRBEstimator(ImmutableObject):
    """Instantiated by :class:`CoercivePrimalDualRBReductor`.

    Not to be used directly.
    """

    def __init__(self, primal_residual_estimator, dual_residual_estimator, coercivity_estimator):
        self.__auto_init(locals())

    def output_error(self, mu, m):
        U, Q = m.solve(mu=mu)._blocks
        est = self.primal_residual_estimator.estimate(U, mu=mu, m=m.primal) \
                * self.dual_residual_estimator.estimate(Q, mu=mu, m=m.dual)
        if self.coercivity_estimator:
            est /= self.coercivity_estimator(mu)
        return est


class PrimalDualRBSurrogate(WeakGreedySurrogate):
    """Surrogate for the :func:`~pymor.algorithms.greedy.weak_greedy` error.

    TODO: does not really belong here, nothing coercive about it

    See :class:`CoercivePrimalDualRBReductor` for more information.

    Given a `fom`, a `training_set`, a `product` and corresponding `coercivity_estimator`,
    this can be used as in ::

        fom = StationaryPrimalDualModel(fom, dual=...)
        reductor = CoercivePrimalDualRBReductor(
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

    def evaluate(self, mus, return_all_values=False):
        if self.rom is None:
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()

        if not isinstance(mus, RemoteObject):
            mus = self.pool.scatter_list(mus)

        result = self.pool.apply(_primal_dual_rb_surrogate_evaluate,
                                 rom=self.rom,
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
        with self.logger.block(f'Computing solution snapshots for mu = {mu} ...'):
            U, Q = self.reductor.fom.solve(mu)._blocks
        with self.logger.block('Extending primal basis with primal solution snapshot ...'):
            extension_params = self.extension_params
            if len(U) > 1 and extension_params is None:
                extension_params = {'method': 'pod'}
            self.reductor.primal.extend_basis(U, copy_U=False, **(extension_params or {}))
        with self.logger.block('Extending dual basis with dual solution snapshot ...'):
            extension_params = self.extension_params
            if len(Q) > 1 and extension_params is None:
                extension_params = {'method': 'pod'}
            self.reductor.dual.extend_basis(Q, copy_U=False, **(extension_params or {}))
        with self.logger.block('Reducing ...'):
            self.rom = self.reductor.reduce()


def _primal_dual_rb_surrogate_evaluate(rom, use_estimator, mus, return_all_values):
    if not mus:
        if return_all_values:
            return []
        else:
            return -1., None

    assert use_estimator in ('OUTPUT', 'STATE', 'DUAL')
    if use_estimator == 'OUTPUT':
        errors = [rom.output_error(mu) for mu in mus]
    elif use_estimator == 'STATE':
        errors = [rom.primal.estimate(rom.primal.solve(mu), mu) for mu in mus]
    else: # 'DUAL'
        errors = [rom.dual.estimate(rom.dual.solve(mu), mu) for mu in mus]
    # most error_norms will return an array of length 1 instead of a number, so we extract the numbers
    # if necessary
    errors = [x[0] if hasattr(x, '__len__') else x for x in errors]
    if return_all_values:
        return errors
    else:
        max_err_ind = np.argmax(errors)
        return errors[max_err_ind], mus[max_err_ind]
