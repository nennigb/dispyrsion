
# This file is part of dispyrsion, A library to locate exceptional points
# from analytical dispersion equations.
# Copyright (C) <2025> <benoit.nennig@isae-supmeca.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import itertools as it
from functools import lru_cache, partial

import numpy as np
import sympy as sym
from polze import PZ
from scipy.optimize import root
from tqdm import tqdm


def _sqrt(z):
    """Define complex square root for lambdifying with numpy."""
    return np.sqrt(z, dtype=complex)


libs = [{'sqrt': _sqrt}, 'numpy']
PZ_OPTIONS = {'_zeros_only': True,
              '_vectorized': True,
              '_Npz_limit': 12,
              '_tol': 1e-4}


class Disp:
    """Define the class to solve dispersion equation defined from sympy equations."""

    def __init__(self, eq, params, const, const_values, nparams=None):
        """Initialize all equations from sympy.

        param_list: list
            List of Param object. The first is intented to be the eigenvalue.
        const: list
            The list of constant parameters. If none, use empty list.
        const_value: list
            The list of constant parameters values. If none, use empty list.
        nparams: int, optional
            The default is `None` and this quantity is infered from params size.
        """
        index = 0
        self.eig_symbol = params[index]
        self.eig_name = self.eig_symbol.__repr__()
        self.const = const
        self.const_values = const_values
        self.params = params
        # Number of parameters
        if nparams is None:
            self.nparams = len(params) - 1
        else:
            self.nparams = nparams
        self.eq = eq
        self._init_eqs()

    def __repr__(self):
        """Return the representation of the instance."""
        return f"{self.__class__.__name__} object with params={self.params} and const={dict(zip(self.const, self.const_values))}."

    def _init_eqs(self):
        """Initialize all equations from sympy."""
        self.K = []
        self.eval_K = []
        K = self.eq.subs(zip(self.const, self.const_values))
        lda = self.eig_symbol
        # Store all eqs
        for i in range(self.nparams + 1):
            self.K.append(sym.diff(K, (lda, i)))
            self.eval_K.append(sym.lambdify(self.params, self.K[i], libs, cse=True))

    def eval_Ki_at(self, i, p):
        """Create the dispersion equation function.

        Parameters
        ----------
        i: int
            The order derivative of the dispersion equation with respect to the
            eigenvalue. If 0, the original equation without derivation is used.
        p: dict
            Paramters are described with dictionnary. The keys are sympy variable,
            and the value is the variable value. Note that the sympy variable
            are converted into string as done in the lambdify process.
            If the conversion to string yields to a non valid python name, error
            may ocured. An example is with 'lambda', in this case use
            `lda = sympy.symbols('lambda_')`.
        """
        return partial(self.eval_K[i], **{str(k): v for k, v in p.items()})

    def solve_K(self, p, Rmax=15, Npz=5, Ni=5000, display=True, options={}, refine=True, raise_error=True,
                multiplicities=False):
        """Solve the dispersion equation."""
        f = self.eval_Ki_at(0, p)
        df = self.eval_Ki_at(1, p)
        # Update default options with the provided ones
        options_ = PZ_OPTIONS.copy()
        options_.update(options)
        pz = PZ((f, df), Rmax=Rmax, Npz=Npz, Ni=Ni, options=options_)
        pz.solve()
        # Stop if multiplicities are not computed below the tolerance
        if raise_error:
            if pz.status['multiplicities'] != 0:
                raise ValueError('Problem in eigenvalues computation. Increase the number of splits')

        if refine:
            pz.iterative_ref()
        if display:
            pz.display()

        if multiplicities:
            _, (z, m) = pz.dispatch(multiplicities=True)
            return pz, z, m
        else:
            _, z = pz.dispatch(multiplicities=False)
            return pz, z

    @lru_cache
    def build_jacobian2(self):
        """Build jacobian function for EP2 case."""
        J = np.empty((self.nparams + 1, self.nparams + 1), dtype=object)
        Jsym = np.empty((self.nparams + 1, self.nparams + 1), dtype=object)
        F = np.empty((self.nparams + 1,), dtype=object)
        # Jacobian
        for (i, j), _ in np.ndenumerate(J):
            Jsym[i, j] = sym.diff(self.K[i], self.params[j])
            J[i, j] = sym.lambdify(self.params, Jsym[i, j], libs, cse=True, docstring_limit=0)
        # System
        for i in range(self.nparams+1):
            F[i] = sym.lambdify(self.params, self.K[i], libs, cse=True, docstring_limit=0)
        return J, F

    def locate_ep(self, x, niter_max=200, method='lm', tolf=1e-10, tolx=1e-12, imag_scaling=1e5,
                  real_param_ep=False):
        """Find a single EP from an intial guess.

        Parameters
        ----------
        x : array
            The initial guess, split real and imaginary part.
        niter_max : int, optional
            The number max of NR iteration. The default is 50.
        tolf : float, optional
            The tolerance to stop NR iertation. This tol is compared to the norm of the
            function. The default is 1e-4.
        method: string
            Provides the solver type {'lm'}. Use scipy Levenberd-Marquard solver.
        tolx : float, optional
            The tolerance to stop NR iertation. This tol is compared to the norm of the
            residual. The default is 1e-4.
        imag_scaling: float
            A constant used as scale factors in 'lm' method to impose constraint
            on the imaginary part. The defaut is 1, but it should be increase if
            complex solution are return. 1e3 seems generaly a good compromise.
         real_param_ep : bool
             If `True`, the solver try to find an EP of order (n/2 + 1) while
             keeping the n parameters real. Since the current implementation require
             Jacobian matrix targeting higher order EP with complex value coefficients,
             the number of eigenvalue in the PCP must be equal to the number of
             parameter + 1.
        """
        # Initialisation
        Jc_lda, Fc_lda = self.build_jacobian2()
        return self._locate_ep(x, Jc_lda, Fc_lda,
                               niter_max=niter_max, method=method, tolf=tolf, tolx=tolx,
                               imag_scaling=imag_scaling, real_param_ep=real_param_ep)

    @staticmethod
    def _locate_ep(x, Jc_lda, Fc_lda,
                   niter_max=200, method='lm', tolf=1e-10, tolx=1e-12, imag_scaling=1e5,
                   real_param_ep=False):
        """Implement the EP solver.

        Parameters
        ----------
        Jc_lda: lambdify
            The jacobian matrix.
        Fc_lda: lambdify
            The RHS function.
        All other parameters are described above.

        Remarks
        -------
        This split allows for pickeling the solver for parallel execution.
        """
        # Initialisation
        # x = np.array(x)
        method = method.lower()
        nparam = x.size // 2

        def f_and_jac(x):
            """Evaluate the function and the jacobian matrix."""
            # Compute the Jac and the function from successive derivative of the
            # Dispersion equation
            nparam = x.size // 2
            if real_param_ep:
                # With n real parameters, target EP of order {(n/2) + 1}
                nparam = nparam // 2 + 1
            z = to_z(x)
            # Eval Fc
            Fc = np.zeros(Fc_lda.size, dtype=complex)
            for i, f in enumerate(Fc_lda):
                Fc[i] = f(*z)
            # Eval Jc
            Jc = np.zeros(Jc_lda.shape, dtype=complex)
            for index, j in np.ndenumerate(Jc_lda):
                Jc[index] = j(*z)

            # Compute the real version
            Jr = np.zeros((x.size, x.size))
            Fr = np.zeros((x.size,))
            # Add terms from p dans d_lda p
            for eq_id in range(nparam):
                eq_idr = eq_id * 2
                Fr[eq_idr] = Fc[eq_id].real
                Fr[eq_idr + 1] = Fc[eq_id].imag
                for i in range(x.size // 2):
                    ir = i * 2
                    # Real part eqs
                    Jr[eq_idr, ir] = Jc[eq_id, i].real
                    Jr[eq_idr, ir+1] = - Jc[eq_id, i].imag
                    # Imag part eqs
                    Jr[eq_idr+1, ir] = Jc[eq_id, i].imag
                    Jr[eq_idr+1, ir+1] = Jc[eq_id, i].real

            if real_param_ep:
                # Add Im nu = 0
                for i, eq_idr in enumerate(range(eq_idr + 2, eq_idr + 2 + (x.size//2 - 1))):
                    ir = 2*i + 3
                    Fr[eq_idr] = x[ir]
                    Jr[eq_idr, ir] = 1.  # imag
            return Fr, Jr
        # See also hybr,s ometimes beter ?
        scaling = np.ones_like(x)
        scaling[(nparam//2 + 2):] = imag_scaling
        sol = root(f_and_jac, x, method='lm', jac=True, callback=None,
                   options={'maxiter': niter_max, 'xtol': tolx, 'ftol': tolf, })
        # 'diag': scaling})  # {}'xtol': 0.00000001,
        z = to_z(sol.x)
        x = sol.x
        if sol.success:
            print(f'Convergence in {sol.nfev} iterations.', sol.message)
        else:
            print(sol.message)
        return z, sol.success

    def locate_eps(self, bounds, n_ig_per_dir=4, niter_max=200, method='lm', tolf=1e-10,
                   tolx=1e-12, imag_scaling=1e5, real_param_ep=False, unique_tol=1e-3):
        """Find most EPs in a region.

        Parameters
        ----------
        bounds : iterable
            Bounds used to generate the initials guess. Each item must contains the 2 corners
            in the complex plane. For instance, if bounds = [(-1-1j, 1+1j), (-2-2j, 2+2j)],
        n_ig_per_dir: int
            The number of initial guess per direction. If bounds are purely real or imaginary,
            `n_ig_per_dim` are used. If bounds are complex, `n_ig_per_dim` are used along the
            real and imaginary direction.
        niter_max : int, optional
            The number max of NR iteration. The default is 50.
        method: string
            Provides the solver type {'lm'}. Use scipy Levenberd-Marquard solver.
        tolf : float, optional
            The tolerance to stop NR iertation. This tol is compared to the norm of the
            function. The default is 1e-4.
        tolx : float, optional
            The tolerance to stop NR iertation. This tol is compared to the norm of the
            residual. The default is 1e-4.
        imag_scaling: float
            A constant used as scale factors in 'lm' method to impose constraint
            on the imaginary part. The defaut is 1, but it should be increase if
            complex solution are return. 1e3 seems generaly a good compromise.
         real_param_ep : bool
             If `True`, the solver try to find an EP of order (n/2 + 1) while
             keeping the n parameters real. Since the current implementation require
             Jacobian matrix targeting higher order EP with complex value coefficients,
             the number of eigenvalue in the PCP must be equal to the number of
             parameter + 1.
        unique_tol: float
            The tolerance used to filter solution find several times.
        """
        # Initialisation
        method = method.lower()
        grid = make_grid(bounds, n_ig_per_dir)
        all_sols = []
        total = np.prod([t.size for t in grid])
        pbar = tqdm(total=total)
        Jc_lda, Fc_lda = self.build_jacobian2()

        work = partial(self._locate_ep, Jc_lda=Jc_lda, Fc_lda=Fc_lda,
                       niter_max=niter_max, method=method, tolf=tolf, tolx=tolx,
                       imag_scaling=imag_scaling, real_param_ep=real_param_ep)
        # Hard to parallelize with concurrent.futures because sympy lamdify
        # are not picklizable... perhaps with dill.
        for i, p in enumerate(it.product(*grid)):
            z, status = work(to_x(np.array(p)))
            pbar.update(1)
            if status:
                all_sols.append(z)
        all_sols = np.array(all_sols)
        # Filter solutions to keep only the unique ones
        decimals = - int(round(np.log10(unique_tol)))
        # Use unique to remove duplicated row
        _, indu = np.unique(all_sols.round(decimals=decimals), axis=0, return_index=True)
        filtered_sols = all_sols[indu, :]
        return filtered_sols


def param_mesh(bound, Npts):
    """Create 1D grid between bounds.

    Parameters
    ----------
    bound: iterable
        It contains the 2 corners in the complex plane. For instance (-1-1j, 1+1j).
        If the 2 corners are purelly real-valued or imaginary only this direction is
        meshed.
    Npts: int
        The number of initial guess per direction. If bounds are purely real or imaginary,
        `Npts` are used. If bounds are complex, `Npts` are used along the
        real and imaginary direction.

    Examples
    --------
    >>> z = param_mesh((0, 1), 3)
    >>> len(z) == 3, sum(abs(z.imag)) == 0
    (True, True)
    >>> z = param_mesh((0, 1j), 3)
    >>> len(z) == 3, sum(abs(z.real)) == 0
    (True, True)
    >>> z = param_mesh((0, 1+1j), 3)
    >>> len(z) == 9
    True
    """
    if (bound[0].imag == 0) & (bound[1].imag == 0):
        Re, Im = np.linspace(bound[0].real, bound[1].real, Npts), 0
    elif (bound[0].real == 0) & (bound[1].real == 0):
        Im, Re = np.linspace(bound[0].imag, bound[1].imag, Npts), 0
    else:
        Re, Im = np.meshgrid(np.linspace(bound[0].real, bound[1].real, Npts),
                             np.linspace(bound[0].imag, bound[1].imag, Npts))
    z = Re + 1j * Im
    return z.ravel()


def make_grid(bounds, Npts):
    """Create bounds.

    Parameters
    ----------
    bounds : iterable
        Each item must contains the 2 corners in the complex plane. For
        instance if bounds = [(-1-1j, 1+1j), (-2-2j, 2+2j)],
    Npts: int
        The number of initial guess per direction. If bounds are purely real or imaginary,
        `Npts` are used. If bounds are complex, `Npts` are used along the
        real and imaginary direction.

    Returns
    -------
    grid: list
        The grid in the parameter space.
    """
    grid = []
    for bound in bounds:
        grid.append(param_mesh(bound, Npts))
    return grid


def to_x(z):
    """Convert the complex unknown `z` toin-line real and imag unknown."""
    return np.stack((z.real, z.imag), axis=1).ravel()


def to_z(x):
    """Convert the in-lined real and imag unknown `x` to complex unknown."""
    return x.reshape((-1, 2)) @ np.array([1, 1j])
