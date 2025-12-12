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
r"""
Consider the following 3 DOF system.
```
           +~~~~~~~~~~+                +~~~~~~~~~~+               +~~~~~~~~~~+
           |          |                |          |               |          |
     mu    |          |        k       |          |        k      |          |     nu
X---/\/\---|    m1    | -----/\/\----- |    m2    | -----/\/\-----|    m3    | ---/\/\---X
           |          |                |          |               |          |
           +~~~O~~~O~~+                +~~~O~~~O~~+               +~~~O~~~O~~+
                 └-> x1                      └-> x2                     └-> x3
```
depending on the two complex parameters mu and nu. We would like to find the EP3.

Examples
--------
>>> error = main() # doctest: +ELLIPSIS
Convergence in ...
>>> error < 1e-5
True

@author: bn
"""
import numpy as np
import sympy as sym
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from dispyrsion import Disp, to_x, to_z

# Reference solution
sol_ana = np.array([[2. - 1.77635684e-15j, 1. + 1.41421356e+00j, 1. - 1.41421356e+00j],
                    [2. + 1.77635684e-15j, 1. - 1.41421356e+00j, 1. + 1.41421356e+00j],
                    [2. - 1.73205081e+00j, 0.5-2.59807621e+00j, 1.5-2.59807621e+00j],
                    [2. + 1.73205081e+00j, 0.5+2.59807621e+00j, 1.5+2.59807621e+00j],
                    [2. - 1.73205081e+00j, 1.5-2.59807621e+00j, 0.5-2.59807621e+00j],
                    [2. + 1.73205081e+00j, 1.5+2.59807621e+00j, 0.5+2.59807621e+00j]])


def error_between(sol1, sol2):
    """Evalute the error between two sets of unordered vectors.

    Parameters
    ----------
    sol1 : np.array
        Array wih m1 lines of n-dimentional solution.
    sol1 : np.array
        Array wih m2 lines of n-dimentional solution.

    Returns
    -------
    error : float
        The global error between the two set using the best permutation.
    """
    D = distance_matrix(sol1, sol2)
    row_ind, col_ind = linear_sum_assignment(D)
    error = D[row_ind, col_ind].sum()
    return error


def main():
    """Run the global computation and return error wrt to analytic solution."""
    mu, nu, lda, k, m = sym.symbols('mu, nu, lambda, k, m', complex=True)

    M = - sym.Matrix([[m, 0, 0],
                      [0, m, 0],
                      [0, 0, m]])
    K = sym.Matrix([[mu+k, -k, 0.],
                    [-k, 2*k, -k],
                    [0., -k, nu+k]])
    # Symbols
    p0 = sym.det(K + lda*M)
    # Ensure the polynomial has the good sign
    a3 = p0.coeff(lda, 3)
    eq = p0 / a3

    # Variable parameters
    params = (lda, mu, nu)
    # Constant parameters and values
    const = (m, k)
    const_values = (1., 1.)
    # Define the solver object
    pb = Disp(eq, params, const=const, const_values=const_values)
    # Locate all complex EP3
    sol = np.zeros_like(sol_ana)
    err_x0 = np.zeros(sol_ana.shape[0])
    for i, z0 in enumerate(sol_ana):
        x0 = to_x(z0) + 0.5 * np.random.rand(2 * z0.size)
        sol[i, :], status = pb.locate_ep(x0, real_param_ep=False)
        # keep error on initial solution
        err_x0[i] = np.linalg.norm(to_z(x0) - z0)
    # Check error with reference solution
    error = error_between(sol, sol_ana)

    print('Found EP3:')
    print(sol)
    print('Error between found EP and analytic solution: ', error)
    return error


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=1)
