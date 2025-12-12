"""Example of complex EP3.

Based on example described in
E. Perrey-Debain, B. Nennig et J. Lawrie. Mode coalescence and the green’s
function in a two-dimensional waveguide with arbitrary admittance boundary condi-
tions. Journal of Sound and Vibration :116510, 2021. issn : 0022-460X. doi : 10.1
016/j.jsv.2021.116510.

Consider the following waveguide problem
```               nu
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

                |||-->

    ---------------------------
                  mu
```
The aim is to find the first EP3.


Examples
--------
>>> error = main() # doctest: +ELLIPSIS
Convergence in ...
>>> error < 1e-6
True
"""

import numpy as np
import sympy as sym

from dispyrsion import Disp

# Reference values from 10.1016/j.jsv.2021.116510
# s_ref = [alpha3_ref, mu_ref, nu_ref]
s_ref = [4.196938882631 - 2.608641537894j, 3.178162507266 + 4.675180387634j,
         3.087536291490 + 3.623417922461j]


def main():
    """Run the global computation and return error wrt to analytic solution."""
    # Define symbolic problem
    alpha, nu, mu = sym.symbols('alpha, nu, mu')
    eq = (nu + mu) * sym.cos(alpha) + (alpha - mu * nu / alpha) * sym.sin(alpha)

    # Variable parameters
    params = (alpha, mu, nu)
    # Constant parameters and values
    # None, use []
    # Define the solver object
    pb = Disp(eq, params, const=[], const_values=[])
    # Locate real EP2
    z, status = pb.locate_ep(np.array((4., -2,
                                        3., 4.,
                                        3., 3.)), real_param_ep=False)
    # Check error with reference solution
    err = np.linalg.norm(z - s_ref)
    print(f'Error wrt ref solution = {err}')
    return err


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=1)
