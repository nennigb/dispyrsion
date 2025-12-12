#!/usr/bin/env python3
"""Duct with 2 wiremesh

This example based on
Higher mode filtering: optimum attenuation in a continuum of exceptional points.
Jane B. Lawrie and Muhammad Afzal

The aim is to find real valued parameters (C1, C2, b, d)) leading to EP3 in the infinite lined duct,
```
^ y
|
  =========================================

b  ..................C2....................

d -------------------C1--------------------

  ==========================================   ---> x
```


Examples
--------
>>> error = main() # doctest: +ELLIPSIS
Starting...
>>> error < 1e-6
True

"""
import numpy as np
import sympy as sym

from dispyrsion import Disp

# Reference values from the paper
# alpha, C1, C2, b, d
s_ref1 = [(3.4625689109189177-1.6654536845259664j), 0.5425099753227288, 0.7446671935748486,
          0.8436930252475578, 0.1334146356319723]
s_ref2 = [(3.5878834407898257-1.6572624988673055j), 0.3517342656498985, 1.473222637198806,
          0.9436137878525618, 0.7230139468206018]


def main(find_all=True):
    """Run the global computation and return error wrt to analytic solution."""
    # Define symbolic problem
    # Create
    alp, C1, C2, b, d = sym.symbols('alp, C1, C2, b, d')
    I = sym.I
    # Eq. 2 - 4
    Omega = I * C1 * alp * sym.sin(alp * d) * sym.sin(alp * (b - d)) - sym.sin(alp * b)
    H = I * C1 * alp * sym.sin(alp * d) * sym.cos(alp * (b - d)) - sym.cos(alp * b)
    Delta = I * C2 * alp * sym.sin(alp * (1 - b)) - sym.cos(alp * (1 - b))

    # Eq. 1
    K = Omega * Delta - sym.sin(alp*(1 - b)) * H
    # Variable parameters
    params = (alp, C1, C2, b, d)
    # Constant parameters and values
    # None, use []
    # Define the solver object
    pb = Disp(K, params, const=[], const_values=[])

    # Find 'most' EP3 in a region, n_ig_per_dir=3 would be better, but too long
    # for tests
    print('Starting')
    if find_all:
        eps = pb.locate_eps([(2-2j, 4-1j),
                             (0.3, 2.),
                             (0.3, 2.),
                             (0.5, 0.9),
                             (0.1, 0.5)], n_ig_per_dir=2, method='lm', tolf=1e-10,
                            real_param_ep=True, unique_tol=1e-3, niter_max=100)
        # Check error with the two reference solution
        id1 = np.argmin(abs(eps[:, 0] - s_ref1[0]))
        id2 = np.argmin(abs(eps[:, 0] - s_ref2[0]))
        err = max(np.linalg.norm(eps[id1, :] - s_ref1),
                  np.linalg.norm(eps[id2, :] - s_ref2))
    else:
        # Find an real EP3 using alpha, C1, C2, b, d from a single initial guess
        z, status = pb.locate_ep(np.array((3., -2,
                                           .5, 0.,
                                           .8, 0,
                                           .8, 0,
                                           .2, 0.)), real_param_ep=True)
        # Check error with reference solution
        err = np.linalg.norm(z - s_ref1)

    print(f'Error wrt ref solution = {err}')
    return err


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=1)
    # err = main(find_all=True)
