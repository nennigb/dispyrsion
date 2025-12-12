r"""Example of real EP2 obtained from mass-srping-damper system.

Based on example described in
N. Even. Interactions modales au voisinage des points exceptionnels et application
pour l'atténuation acoustique, Université de technologie de Compiègne, 2024.

Consider the following 2 DOF system.
```
           +~~~~~~~~~~+                +~~~~~~~~~~+        c2
           |          |                |          |       _____    |/
     k1    |          |      k12       |          | -----|_‖__-----|/
X---/\/\---|    m1    | -----/\/\----- |    m2    |                |/
           |          |                |          | -----/\/\------|/
           +~~~O~~~O~~+                +~~~O~~~O~~+       k2       |/
                 └-> x1                      └-> x2
```
The aim is to find the real value of c2 and k2 to have an EP2.


The references values are obtained from the continuation strategy implemented in
`real-valued-ep2` (https://github.com/nicolase7en/real-valued-ep2)

Examples
--------
>>> error = main() # doctest: +ELLIPSIS
Convergence in ...
>>> error < 1e-5
True
"""

import numpy as np
import sympy as sym

from dispyrsion import Disp


def main():
    # Reference values from `real-valued-ep2`
    omega_ref, k2_ref, c2_ref = [1j * (-0.049999999998541184-0.9987492177712877j),
                                 0.9100000000002494, 0.19999999999995516]
    # Define symbolic problem
    omega = sym.symbols('omega')
    m1, m2 = sym.symbols(r'm_1 m_2', real=True, positive=True)
    k1, k2, k12 = sym.symbols(r'k_1 k_2 k_{12}', real=True, positive=True)
    c2 = sym.symbols(r'c_2', real=True, positive=True)
    # Create the mass, stiffness and damping symbolic matrices
    M = sym.Matrix([[m1,  0],
                    [0, m2]])
    K = sym.Matrix([[k1 + k12,     -k12],
                    [-k12, k2 + k12]])
    C = sym.Matrix([[0,  0],
                    [0, c2]])
    # Generate the analytic equation
    eq = sym.det(K - 1j * omega * C - omega**2 * M)

    # Variable parameters
    params = (omega, k2, c2)
    # Constant parameters and values
    const = (m1, m2, k1, k12)
    const_values = (1, 1, 0.9, 0.1)
    # Define the solver object
    pb = Disp(eq, params, const=const, const_values=const_values)
    # Locate real EP2
    z, status = pb.locate_ep(np.array((1., 0.,
                                        1., 0.,
                                        0.1, 0.)), real_param_ep=True)
    # Check error with reference solution
    err = np.linalg.norm(z - (omega_ref, k2_ref, c2_ref))
    print(f'Error wrt ref solution = {err}')
    return err


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=1)
