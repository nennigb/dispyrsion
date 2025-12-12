Dispyrsion
=========

`Dispysion` is a library to locate exceptional points (EP) from analytical dispersion equations $K(\lambda,\boldsymbol\nu)=0$,
where, $\lambda$ is the eigenvalue of a nonlinear eigenvalue problem and $\boldsymbol\nu$ is a vector of parameters.
An EP corresponds to the specific value of $\boldsymbol\nu^*$ where $\lambda$ is a multiple eigenvalue. A necessary condition to find them
is to have $K(\lambda, \boldsymbol\nu)=0$, $\partial_\lambda K(\lambda, \boldsymbol\nu)=0$ up to $\partial^n_\lambda K(\lambda, \boldsymbol\nu)=0$ for an EP(n+1).

With this package, only the equation $K(\lambda, \boldsymbol\nu)=0$ have to be provided. The derivatives are automacally obtained with `sympy`.
It avoids the subsitution step required to establish the equation of the EP, which may be sometimes cumbersome or unavailable.
The EP with **complex**- or **real**-valued parameters are obtained with Levenberg-Marquard solver (adapted from [`EasterEig`](https://github.com/nennigb/EasterEig)).
This solver is iterative and requires an initial guess. If several EPs are required, several initial guesses have to be tested.

When the EP equation is available, solver leading to all solutions like those based on based on the argument principle (see for instance [`polze`](https://github.com/nennigb/polze) are generally more robust but limited are to complex-valued parameters.

For EP2 with real-valued parameters, an alternative, based on continuation, exists and can be found in [`real-valued-ep2`](https://github.com/nicolase7en/real-valued-ep2) see [^1] for details.


[^1]: Even, N., Nennig, B., Lefebvre, G., & Perrey-Debain, E. (2023). Experimental observation of exceptional points in coupled pendulums. Journal of Sound and Vibration, 575:118239, 2024. doi:10.1016/j.jsv.2024.118239. arXiv preprint arXiv:2308.01089. [10.48550/arXiv.2308.01089](https://doi.org/10.48550/arXiv.2308.01089).


References
----------
todo

 
       
Install 
--------

### From the wheel
todo

### From the source
If you need to modify the code or the last development version, you need to install `dispyrsion` from the source. The sources are available on the github [repos](https://github.com/nennigb/dispyrsion/).

First, you need to download and to install [`polze`](https://github.com/nennigb/polze) from the polze README page.

Then, you need to get the `dispyrsion` sources. In a console opened in the `dispyrsion` top level folder, run
```console
pip install .
```

Running tests
-------------
Tests are handled with `doctest` and with `unittest`.  To execute the full test suite, run :
```console
python -m dispyrsion
```

Getting started
---------------
Several examples are given in the `examples` folder.

