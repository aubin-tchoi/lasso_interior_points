# Interior points method applied to the LASSO problem

This repo contains resources to run the interior points method applied to the dual of the LASSO problem.
A backtracking line search is carried out to find the step size of the centering step in the method.

The theoretical part containing the derivation of the dual of the LASSO problem can be found in `main.ipynb`,
alongside the results of the numerical experiments.

File `oracles.py` contains first and second-order oracles for the problem.
File `algorithm.py` contains the implementation of the barrier method.
