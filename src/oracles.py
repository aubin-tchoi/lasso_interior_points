import numpy as np


def f0(v, Q, p) -> float:
    """
    Objective function of the quadratic problem.
    """
    return float((v.T @ Q @ v + p.T @ v)[0][0])


def g(v, Q, p, A, b, t0) -> float:
    """
    g is defined as g = tf0 + phi
    """
    return float((t0 * f0(v, Q, p) - sum(np.log(-(A @ v - b))))[0])


def grad_g(v, Q, p, A, b, t) -> np.ndarray:
    """
    Computes the gradient of g at a point v.
    """
    return t * (2 * Q @ v + p) + sum(
        A[i, np.newaxis].T / (b[i] - A[i] @ v) for i in range(b.shape[0])
    )


def hessian_g(v, Q, A, b, t) -> np.ndarray:
    """
    Computes the Hessian of g at a point v.
    """
    return t * (Q + Q.T) + np.tensordot(
        A.T, np.reciprocal((A @ v - b) ** 2) * A, axes=1
    )
