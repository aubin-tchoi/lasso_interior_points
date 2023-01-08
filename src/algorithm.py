import numpy as np
from typing import Tuple, List

from .oracles import f0, g, grad_g, hessian_g


def backtracking_line_search(
    Q, p, A, b, t0, v, grad, step, t: float = 1, alpha: float = 1e-2, beta: float = 0.5
) -> np.ndarray:
    """
    Backtracking line search method to find the step size.
    Here, t is the line search variable and t0 the barrier parameter.

    Returns:
        The updated variable.
    """
    while (
        g(v + t * step, Q, p, A, b, t0)
        > (g(v, Q, p, A, b, t0) + alpha * t * grad.T @ step)
        and ((b - A @ (v + t * step)) <= 0).any()
        # prevents infinite log values by not moving outside the polyhedron
    ):
        t *= beta
    return v + t * step


def centering_step(
    Q, p, A, b, t, v0, eps: float, n_iter: int = 0
) -> Tuple[np.ndarray, int]:
    """
    Centering step using the Newton method.
    The step size is found using a backtracking line search.

    Returns:
        v0: The step
    """
    while True:
        grad_value = grad_g(v0, Q, p, A, b, t)
        hessian_value = hessian_g(v0, Q, A, b, t)

        # Solving a linear system is faster than inverting the Hessian
        step = -np.linalg.solve(hessian_value, grad_value)
        lambda_square = -grad_value.T @ step

        if lambda_square / 2 <= eps:
            return v0, n_iter

        v0 = backtracking_line_search(Q, p, A, b, t, v0, grad_value, step)
        n_iter += 1


def barr_method(
    Q, p, A, b, v0, eps, mu
) -> Tuple[List[int], List[np.ndarray], np.ndarray]:
    """
    Implementation of the barrier method.

    Returns:
        iter_seq: Sequence of iteration steps.
        v_seq: Sequence of variable iterates.
        f_seq: Sequence of values of the objective function.
    """
    iter_seq, v_seq, f_seq = [0], [v0], [f0(v0, Q, p)]
    t, n_iter = 1, 0
    while True:
        v_center, n_iter_inner = centering_step(Q, p, A, b, t, v0, eps)
        iter_seq.append(n_iter)
        v_seq.append(v_center)
        f_seq.append(f0(v_center, Q, p))

        if b.shape[0] / t < eps:
            return iter_seq, v_seq, np.array(f_seq)

        t *= mu
        n_iter += n_iter_inner
