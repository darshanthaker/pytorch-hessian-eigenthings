"""
This module contains functions to perform power iteration with deflation
to compute the top eigenvalues and eigenvectors of a linear operator
"""
import numpy as np
import torch

from hessian_eigenthings.utils import log, progress_bar


class Operator:
    """
    maps x -> Lx for a linear operator L
    """

    def __init__(self, size):
        self.size = size

    def apply(self, vec):
        """
        Function mapping vec -> L vec where L is a linear operator
        """
        raise NotImplementedError


class LambdaOperator(Operator):
    """
    Linear operator based on a provided lambda function
    """

    def __init__(self, apply_fn, size):
        super(LambdaOperator, self).__init__(size)
        self.apply_fn = apply_fn

    def apply(self, x):
        return self.apply_fn(x)


def deflated_power_iteration(
    operator,
    num_eigenthings=10,
    power_iter_steps=20,
    power_iter_err_threshold=1e-4,
    momentum=0.0,
    use_gpu=True,
    most_negative=False,
    to_numpy=True,
    verbose=True,
):
    """
    Compute top k eigenvalues by repeatedly subtracting out dyads
    operator: linear operator that gives us access to matrix vector product
    num_eigenvals number of eigenvalues to compute
    power_iter_steps: number of steps per run of power iteration
    power_iter_err_threshold: early stopping threshold for power iteration

    most_negative: if True, get the most negative eigenvalues instead

    returns: np.ndarray of top eigenvalues, np.ndarray of top eigenvectors
    """
    eigenvals = []
    eigenvecs = []

    if most_negative:
        top_eigenval, top_eigenvec = power_iteration(operator, power_iter_steps,
                                          power_iter_err_threshold,
                                          momentum=momentum,
                                          use_gpu=use_gpu, 
                                          verbose=verbose)
        eigenvals.append(top_eigenval)
        eigenvecs.append(top_eigenvec)
        SCALE = 1.5  # make sure we deflate the main op enough
        top_eigenval = SCALE * abs(top_eigenval)

        def _min_op(x, op=operator):
            return op.apply(x) - top_eigenval * x
        operator = LambdaOperator(_min_op, operator.size)

    current_op = operator
    prev_vec = None

    def _deflate(x, val, vec):
        return val * vec.dot(x) * vec

    if verbose:
        log("beginning deflated power iteration")
    for i in range(num_eigenthings):
        if verbose:
            log("computing eigenvalue/vector %d of %d" % (i + 1, num_eigenthings))
        eigenval, eigenvec = power_iteration(
            current_op,
            power_iter_steps,
            power_iter_err_threshold,
            momentum=momentum,
            use_gpu=use_gpu,
            init_vec=prev_vec,
            verbose=verbose
        )
        if verbose:
            log("eigenvalue %d: %.4f" % (i + 1, eigenval))

        def _new_op_fn(x, op=current_op, val=eigenval, vec=eigenvec):
            return op.apply(x) - _deflate(x, val, vec)

        current_op = LambdaOperator(_new_op_fn, operator.size)
        prev_vec = eigenvec
        if most_negative:
            eigenval += top_eigenval
        eigenvals.append(eigenval)
        eigenvec = eigenvec.cpu()
        if to_numpy:
            eigenvecs.append(eigenvec.numpy())
        else:
            eigenvecs.append(eigenvec)

    eigenvals = np.array(eigenvals)
    eigenvecs = np.array(eigenvecs)

    # sort them in descending order
    sorted_inds = np.argsort(eigenvals)
    eigenvals = eigenvals[sorted_inds][::-1]
    eigenvecs = eigenvecs[sorted_inds][::-1]
    return eigenvals, eigenvecs


def power_iteration(
    operator, steps=20, error_threshold=1e-4, momentum=0.0, use_gpu=True, init_vec=None,
    verbose=True
):
    """
    Compute dominant eigenvalue/eigenvector of a matrix
    operator: linear Operator giving us matrix-vector product access
    steps: number of update steps to take
    returns: (principal eigenvalue, principal eigenvector) pair
    """
    vector_size = operator.size  # input dimension of operator
    if init_vec is None:
        vec = torch.rand(vector_size)
    else:
        vec = init_vec

    if use_gpu:
        vec = vec.cuda()

    prev_lambda = 0.0
    prev_vec = torch.randn_like(vec)
    for i in range(steps):
        prev_vec = vec / (torch.norm(vec) + 1e-6)
        new_vec = operator.apply(vec) - momentum * prev_vec
        # need to handle case where we end up in the nullspace of the operator.
        # in this case, we are done.
        if torch.sum(new_vec).item() == 0.0:
            return 0.0, new_vec
        lambda_estimate = vec.dot(new_vec).item()
        diff = lambda_estimate - prev_lambda
        vec = new_vec.detach() / torch.norm(new_vec)
        if lambda_estimate == 0.0:  # for low-rank
            error = 1.0
        else:
            error = np.abs(diff / lambda_estimate)
        if verbose:
            progress_bar(i, steps, "power iter error: %.4f" % error)
        if error < error_threshold:
            return lambda_estimate, vec
        prev_lambda = lambda_estimate

    return lambda_estimate, vec
