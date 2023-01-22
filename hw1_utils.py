import numpy as np


def logsumexp(Z, axis=1):
    """
    Z - an ndarray
    axis - the dimension over which to logsumexp
    returns:
        logsumexp over the axis'th dimension; returned tensor has same ndim as Z
    """
    maxes = np.max(Z, axis=axis, keepdims=True)
    return maxes + np.log(np.exp(Z - maxes).sum(axis, keepdims=True))


def score(X, theta):
    """
    X - bsz x D_1
    theta - K x D_1
    returns: bsz x K
    """
    return np.matmul(X, theta.transpose())


def xent(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    return -np.matmul(Y.T, np.matmul(theta, X)) + np.log(
        np.sum(np.exp(np.matmul(theta, X)))
    )


def grad_theta_xent(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of xent(X, Y, theta).sum() wrt theta
    """
    s = np.matmul(theta, X)
    grad_s = np.sum(s, axis=0)
    return -np.matmul(Y.T, grad_s) + np.matmul(np.exp(s), grad_s) / np.sum(
        np.exp(s), axis=0
    )


def mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    return (
        1
        / X.shape[0]
        * np.matmul((Y - np.matmul(theta, X)).T, (Y - np.matmul(theta, X)))
    )


def grad_theta_mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of mse(X, Y, theta).sum() wrt theta
    """
    return 2 / X.shape[0] * (np.matmul(theta, X) - Y)


def softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    return 1 / X.shape[0] * (Y - softmax(np.matmul(theta, X))) ** 2


def grad_theta_softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of softmse(X, Y, theta).sum() wrt theta
    """
    ## TODO: your code here


def myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    ## TODO: your code here


def grad_theta_myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of myloss(X, Y, theta).sum() wrt theta
    """
    ## TODO: your code here


def softmax(X):
    """
    X - an ndarray
    returns:
        softmax of X
    """
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
