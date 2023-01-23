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

    b = Y.shape[0]  # batch size

    e3 = np.hstack((np.zeros((theta.shape[0], theta.shape[1]//2)), np.ones((theta.shape[0], theta.shape[1] - theta.shape[1]//2))))

    # print(-np.diagonal(np.matmul(Y, np.matmul(X, e3.T).T)))
    # print(np.log(
    #     np.sum(np.exp(np.matmul(X, e3.T)), axis=1)))
    
    # print(-np.diagonal(np.matmul(Y, np.matmul(X, e3.T).T)) + np.log(
    #     np.sum(np.exp(np.matmul(X, e3.T)), axis=1)))
    
    # print(np.diagonal(np.matmul(Y, np.matmul(X, theta.T).T)) + np.log(
    #     np.sum(np.exp(np.matmul(X, theta.T)), axis=1)))

    # print(np.matmul(Y.T, np.matmul(X, theta.T)))

    return 1 / b * (np.sum(-np.diagonal(np.matmul(Y, np.matmul(X, theta.T).T)) + np.log(
        np.sum(np.exp(np.matmul(X, theta.T)), axis=1)), axis=0, keepdims=True))


def grad_theta_xent(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of xent(X, Y, theta).sum() wrt theta
    """
    s = np.matmul(X, theta.T)
    grad_s = X

    b = Y.shape[0]  # batch size

    # print(Y.shape, X.shape, theta.shape)
    # print((-np.matmul(Y.T, grad_s)).shape)
    # print(np.diagonal(np.matmul(np.exp(s).T, grad_s)).shape)
    # print(np.sum(np.exp(s), axis=1, keepdims=True).shape)

    # return 1 / b * (np.sum(-np.matmul(Y.T, grad_s) + np.divide(np.matmul(np.exp(s).T, grad_s)), np.sum(
    #     np.exp(s), axis=1, keepdims=True)), axis=0, keepdims=True)

    # print(np.sum(-np.matmul(Y.T, grad_s),axis=0, keepdims=True))

    return 1 / b * (-np.matmul(Y.T, grad_s) + np.sum(X, axis = 0, keepdims=True))


def mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """

    delta = Y - np.matmul(X, theta.T)

    return 1 / Y.shape[1] * np.sum(np.diagonal(np.matmul(delta.T, delta)), keepdims=True)


def grad_theta_mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of mse(X, Y, theta).sum() wrt theta
    """

    b = Y.shape[0]  # batch size

    return 1 / b * np.sum(2 / Y.shape[1] * np.matmul((np.matmul(X, theta.T) - Y).T, X), axis=0, keepdims=True)


def softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """

    return 1 / Y.shape[0] * np.sum(np.diagonal(np.matmul((Y - softmax(np.matmul(X, theta.T))).T, (Y - softmax(np.matmul(X, theta.T))))), keepdims= True)


def grad_theta_softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of softmse(X, Y, theta).sum() wrt theta
    """
    # return 2 / Y/shape[1] * (Y - softmax(np.matmul(X, theta.T))), (np.matmul(softmax(np.matmul(X, theta.T)).T, X) - np.matmul(softmax(np.matmul(X, theta.T)).T, softmax(np.matmul(X, theta.T))))

    b = Y.shape[0]

    return 1 / b * (-np.matmul(Y.T, softmax(X)) + np.sum(softmax(X), axis = 0, keepdims=True))


def myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """
    finiteDifferences(X,Y, theta)


def grad_theta_myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of myloss(X, Y, theta).sum() wrt theta
    """
    # TODO: your code here


def softmax(X):
    """
    X - an ndarray
    returns:
        softmax of X
    """

    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)


def finiteDifferences(X, Y, theta):
    e = np.identity(theta.shape[1])
    e2 = (np.vstack((e[0], np.zeros((theta.shape[0] - 1, theta.shape[1]))))) 
    e3 = np.hstack((np.zeros((theta.shape[0], theta.shape[1]//2)), np.ones((theta.shape[0], theta.shape[1] - theta.shape[1]//2))))
    constant = 1 / 10 ** 5
    print(grad_theta_mse(X, Y, theta)[0, 0])
    print((mse(X, Y, theta + e2 * constant) - mse(X, Y, theta - e2 * constant)) / (2 * constant))
    print(grad_theta_xent(X, Y, theta)[0, 0])
    print((xent(X, Y, theta + e2 * constant) - xent(X, Y, theta - e2 * constant)) / (2 * constant))
    print(grad_theta_softmse(X, Y, theta)[0, 0])
    print((softmse(X, Y, theta + e2 * constant) - softmse(X, Y, theta - e2 * constant)) / (2 * constant))
    # print(mse(X, Y, theta))
    # print(softmse(X, Y, theta))
    # print(grad_theta_mse(X, Y, theta)[0, 0])
    # print(mse(X, Y, e2))
    # print(mse(X, Y, np.matmul(e2, e * (1 + constant))) - mse(X, Y, np.matmul(e2, e * (1 - constant))))
    # print(theta)
    # print(grad_theta_mse(X, Y, theta))
    # print(theta + e3 * constant)
    # epsilon = 1 / 10 ** 5
    # # print(grad_theta_xent(X, Y, theta))
    # print(xent(X, Y, e3 * .233))
    # print(xent(X, Y, theta + e3 * constant))
    # print((1+ epsilon) * e)
    # print(np.matmul(theta, (1 + epsilon) * e).shape)
    # print((xent(X, Y, np.matmul(theta, (1 + epsilon) * e))))