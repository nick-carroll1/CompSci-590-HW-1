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

    return 1 * (np.matmul((-Y + softmax(s)).T, X))



def mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """

    delta = Y - np.matmul(X, theta.T)
    b = Y.shape[0]
    k = Y.shape[1]

    return 1 / b / k * np.sum(np.diagonal(np.matmul(delta.T, delta)))
    


def grad_theta_mse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of mse(X, Y, theta).sum() wrt theta
    """

    b = Y.shape[0]  # batch size
    k = Y.shape[1]
    s = np.matmul(X, theta.T)

    return -2 / k * np.matmul((Y - s).T, X)


def softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """

    return 1 / Y.shape[0] / Y.shape[1] * np.sum(np.diagonal(np.matmul((Y - softmax(np.matmul(X, theta.T))).T, (Y - softmax(np.matmul(X, theta.T))))), keepdims= True)


def grad_theta_softmse(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of softmse(X, Y, theta).sum() wrt theta
    """

    k = Y.shape[1]
    s = np.matmul(X, theta.T)

    return - 2 / k * np.matmul((np.multiply(Y-softmax(s), softmax(s)) - np.multiply(Y-softmax(s), np.multiply(softmax(s), softmax(s)))).T, X)


def myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
       bsz-length array of losses
    """

    k = Y.shape[1]
    s = np.matmul(X, theta.T)
    sigmoid = np.divide(np.ones(s.shape), np.ones(s.shape) + np.exp(-s))

    return 1 / k * np.sum(np.diagonal(np.matmul((Y - sigmoid).T, (Y - sigmoid))), keepdims= True)


def grad_theta_myloss(X, Y, theta):
    """
    X - bsz x D_1
    Y - bsz x K
    theta - K x D_1
    returns:
        K x D_1 gradient of myloss(X, Y, theta).sum() wrt theta
    """
    k = Y.shape[1]
    s = np.matmul(X, theta.T)
    sigmoid = np.divide(np.ones(s.shape), np.ones(s.shape) + np.exp(-s))

    return - 2 / k * np.matmul(((Y - sigmoid) - np.multiply(Y - sigmoid, sigmoid)).T, X)


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
    b = Y.shape[0]
    for loop in range(0, 3):
        grad_point1 = np.random.randint(0, theta.shape[0])
        grad_point2 = np.random.randint(0, theta.shape[1])
        print(f"Cross Entropy gradient value: {grad_theta_xent(X, Y, theta)[grad_point1, grad_point2] / b}")
        print(f"Cross Entropy observed change: {(xent(X, Y, theta + e2 * constant) - xent(X, Y, theta - e2 * constant)) / (2 * constant)}")
        print(f"ratio: {grad_theta_xent(X, Y, theta)[grad_point1, grad_point2] / b / ((xent(X, Y, theta + e2 * constant) - xent(X, Y, theta - e2 * constant)) / (2 * constant))}")
        print(f"MSE gradient value: {grad_theta_mse(X, Y, theta)[grad_point1, grad_point2] / b}")
        print(f"MSE observed change: {(mse(X, Y, theta + e2 * constant) - mse(X, Y, theta - e2 * constant)) / (2 * constant)}")
        print(f"ratio: {grad_theta_mse(X, Y, theta)[grad_point1, grad_point2] / b / ((mse(X, Y, theta + e2 * constant) - mse(X, Y, theta - e2 * constant)) / (2 * constant))}")
        print(f"SoftMSE gradient value: {grad_theta_softmse(X, Y, theta)[grad_point1, grad_point2] / b}")
        print(f"SoftMSE observed change: {(softmse(X, Y, theta + e2 * constant) - softmse(X, Y, theta - e2 * constant)) / (2 * constant)}")
        print(f"ratio {grad_theta_softmse(X, Y, theta)[grad_point1, grad_point2] / b / ((softmse(X, Y, theta + e2 * constant) - softmse(X, Y, theta - e2 * constant)) / (2 * constant))}")
        print(f"MyLoss gradient value: {grad_theta_myloss(X, Y, theta)[0, 0]}")
        print(f"MyLoss observed change: {(myloss(X, Y, theta + e2 * constant) - myloss(X, Y, theta - e2 * constant)) / (2 * constant)}")

if __name__ == '__main__':
    from hw1_main import get_mnist_data, batchify
    X, Y, val_X, val_Y = get_mnist_data()
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    val_X = np.hstack((val_X, np.ones((val_X.shape[0], 1))))
    theta = np.zeros((Y.shape[1], X.shape[1])) # K x D_1
    bsz = 64
    Xs, Ys = batchify(X, Y, bsz)
    finiteDifferences(Xs[0],Ys[0], theta)
