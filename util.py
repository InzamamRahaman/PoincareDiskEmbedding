import torch
import random







def proj(x, epsilon=0.00001):
    """
    Projects a matrix of vectors in hyperbolic geometry back into the unit ball
    Parameters
    ----------
    x
        the matrix with each row representing a vector
    epsilon
        the small difference to subtract to ensure that vectors remain in the unit ball

    Returns
    -------
        a matrix of vectors modified to ensure that all vectors are contained
        in the unit ball

    """
    size = x.size()
    norms = torch.norm(x, p=2, dim=1)
    #print(norms)
    norms = norms.repeat(1, size[1])
    res = x / norms
    res -= epsilon
    return res

def metric_tensor(x):
    """
    Computes the metric tensor for a matrix containing vectors in hyperbolic space
    Parameters
    ----------
    x
        the matrix with each row representing a vector

    Returns
    -------
        a Tensor holding the metric tensor values for each vector

    """
    denom = 1 - torch.pow(torch.norm(x, p=2, dim=1), 2)
    numer = 2
    frac = numer / denom
    res = torch.pow(frac, 2)
    return res


def arcosh(x):
    factor1 = torch.sqrt(x - 1)
    factor2 = torch.sqrt(x + 1)
    prod = factor1 * factor2
    inner = x + prod
    res = torch.log(inner)
    return res


def hyperbolic_distance(x, y):
    denom1 = torch.pow(1 - torch.norm(x, p=2, dim=1), 2)
    denom2 = torch.t(torch.pow(1 - torch.norm(y, p=2, dim=1), 2))
    denom = denom1 @ denom2

    x_size = x.size()
    y_size = y.size()

    y1 = y.repeat(x_size[0], 1, 1)
    x1 = x.view(x_size[0], 1, x_size[1])
    x1 = x.repeat(1, y_size[0], 1)
    z = (x1 - y1)
    norms = torch.norm(z, p=2, dim=2)
    norms = norms.view(x_size[0], y_size[0])
    numer = torch.pow(norms, 2)
    factor = numer / denom
    factor *= 2
    inner = factor + 1
    res = arcosh(inner)
    return res



