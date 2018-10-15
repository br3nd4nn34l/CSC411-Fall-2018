import numpy as np
from sklearn import datasets

# Hyper Parameters
DELTA = 1
LEARN_RATE = 0.01

def huber_loss(y, t):

    residuals = y - t

    inside_delta = np.where(np.abs(residuals) <= DELTA)
    outside_delta = np.where(np.abs(residuals) > DELTA)

    errors = residuals
    errors[inside_delta] = (errors[inside_delta] ** 2) / 2
    errors[outside_delta] = DELTA * (np.abs(errors[outside_delta]) - (0.5 * DELTA))

    # Take mean of all errors
    return np.mean(errors)


def H_delta_prime(a):

    # Return modified copy of a
    ret = np.copy(a)

    # a < -delta ==> set to -delta
    less_than_neg_delta = np.where(a < -DELTA)
    ret[less_than_neg_delta] = -DELTA

    # a > delta ==> set to delta
    more_than_delta = np.where(a > DELTA)
    ret[more_than_delta] = DELTA

    return ret

def loss_d_weights(x, y, t):
    """Computes dL/dW"""
    vec = np.matmul(H_delta_prime(y - t), x)
    mag = np.sum(vec ** 2) ** 0.5
    return vec / mag


def loss_d_bias(y, t):
    """Computes dL/db"""
    return H_delta_prime(y - t)

def predict(x, w, b):
    return np.matmul(x, w) + b

def fit_model(x, t, num_iters):

    x_rows, x_cols = x.shape

    w = np.zeros(x_cols)
    b = 0

    for i in range(num_iters):
        y = predict(x, w, b)
        loss = huber_loss(y, t)

        # Print out (at least) 10 messages
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}, Loss {loss}")

        w -= loss_d_weights(x, y, t) * LEARN_RATE
        b -= loss_d_bias(y, t) * LEARN_RATE

    return w, b

if __name__ == '__main__':
    boston_ds = datasets.load_boston()
    x = boston_ds.data
    t = boston_ds.target
    fit_model(x, t, 1000)