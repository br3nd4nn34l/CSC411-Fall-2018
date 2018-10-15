# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from sklearn.datasets import load_boston

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist

def make_A(test_datum, X, tau):
    datum_row = np.reshape(test_datum, (1, test_datum.shape[0]))
    norms = l2(datum_row, X)

    exponents = -norms / (2 * (tau ** 2))

    # https://blog.feedly.com/tricks-of-the-trade-logsumexp/

    # exp(x) / sum(exp(x))
    # = exp(log(exp(x) / sum(exp(x))
    # = exp(log(exp(x)) - log(sum(exp(x)))
    # = exp(x - LogSumExp(x))
    return np.exp(exponents - logsumexp(exponents))

# to implement

def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    # Renaming for easier reasoning
    X, Y = x_train, y_train
    X_T = x_train.transpose()
    A = make_A(test_datum, X, tau)

    # The two sides of the equation
    # (use element wise multiplication instead of
    # matrix multiplication because A is a vector)
    left = ((X_T * A) @ X) + lam
    right = (X_T * A) @ Y

    # Solve (don't use slow inverse)
    try:
        w = np.linalg.solve(left, right)
    except:
        w = np.linalg.pinv(left) @ right

    # Predict y (dot product of w and test_datum)
    return np.dot(w, test_datum)

def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''

    # Randomly split the X's into training (70%), validation examples (30%)
    shuff_inds = np.arange(x.shape[0])
    np.random.shuffle(shuff_inds)

    shuff_x, shuff_y = x[shuff_inds], y[shuff_inds]
    split_ind = int(x.shape[0] * (1 - val_frac))
    x_train, x_test = shuff_x[:split_ind], shuff_x[split_ind:]
    y_train, y_test = shuff_y[:split_ind], shuff_y[split_ind:]

    # Prep vectors to return
    train_losses = np.empty_like(taus)
    test_losses = np.empty_like(taus)

    # Compute average loss for each tau
    for (i, t) in enumerate(taus):

        train_predictions = np.array([
            LRLS(datum, x_train, y_train, t)
                for datum in x_train
        ])

        test_predictions = np.array([
            LRLS(datum, x_train, y_train, t)
                for datum in x_test
        ])

        # Error for each datum
        train_errs = (train_predictions - y_train)
        test_errs = (test_predictions - y_test)

        # Use mean squared error
        train_losses[i] = np.mean(train_errs ** 2)
        test_losses[i] = np.mean(test_errs ** 2)

        print(i, t, train_losses[i], test_losses[i])
        print("")



    return train_losses, test_losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    num_taus = 50 # For modularity
    taus = np.logspace(1, 3, num_taus)

    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)

    plt.semilogx(taus, train_losses)
    plt.xlabel("Tau")
    plt.ylabel("Average Training Loss")
    plt.show()

    # Break test_losses into two graphs
    # because loss seems to be very high for first ~25% taus
    test_break = int(num_taus * 0.25)

    plt.semilogx(taus[:test_break], test_losses[:test_break])
    plt.xlabel("Tau")
    plt.ylabel("Average Test Loss")
    plt.show()

    plt.semilogx(taus[test_break:], test_losses[test_break:])
    plt.xlabel("Tau")
    plt.ylabel("Average Test Loss")
    plt.show()
