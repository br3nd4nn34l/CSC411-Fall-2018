'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
from scipy.misc import logsumexp

# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

NUM_CLASSES = 10


def label_to_rows(train_data, train_labels):
    """
    Breaks train_data into a list of length NUM_CLASSES.
    i-th element of the list are the rows that belong to class i from train_data
    Assumption: train_labels is aligned with train_data
    """
    # i-th element is indices of train_labels where label=i
    label_indices = [np.nonzero(train_labels == i) for i in range(NUM_CLASSES)]

    # i-th element: all rows that belong to class i
    return [train_data[inds] for inds in label_indices]


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    # For each collection of rows, take mean across axis 0 (row-wise)
    return np.array([
        np.mean(rows, axis=0)
        for rows in label_to_rows(train_data, train_labels)
    ])


def covariance(X):
    """
    Outputs a (cols x cols) covariance matrix of X (rows x cols)
    """

    rows, cols = X.shape

    # Mean of each column
    mu = np.mean(X, axis=0)

    # Differences between results and mean
    mu_diff = X - mu

    # Terms of covariance matrix
    # Each term is sum of multiplied mu_diff pairs
    covariance = (mu_diff.transpose() @ mu_diff) / rows

    # For numerical stability add 0.01I
    stabilizer = np.eye(cols) * 0.01

    return covariance + stabilizer


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''

    # Rows separated by class
    per_class = label_to_rows(train_data, train_labels)

    # Compute covariance within each class
    return np.array([
        covariance(rows)
        for rows in per_class
    ])


def log_class_likelihood(digits, class_mean, class_cov):
    """
    Compute the log probability that each row of digits is of class,
    given the class_mean (mu) and class_cov (sigma)
    """
    dimensionality = class_mean.shape[0]

    # Log(2pi ** (-d/2)) -> (-d/2)Log(2pi)
    two_pi_term = (-dimensionality / 2) * np.log(2 * np.pi)

    # Log(det(sigma_k) ** (-1/2)) -> (-1/2)Log(det(sigma_k))
    sigma_term = (-1 / 2) * np.log(np.linalg.det(class_cov))

    # Log(exp(-0.5 * (x - mu_k)^T * sigma_k^-1 (x-mu_k))) ->
    # -0.5 * (x - mu_k)^T * sigma_k^-1 (x-mu_k)
    inv_sigma = np.linalg.inv(class_cov)
    mu_diff = digits - class_mean

    # Rephrasing of the above equation
    # (need dot product of (diff x inv) and diff, for each row)
    exponents = (-0.5 * ((mu_diff @ inv_sigma) * mu_diff)) \
        .sum(axis=1) \
        .flatten()

    # Log(Multiplication of terms) -> Addition of terms
    return two_pi_term + sigma_term + exponents


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    return np.array([
        log_class_likelihood(digits, means[i], covariances[i])
        for i in range(NUM_CLASSES)
    ]).transpose()


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    # p(y|x) = p(x|y)p(y)/p(x) -> L(y|x) = L(x|y) + L(y) - L(x)
    log_x_given_y = generative_likelihood(digits, means, covariances)

    # Labels are distributed evenly
    log_p_y = -np.log(NUM_CLASSES)

    # P(x) = (Sum of all Y) of (P(x, y) = P(x|y)P(y))
    log_p_xy = log_x_given_y + log_p_y
    log_p_x = logsumexp(log_p_xy, axis=1).reshape(-1, 1)

    return log_x_given_y + log_p_y - log_p_x


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    log_cond_like = conditional_likelihood(digits, means, covariances)

    # i-th element is conditional log likelihood of i-th sample's ACTUAL label
    log_p_y_correct_given_x = log_cond_like[(
        np.arange(len(labels)), # take all rows
        labels.astype(int) # select row[label]
    )]

    # Return mean value of the logs
    return np.mean(log_p_y_correct_given_x)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Index with highest likelihood corresponds to most likely class
    return np.argmax(cond_likelihood, axis=1)


def leading_eigenvector(covariance):
    eigvals, eigvecs = np.linalg.eig(covariance)

    # Eigen-vectors are column-wise
    return eigvecs[:, np.argmax(eigvals)]


def run_a(train_data, train_labels, test_data, test_labels, means, covariances):
    print("Part A: Average Conditional Log-Likelihoods")

    train_acl = avg_conditional_likelihood(train_data, train_labels,
                                           means, covariances)
    print(f"Train: {train_acl}")

    test_acl = avg_conditional_likelihood(test_data, test_labels,
                                          means, covariances)
    print(f"Test: {test_acl}")


def run_b(train_data, train_labels, test_data, test_labels, means, covariances):
    print("Part B: Accuracy")

    train_preds = classify_data(train_data, means, covariances)
    train_acc = np.mean(train_preds == train_labels)
    print(f"Train Accuracy: {train_acc}")

    test_preds = classify_data(test_data, means, covariances)
    test_acc = np.mean(test_preds == test_labels)
    print(f"Test Accuracy: {test_acc}")

def run_c(covariances):
    for i, cov in enumerate(covariances):
        eigvec = leading_eigenvector(cov)
        plt.title(f"Leading Eigenvector of {i}")
        plt.imshow(eigvec.reshape(8, 8))
        plt.savefig(f"plots/q1c-{i}.png")


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # PART A
    run_a(train_data, train_labels, test_data, test_labels, means, covariances)
    print("")

    # PART B
    run_b(train_data, train_labels, test_data, test_labels, means, covariances)
    print("")

    # PART C
    run_c(covariances)


if __name__ == '__main__':
    main()
