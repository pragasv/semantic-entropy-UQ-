import numpy as np


########### functions to generate regression data (input = x, desired = function output)) ##################
############### function 1 #############
def f2(x, alpha=4., beta=13.):
    w = np.random.normal(0, 0.03)
    return x + np.sin(alpha*(x+w)) + np.sin(beta*(x+w))


# ############### function 2 #############
def f1(x):
    # w = np.random.normal(0, 0.3)
    return x*np.sin(x)

############### function to get k random samples in range (a, b) ##############
def sample(a, b, k):
    assert b>a
    return np.random.random(k)*(b-a) + a

############### generate x and y (i.e. f(x)) for regression, but only in specific regions: (-1.2, 0.1) and (0.7, 1) ##################

def generate_test_set(dataset="case_1"):
    if dataset == "case_1":
        x_list = np.r_[sample(-5, 5, 60)]
        X_train = np.asarray(sorted(x_list)).reshape(-1, 1)
        y_train = np.asarray([f1(x) for x in X_train]).reshape(-1, 1)

        x_list = np.r_[sample(-15, 15, 120)]
        X_test = np.asarray(sorted(x_list)).reshape(-1, 1)
        y_test = np.asarray([f1(x) for x in X_test]).reshape(-1, 1)

    elif dataset == "case_2":
        x_list = np.r_[sample(-1.2, 0.1, 80), sample(0.7, 1., 30)]  ## this needs to be changed
        X_train = np.asarray(sorted(x_list)).reshape(-1, 1)
        y_train = np.asarray([f2(x) for x in X_train]).reshape(-1, 1)

        x_list = np.r_[sample(-2, 2, 120)]
        X_test = np.asarray(sorted(x_list)).reshape(-1, 1)
        y_test = np.asarray([f2(x) for x in X_test]).reshape(-1, 1)

    else:
        raise ValueError("Unknown Dataset requested")

    return X_train, y_train, X_test, y_test