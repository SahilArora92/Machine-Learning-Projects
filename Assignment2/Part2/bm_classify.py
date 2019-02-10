import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    y = np.where(y == 0, -1, 1)
    w = np.append(w, b)
    X = np.c_[X, np.ones(N)]
    if N == 0:
        return w, b
    else:
        if loss == "perceptron":
            ############################################
            # TODO 1 : Edit this if part               #
            #          Compute w and b here            #

            # w_t = np.transpose(w)
            # check Sign
            # mis_class = np.dot(y, (np.dot(X, w_t)+b).T)
            # mis = [1 if x <= 0 else 0 for x in np.ndindex(mis_class.shape)]

            # SGD update
            # for epoch in range(max_iterations):
            #     for i, x in enumerate(X):
            #         if ((np.dot(X[i].T, w) + b) * y[i]) <= 0:
            #             w = w + step_size*X[i]*y[i]
            #             b = b + step_size*y[i]
            # return w, b

            # GD update
            for epoch in range(max_iterations):
                y_estimated = np.dot(X, w.T)
                mult_val = np.multiply(y, y_estimated)
                indexes = mult_val <= 0
                x_ind = X[indexes]
                y_ind = y[indexes]
                w += step_size * (np.dot(y_ind.T, x_ind) / N)
            return w[:-1], w[-1]
            ############################################


        elif loss == "logistic":
            ############################################
            # TODO 2 : Edit this if part               #
            #          Compute w and b here            #
            # w = np.tile(w, (N,1))
            for epoch in range(max_iterations):
                z = w*X[:, np.newaxis]
                z = np.sum(z, axis=1)
                z = np.sum(z, axis=1)
                h = sigmoid(-(y.T * z).T)
                temp = (X.T * (h.T * y).T).T
                w += (step_size * (temp.sum(axis=0) / N))

            return w[:-1], w[-1]
            ############################################


        else:
            raise "Loss Function is undefined."


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    ############################################
    return 1 / (1 + np.exp(-z))


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        # preds = np.dot(X, w.T) + b
        return np.where((np.dot(X, w.T) + b) > 0, 1, 0)
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        # preds = sigmoid(np.dot(X, w.T) + b)
        return np.where((sigmoid(np.dot(X, w.T) + b)) > 0.5, 1, 0)
        ############################################

    else:
        raise "Loss Function is undefined."


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    w = np.c_[w, b]
    X = np.c_[X, np.ones(N)]
    # classes = np.unique(y)
    # classes.sort()
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        # SGD update
        # y_enc = (np.arange(np.max(y) + 1) == y[:, None]).astype(float)
        for epoch in range(max_iterations):
            rand_index = np.random.choice(N)
            y_n = y[rand_index]
            input_mat = np.multiply(w, X[rand_index])
            input_temp = np.sum(input_mat, axis=1)
            soft_m = soft_max(input_temp-np.max(input_temp))
            soft_m[y_n] -= 1
            w -= step_size * np.multiply(np.tile(X[rand_index], (C, 1)), soft_m[:, np.newaxis])
        return w.T[:-1].T, w.T[-1]
        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        return w, b
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."


def soft_max(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=0)).T


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    return np.argmax(np.dot(X, w.T) + b, axis=1)
    ############################################
