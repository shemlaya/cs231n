import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]   
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        int_sum = np.sum(np.exp(scores))
        p = lambda k: np.exp(scores[k]) / int_sum
        loss += -np.log(p(y[i]))
        
        for k in range(num_classes):
            p_k = p(k)
            dW[:, k] += (p_k - (k == y[i])) * X[i]
            
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W
                
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    scores = X.dot(W) #shape: (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)
    int_sum = np.sum(np.exp(scores), axis=1) #shape: (N, )
    
    p = np.exp(scores) / int_sum.reshape((num_train, 1)) #shape: (N, )
    loss = np.sum(-np.log(p[np.arange(num_train), y]))
    
    ind = np.zeros(p.shape)
    ind[np.arange(num_train), y] = 1
    dW = X.T.dot(p - ind)
    #dW[np.arange(num_classes), y] = p - 
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

