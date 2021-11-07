import numpy as np
from matplotlib import pyplot as plt

Y = np.loadtxt('Probabilistic and Unsupervised Learning/CW1/binarydigits.txt')

# My code as per below
X = Y.T
# Create log likelihood function that we will look to maximise (as positive as possible)
# This looks to tackle each parameter by itself and look through all the 100 data points of each parameter and
# then optimise p such as to maximise the log likelihood equation given by sum(ln_prob_bern).  Given this is a
# binary data set, it is not suitable to use gradient descent unless we use some sort of sigmoid function with it.
# Alternatively, we can solve it analytically through the use of Y.sum(axis=0)/100 where we find the mean of each
# parameter as its maximum likelihood estimator
ln_prob_bern = lambda x, p, alpha, beta: x * np.log(p) + (1.0 - x) * np.log(1. - p)


def iterate_p_for_params(log_likelihood_func, X, alpha=0, beta=0):
    output = []
    for param in X:
        holder = []
        for increment in range(1, 1000):
            p = increment / 1000.0
            new = sum(log_likelihood_func(param, p, alpha, beta))
            # Initialise the value of p
            if not holder:
                holder = [new, p]
            # Retain the value of p that maximises the function
            if new > holder[0]:
                holder = [new, p]
        output.append(holder)
    return output


ML = iterate_p_for_params(ln_prob_bern, X)

plt.figure()
plt.imshow(np.reshape(list(zip(*ML))[1], (8, 8)),
           interpolation="None",
           cmap='gray')
plt.axis('off')
plt.show()
