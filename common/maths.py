from libs import *

# This is rather a prototype function
def f_divergence(p1, p2, f):
    s = 0
    for i in range(len(p1)):
        if p1[i] < 1e-9 or p2[i] < 1e-9:
            s += 0
        else:
            s += p1[i] * f(p2[i] / p1[i])
    return s

# Statistical distance
def hellinger_distance(p1, p2):
    def func(x):
        return (1 - np.sqrt(x))
    return f_divergence(p1, p2, func)


def bhattacharyya_distance(p1, p2):
    return -np.log(np.sum(np.sqrt(p1*p2)))


# Divergence measurements
def KL_divergence(p1, p2):
    def func(x):
        return x*np.log2(x)
    return f_divergence(p1, p2, func)


def JS_divergence(p1, p2):
    return jensenshannon(p1, p2, base=2)
