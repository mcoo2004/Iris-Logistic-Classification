import math
import numpy as np

def sigmoid(z):
    if not (np.isscalar(z)):
        return [1.0 / (1.0 + math.exp(-x)) for x in z]
    else:
        return 1.0 / (1.0 + math.exp(-z))

