import numpy as np

def expected_value_discrete(x, p):
    if not np.isclose(np.sum(p), 1):
        raise ValueError("wrong")
    x = np.array(x)
    p = np.array(p)
    sum = np.sum(x * p)
    return sum
