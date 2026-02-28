import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asanyarray(y)
    if len(y) == 0:
        return 0.0
    i, counts = np.unique(y, return_counts = True)
    
    p = counts / counts.sum()
    p = p[p > 0]
    
    hs = -np.sum(p * np.log2(p))
    
    return float(max(hs, 0.0))