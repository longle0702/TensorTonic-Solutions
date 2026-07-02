import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    x = np.array(a)
    y = np.array(b)

    tu = np.dot(x, y)
    mau = np.linalg.norm(x) * np.linalg.norm(y)
    
    if mau == 0:
        return 0.0
    else:
        return float(tu / mau)