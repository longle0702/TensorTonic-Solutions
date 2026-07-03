import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)
    
    SS_res = np.sum((y_true - y_pred)**2)
    SS_tot = np.sum((y_true - y_mean)**2)

    if SS_tot == 0:
        if np.array_equal(y_true, y_pred):
            return 1.0
        else:
            return 0.0

    r2 = 1 - ( SS_res / SS_tot )
    return r2