import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    
    tp = np.sum(y_true == y_pred)

    total_instances = len(y_true)
    fp = total_instances - tp
    fn = total_instances - tp

    f1_mic = (2 * tp) / (2 * tp + fp + fn)
    
    return f1_mic