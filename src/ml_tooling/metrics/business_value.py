import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve

from ml_tooling.utils import MetricError


def business_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    tp_value: float = 1,
    fp_value: float = -1,
    tn_value: float = 1,
    fn_value: float = -1,
    normalized: bool = True,
) -> np.ndarray:
    """Calculate business values for a range of classification thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilites for labels being 1
    tp_value : float, optional
        Business value of true positives, by default 1
    fp_value : float, optional
        Business value of false positives, by default -1
    tn_value : float, optional
        Business value of true negatives, by default 1
    fn_value : float, optional
        Business value of false negatives, by default -1
    normalized: bool, optional
        Whether to normalize business values relative to highest value, by default True

    Returns
    -------
    np.ndarray
        Array with thresholds, business values
    """
    # calculate counts of fps, tps, fns, tns for all thresholds
    fps, tps, threshold = _binary_clf_curve(y_true, y_proba)

    tns = fps[-1] - fps
    fns = tps[-1] - tps

    # vector with business values
    X = np.array([tp_value, fp_value, tn_value, fn_value]).reshape((4, 1))

    # actual business value for each threshold
    bv = np.sum(np.array([tps, fps, tns, fns]) * X, axis=0)

    if normalized:
        max_bv = np.max(bv)
        if max_bv > 0:
            bv = bv / max_bv
        elif max_bv < 0:
            bv = (bv / max_bv) ** -1
        else:
            raise MetricError(
                "Highest business value is 0. Cannot normalize by this value."
            )

    return np.array([threshold, bv])
