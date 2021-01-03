from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import ml_tooling.metrics
from ml_tooling.utils import DataType


def plot_business_value(
    y_true: DataType,
    y_proba: DataType,
    tp_value: float = 1,
    fp_value: float = -1,
    tn_value: float = 1,
    fn_value: float = -1,
    title: str = None,
    ax: Axes = None,
    normalized: bool = True,
) -> Axes:
    """Plot business value as a function of the classification threshold.

    Parameters
    ----------
    y_true : DataType
        True labels
    y_proba : DataType
        Predicted probabilities of labels being 1
    tp_value : float, optional
        Business value of true positives, by default 1
    fp_value : float, optional
        Business value of false positives, by default -1
    tn_value : float, optional
        Business value of true negatives, by default 1
    fn_value : float, optional
        Business value of false negatives, by default -1
    title : str, optional
        Title for plot, by default None
    ax : Axes, optional
        Pass your own ax, by default None
    normalized : bool, optional
        Whether to normalize by highest business value, by default True

    Returns
    -------
    plt.Axes
        Returns a plot of business value for a range of classification thresholds.
    """
    title = "Business value" if title is None else title

    thresholds, bvs = ml_tooling.metrics.business_value(
        y_true, y_proba, tp_value, fp_value, tn_value, fn_value, normalized
    )

    if ax is None:
        fig, ax = plt.subplots()

    ax.step(x=thresholds, y=bvs, where="pre")

    ax.set_ylabel("Normalized Business Value" if normalized else "Busineses Value")
    ax.set_xlabel("Threshold")
    ax.set_title(title)

    plt.tight_layout()
    return ax
