import pytest
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.svm import SVC

from ml_tooling import Model
from ml_tooling.data import load_demo_dataset
from ml_tooling.metrics import business_value
from ml_tooling.utils import VizError


class TestBusinessValueCurve:
    @pytest.fixture(scope="class")
    def ax(self, classifier: Model) -> Axes:
        """Setup a business value curve."""
        yield classifier.result.plot.business_value_curve()
        plt.close()

    def test_plots_can_be_given_an_ax(self, classifier: Model):
        fig, ax = plt.subplots()
        test_ax = classifier.result.plot.business_value_curve(ax=ax)
        assert ax == test_ax
        plt.close()

    def test_has_correct_title(self, ax: Axes):
        assert ax.title.get_text() == "Business Value - LogisticRegression"

    def test_has_correct_y_label(self, ax: Axes):
        assert ax.get_ylabel() == "Normalized Business Value"

    def test_plot_fails_correctly_without_predict_proba(self):
        dataset = load_demo_dataset("iris")
        svc = Model(SVC(gamma="scale"))
        result = svc.score_estimator(dataset)
        with pytest.raises(VizError):
            result.plot.business_value_curve()
        plt.close()

    def test_correct_data_is_passed_to_plot(self, ax, classifier):
        y_true = classifier.result.plot._data.test_y
        x = classifier.result.plot._data.test_x

        y_proba = classifier.result.estimator.predict_proba(x)[:, 1]
        thresh, bvs = business_value(y_true, y_proba)

        assert np.all(thresh == ax.lines[0].get_xdata())
        assert np.all(bvs == ax.lines[0].get_ydata())
