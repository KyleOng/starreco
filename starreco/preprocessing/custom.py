import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class CustomMultiLabelBinarizer(MultiLabelBinarizer):
    """
    Custom MultiLabelBinarizer.

    Notes: Original MultiLabelBinarizer fit_transform() only takes 2 positional arguments. However, our custom pipeline assumes the MultiLabelBinarizer fit_transform() is defined to take 3 positional arguments. Hence, adding an additional argument y to fit_transform() fix the problem.
    """

    def fit_transform(self, X, y = None):
        """
        Fix original MultiLabelBinarizer fit_transform().

        :param X: X.

        :param y: y (target), set as None.

        :return: transformed X.      
        """
        y = None
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype = "object")
        return super().fit_transform(X.flatten())
