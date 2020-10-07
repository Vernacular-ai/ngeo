from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class NgeoFeaturizer(BaseEstimator, TransformerMixin):
    """
    Default featurizer for ngeo.
    """

    def __init__(self):
        self.cvec = CountVectorizer()

    def fit(self, X: List[str], y=None):
        self.cvec.fit(X)

        return self

    def transform(self, X: List[str], y=None):
        return self.cvec.transform(X)
