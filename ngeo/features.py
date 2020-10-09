from typing import List

import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class NgeoFeaturizer(BaseEstimator, TransformerMixin):
    """
    Default featurizer for ngeo.
    """

    def __init__(self):
        self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 10))

    def fit(self, X: List[str], y=None):
        self.cvec.fit(X)
        return self

    def transform(self, X: List[str], y=None):
        return sp.sparse.hstack([self.cvec.transform(X)])
