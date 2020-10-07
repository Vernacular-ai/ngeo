from typing import List

import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class NgeoFeaturizer(BaseEstimator, TransformerMixin):
    """
    Default featurizer for ngeo.
    """

    def __init__(self):
        self.cvec = CountVectorizer(analyzer="char_wb", ngram_range=(1, 3))
        self.wvec = CountVectorizer()

    def fit(self, X: List[str], y=None):
        self.cvec.fit(X)
        self.wvec.fit(X)

        return self

    def transform(self, X: List[str], y=None):
        return sp.sparse.hstack([self.cvec.transform(X), self.wvec.transform(X)])
