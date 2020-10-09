from functools import partial

import numpy as np

from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate


def classification_report_with_f1(y_true, y_pred, pos_label):
    print(classification_report(y_true, y_pred))

    return f1_score(y_true, y_pred, pos_label=pos_label)


def cv_train(estimator, X, y):
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    pos_label = sorted(set(y))[0]
    print(f":: Choosing {pos_label} as positive")

    print(f":: Training using {n_splits} cross validation splits\n")
    scorer = make_scorer(partial(classification_report_with_f1, pos_label=pos_label))
    scores = cross_validate(estimator, X, y, cv=cv, scoring=scorer)

    print(f":: F1 scores across splits: {scores['test_score']}")
    print(f":: Mean F1 score: {np.mean(scores['test_score'])}")
