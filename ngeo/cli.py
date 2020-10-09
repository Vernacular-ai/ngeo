"""
ngeo is not an NGO

Usage:
  ngeo fit --csv-file=<csv-file> --output-model=<output-model>
  ngeo predict --csv-file=<csv-file> --model=<model>
  ngeo evaluate --csv-file=<csv-file>

Options:
  --csv-file=<csv-file>     CSV file with name and class information
"""

import pickle

import pandas as pd
from docopt import docopt

from ngeo.core import cv_train
from ngeo.features import NgeoFeaturizer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def main():
    args = docopt(__doc__)

    if args["fit"]:
        df = pd.read_csv(args["--csv-file"])
        df = df[["text", "class-true"]].dropna()

        print(f":: Found {len(df)} training examples")

        X = df["text"].tolist()
        y = df["class-true"].tolist()

        f = NgeoFeaturizer()
        clf = SVC(kernel="linear")
        pipeline = make_pipeline(f, clf)

        cv_train(pipeline, X, y)

        # Training on full
        pipeline.fit(X, y)

        with open(args["--output-model"], "wb") as fp:
            pickle.dump(pipeline, fp)

    elif args["predict"]:
        df = pd.read_csv(args["--csv-file"])

        with open(args["--model"], "rb") as fp:
            pipeline = pickle.load(fp)

        X = df["text"].tolist()
        labels = list(pipeline.classes_)
        y_pred_scores = pipeline.predict_proba(X)
        y_pred_labels = [labels[i] for i in y_pred_scores.argmax(axis=1)]
        df["class-pred"] = y_pred_labels
        df["score-pred"] = y_pred_scores.max(axis=1)

        df.to_csv(args["--csv-file"], index=False)

    elif args["evaluate"]:
        df = pd.read_csv(args["--csv-file"])
        df = df[["class-true", "class-pred"]].dropna()

        print(f":: Evaluating on {len(df)} data points\n")

        print(classification_report(df["class-true"].tolist(), df["class-pred"].tolist()))
    else:
        raise NotImplementedError()
