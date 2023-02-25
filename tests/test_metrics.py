import os
import random

import pandas as pd
import numpy as np

def create_random_submission(csv_path: os.PathLike, gt_path: os.PathLike, test_path: os.PathLike):
    gt = pd.read_csv(gt_path, sep=None, engine='python')

    text_col = gt["Text Query ID"] if "Text Query ID" in gt.columns else gt["Sketch Query ID"]
    model_col = gt["Model ID"]

    text_ids = list(set(text_col))
    model_ids = list(set(model_col))
    num_queries = len(text_ids)
    matrix = [None] * num_queries

    for i in range(num_queries):
        matrix[i] = model_ids.copy()
        random.shuffle(matrix[i])
    matrix = np.array(matrix)
    pd.DataFrame.from_records(matrix).to_csv(csv_path, index=None, header=None)

def test_metrics():
    from metrics import evaluate_submission

    create_random_submission("sample_submission.csv", "SketchQuery_GT_Train.csv", "SketchQuery_Train.csv")
    print(evaluate_submission("sample_submission.csv", "SketchQuery_GT_Train.csv", "SketchQuery_Train.csv"))

if __name__ == "__main__":
    test_metrics()

