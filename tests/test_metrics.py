import os
import random

import pandas as pd
import numpy as np

def create_random_submission(csv_path: os.PathLike, query_path: os.PathLike, model_path: os.PathLike):
    text_ids = pd.read_csv(query_path)["ID"].tolist()
    model_ids = pd.read_csv(model_path)["ID"].tolist()

    num_queries = len(text_ids)
    matrix = [None] * num_queries

    for i in range(num_queries):
        matrix[i] = model_ids.copy()
        random.shuffle(matrix[i])
        matrix[i] = [text_ids[i]] + matrix[i]
    matrix = np.array(matrix)
    pd.DataFrame.from_records(matrix).to_csv(csv_path, index=None, header=None)

def test_metrics():
    from metrics import evaluate_submission

    create_random_submission("sample_submission.csv", "SketchQuery_Train.csv", "References.csv")
    # print(evaluate_submission("sample_submission.csv", "SketchQuery_GT_Train.csv", "SketchQuery_Train.csv"))

if __name__ == "__main__":
    test_metrics()

