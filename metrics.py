import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt

def evaluate(rank_matrix: np.ndarray, model_label: np.ndarray, image_label: np.ndarray, savefig=False):
    """Compute retrieval metrics given ranklist and groundtruth labels

    Args:
        rank_matrix: matrix of shape (num_queries, top-k) denoting the IDs (in range [0..num_models-1]) of the retrieved models
        model_label: class labels for 3D models, shape (num_models, )
        image_label: class labels for 2D images, shape (num_queries, )

    Returns:
        PRC, NN, P@10, NDCG, mAP
    """
    num_query, top_k = rank_matrix.shape
    num_retrieval = len(model_label)
    precision = np.zeros((num_query, num_retrieval))
    recall = np.zeros((num_query, num_retrieval))

    # transform rank matrix to 0-1 matrix denoting irrelevance/relevance
    rel_matrix = model_label[rank_matrix] == image_label[..., np.newaxis]
    map_ = 0.0
    for i in range(num_query):
        max_match = np.sum(model_label == image_label[i])
        r_points = np.zeros(max_match)
        G_sum = np.cumsum(np.int8(rel_matrix[i]))
        for j in range(max_match):
            r_points[j] = np.where(G_sum == (j+1))[0][0] + 1
        r_points_int = np.array(r_points, dtype=int)
        map_ += np.mean(G_sum[r_points_int-1] / r_points)

        r1 = G_sum / float(max_match)
        p1 = G_sum / np.arange(1, num_retrieval+1)
        recall[i] = r1
        precision[i] = p1

    map_ /= num_query

    nn = np.mean(rel_matrix[:, 0])
    p10 = np.mean(rel_matrix[:, :10])

    logf = np.log2(1 + np.arange(1, top_k + 1))[np.newaxis, :]  # reduction factor for DCG
    dcg = np.sum(rel_matrix / logf, axis=-1)        # (num_queries, )
    idcg = np.zeros((num_query, num_retrieval))
    for i in range(num_query):
        idcg[i, :np.sum(rel_matrix[i])] = 1
    idcg = np.sum(idcg / logf, axis=-1)
    ndcg = np.mean(dcg / idcg)

    pre = np.mean(precision, axis=0)
    rec = np.mean(recall, axis=0)
    auc_ = auc(rec, pre)
    _, ax = plt.subplots(figsize=(7, 8))
    display = PrecisionRecallDisplay(recall=rec, precision=pre)
    display.plot(ax=ax, name=f"Precision-recall curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    if savefig:
        plt.savefig('PR_curve.jpg')

    return {
        "NN": nn,
        "P@10": p10,
        "NDCG": ndcg,
        "mAP": map_,
        # "auc": auc_
    }

def evaluate_submission(csv_path: os.PathLike, gt_path: os.PathLike, test_path: os.PathLike):
    """Runs evaluation for codalab. Works for both SketchANIMAR and TextANIMAR.

    Args:
        csv_path (os.PathLike): Path to submission csv
        gt_path (os.PathLike): Path to groundtruth csv
        test_path (os.PathLike): Path to query list csv

    Returns:
        tuple: a tuple of the metrics in order: NN, P@10, NDCG, mAP
    """
    gt = pd.read_csv(gt_path, sep=None, engine='python')

    text_col = gt["Text Query ID"] if "Text Query ID" in gt.columns else gt["Sketch Query ID"]
    model_col = gt["Model ID"]
    text_ids = list(set(text_col))
    model_ids = list(set(model_col))
    num_queries = len(text_ids)
    num_models = len(model_ids)

    labels = {}
    parent = {}

    # list out and label all connected components
    def get_parent(x):
        if x not in parent:
            return x
        parent[x] = get_parent(parent[x])
        return parent[x]

    def join(x, y):
        x = get_parent(x)
        y = get_parent(y)
        if x != y:
            parent[y] = x

    for text_id, model_id in zip(text_col, model_col):
        join(text_id, model_id)

    counter = 0
    for text_id in filter(lambda x: x not in parent, text_col):
        labels[text_id] = counter
        counter += 1
    for id in [x for x in text_col if x in parent] + model_ids:
        labels[id] = labels[get_parent(id)]

    edges = set(zip(text_col, model_col))
    for text_id in text_ids:
        for model_id in model_ids:
            if labels[text_id] == labels[model_id]:
                assert (text_id, model_id) in edges

    # get the order of the queries
    query_df = pd.read_csv(test_path)
    query_list = query_df["ID"]
    assert len(query_list) == num_queries

    # read the submission
    submission = pd.read_csv(csv_path, sep=None, engine='python', header=None)

    def return_error(message):
        print(message)
        return (0, 0, 0, 0)

    if len(submission) != num_queries or len(submission.columns) != num_models:
        return return_error(f"Submission must have shape (num_queries, num_models)! \n"
                            f"Expected ({num_queries}, {num_models}), got ({submission.size}, {len(submission.columns)})")
    
    id_matrix = submission.to_numpy()
    for row in id_matrix:
        if np.unique(row).size != row.size:
            return return_error("List of IDs for a query must be unique!")
        
    model_order = {id:idx for idx, id in enumerate(model_ids)}
    rank_matrix = np.zeros((num_queries, num_models), dtype=np.int64)
    
    for i in range(num_queries):
        try:
            rank_matrix[i] = [model_order[x] for x in id_matrix[i]]
        except KeyError as err:
            return return_error(f"Model ID {err} does not exist!")

    model_labels = np.array([labels[x] for x in model_order.keys()])
    query_labels = np.array([labels[x] for x in query_list])

    return tuple(evaluate(rank_matrix, model_labels, query_labels).values())

def main():
    print(evaluate_submission("sample_submission.csv", "SketchQuery_GT_Train.csv", "SketchQuery_Train.csv"))

if __name__ == "__main__":
    main()