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

def get_labels(text_ids: list[str], model_ids: list[str], edges: list[tuple[str, str]]):
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

    for text_id, model_id in edges:
        join(text_id, model_id)

    counter = 0
    for text_id in [x for x in text_ids + model_ids if x not in parent]:
        labels[text_id] = counter
        counter += 1
    for id in [x for x in text_ids + model_ids if x in parent]:
        labels[id] = labels[get_parent(id)]

    # data check
    edges = set(edges)
    for text_id in text_ids:
        for model_id in model_ids:
            if labels[text_id] == labels[model_id]:
                assert (text_id, model_id) in edges

    return labels

def evaluate_submission(csv_path: os.PathLike, gt_path: os.PathLike, query_path: os.PathLike, model_path: os.PathLike, subset_path: os.PathLike):
    """Runs evaluation for codalab. Works for both SketchANIMAR and TextANIMAR.

    Args:
        csv_path (os.PathLike): Path to submission csv
        gt_path (os.PathLike): Path to groundtruth csv
        query_path (os.PathLike): Path to query list csv
        model_path (os.PathLike): Path to model list csv
        subset_path (os.PathLike): Path to train/test subset csv

    Returns:
        tuple: a tuple of the metrics in order: NN, P@10, NDCG, mAP
    """
    gt = pd.read_csv(gt_path, sep=None, engine='python')

    text_col = gt["Text Query ID"] if "Text Query ID" in gt.columns else gt["Sketch Query ID"]
    model_col = gt["Model ID"]

    # read all queries and models
    text_ids = pd.read_csv(query_path)["ID"].tolist()
    model_ids = pd.read_csv(model_path)["ID"].tolist()
    
    num_tot_queries = len(text_ids)
    num_models = len(model_ids)
    labels = get_labels(text_ids, model_ids, list(zip(text_col, model_col)))

    # read the submission
    submission = pd.read_csv(csv_path, sep=None, engine='python', header=None)

    def return_error(message):
        print(message)
        return (0, 0, 0, 0)

    if len(submission) != num_tot_queries or len(submission.columns) != num_models + 1:
        return return_error(f"Submission must have shape (num_queries, num_models + 1)! \n"
                            f"Expected ({num_tot_queries}, {num_models + 1}), got ({len(submission)}, {len(submission.columns)})")
    
    submission = submission.to_numpy()
    ranklist = {}
    for row in submission:
        if np.unique(row[1:]).size != row.size - 1:
            return return_error("List of IDs for a query must be unique!")
        ranklist[row[0]] = row[1:]
        
    # read split
    subset = set(pd.read_csv(subset_path)["ID"])
    num_queries = len(subset)
    assert len([x for x in text_ids if x in subset]) == len(subset)

    model_order = {id:idx for idx, id in enumerate(model_ids)}
    model_labels = np.array([labels[x] for x in model_order.keys()], dtype=np.int64)
    rank_matrix = np.zeros((num_queries, num_models), dtype=np.int64)
    query_labels = np.zeros((num_queries, ), dtype=np.int64)

    print(submission.shape)
    print(query_labels.shape)

    for i, query_id in enumerate(subset):
        row = ranklist[query_id]
        try:
            query_labels[i] = labels[query_id]
        except KeyError as err:
            return return_error(f"Query ID {err} does not exist!")
        try:
            rank_matrix[i] = [model_order[x] for x in row]
        except KeyError as err:
            return return_error(f"Model ID {err} does not exist!")

    return tuple(evaluate(rank_matrix, model_labels, query_labels).values())

def main():
    print(evaluate_submission("sample_submission.csv", "SketchQuery_GT_Train.csv", "SketchQuery_Train.csv", "References.csv", "SketchQueryID_TrainTrain.csv"))

if __name__ == "__main__":
    main()