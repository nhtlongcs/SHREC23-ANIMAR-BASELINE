import numpy as np
from sklearn.metrics import auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt

def evaluate(rank_matrix: np.ndarray, model_label: np.ndarray, image_label: np.ndarray):
    """Compute retrieval metrics given ranklist and groundtruth labels

    Args:
        rank_matrix: matrix of shape (num_queries, top-k) denoting the IDs of the retrieved models
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
    plt.savefig('PR_curve.jpg')

    return {
        "NN": nn,
        "P@10": p10,
        "NDCG": ndcg,
        "mAP": map_,
        # "auc": auc_
    }


def main():
    print(evaluate(np.array([[0, 2, 1, 3], [3, 0, 1, 2]]), np.array([-2, -1, -2, -3]), np.array([-2, -3])))
    print(evaluate(np.array([[0, 2, 1, 3], [0, 3, 1, 2]]), np.array([-2, -1, -2, -3]), np.array([-2, -3])))

if __name__ == "__main__":
    main()