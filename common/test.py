from retriever import FaissRetrieval
import numpy as np 
from tqdm import tqdm
import torch 
from metrics import evaluate
from typing import List

def encode_labels(labels):
    unique_labels = np.unique(labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    return [label_to_id[label] for label in labels]

def print_results(results):
    for metric, value in results.items():
        if isinstance(value, np.ndarray):
            value = np.mean(value)
        print(f'- {metric}: {value}')

def test_loop(obj_embedder, query_embedder, obj_input, query_input, dimension, dl, device):
    gallery_embeddings = []
    query_embeddings = []
    retriever = FaissRetrieval(dimension=dimension, cpu=True) # Temporarily use CPU to retrieve (avoid OOM)
    
    obj_embedder.eval()
    query_embedder.eval()
    query_ids = []
    target_ids = []
    gallery_ids = []
    print('- Evaluation started...')
    print('- Extracting embeddings...')
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dl), total=len(dl)):
            g_emb = obj_embedder(batch[obj_input].to(device))
            q_emb = query_embedder(batch[query_input].to(device))
            gallery_embeddings.append(g_emb.detach().cpu().numpy())
            query_embeddings.append(q_emb.detach().cpu().numpy())
            query_ids.extend(batch['query_ids'])
            gallery_ids.extend(batch['gallery_ids'])
            target_ids.extend(batch['gallery_ids'])

    max_k = len(gallery_ids) # retrieve all available gallery items
    query_embeddings = np.concatenate(query_embeddings, axis=0)
    gallery_embeddings = np.concatenate(gallery_embeddings, axis=0)
    print('- Calculating similarity...')
    top_k_scores_all, top_k_indexes_all = retriever.similarity_search(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            top_k=max_k,
            query_ids=query_ids, target_ids=target_ids, gallery_ids=gallery_ids,
            save_results="temps/query_results.json"
        )

    model_labels = np.array(encode_labels(query_ids), dtype=np.int32)
    rank_matrix = top_k_indexes_all

    np.savetxt("temps/rank_matrix.csv", rank_matrix, delimiter=",",fmt='%i')
    print('- Evaluation results:')
    print_results(evaluate(rank_matrix, model_labels, model_labels))
