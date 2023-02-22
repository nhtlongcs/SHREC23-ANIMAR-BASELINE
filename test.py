from retriever import FaissRetrieval
import numpy as np 
from tqdm import tqdm
import torch 
from metrics import evaluate

def encode_labels(labels):
    unique_labels = np.unique(labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    return [label_to_id[label] for label in labels]
def print_results(results):
    for metric, value in results.items():
        print(f'- {metric}: {value}')

def test(obj_embedder, query_embedder, query_field, dimension, test_loader, device):
    gallery_embeddings = []
    query_embeddings = []
    retriever = FaissRetrieval(dimension=dimension)
    
    obj_embedder.eval()
    query_embedder.eval()
    query_ids = []
    target_ids = []
    gallery_ids = []
    print('- Evaluation started...')
    print('- Extracting embeddings...')
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            g_emb = obj_embedder(batch['object_ims'].to(device))
            q_emb = query_embedder(batch[query_field].to(device))
            gallery_embeddings.append(g_emb.detach().cpu().numpy())
            query_embeddings.append(q_emb.detach().cpu().numpy())
            query_ids.extend(batch['query_ids'])
            gallery_ids.extend(batch['gallery_ids'])
            target_ids.extend(batch['gallery_ids'])

    max_k = len(set(gallery_ids))
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

    query_ids = encode_labels(query_ids)
    target_ids = encode_labels(target_ids)
    gallery_ids = encode_labels(gallery_ids)
    max_query = len(set(query_ids))

    rank_matrix = np.zeros((max_query, max_k), dtype=np.int32)
    for top_k_indexes, query_id in zip(top_k_indexes_all, query_ids):
        pred_ids = [gallery_ids[i] for i in top_k_indexes] # gallery id
        rank_matrix[query_id, :] = np.array(pred_ids, dtype=np.int32)

    model_labels = np.arange(len(set(gallery_ids)))
    query_labels = np.arange(max_query)
    print('- Evaluation results:')
    print_results(evaluate(rank_matrix, model_labels, query_labels))

def test_txt():
    from torch.utils.data import DataLoader
    from dataset import SHREC23_Rings_RenderOnly_TextQuery
    from models import Base3DObjectRingsExtractor, BertExtractor, MLP
    
    batch_size = 4
    latent_dim = 128
    device = 'cuda'

    test_ds = SHREC23_Rings_RenderOnly_TextQuery(
            'data/csv/test_tex.csv', 'data/SketchANIMAR2023/3D_Model_References/generated_models', None, [1, 3, 5])

    dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=test_ds.collate_fn)
    

    obj_extractor = Base3DObjectRingsExtractor(
        nheads=4,
        dropout=0.1,
        nrings=3,
    )
    obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)
    query_extractor = BertExtractor(is_frozen=True) # OOM, so freeze for baseline
    query_embedder = MLP(query_extractor, latent_dim=latent_dim).to(device)

    test(obj_embedder, query_embedder, 'tokens', latent_dim, dl, device=device)


def test_image():
    from torch.utils.data import DataLoader
    from dataset import SHREC23_Rings_RenderOnly_ImageQuery
    from models import Base3DObjectRingsExtractor, ResNetExtractor, MLP
    
    batch_size = 4
    latent_dim = 768
    device = 'cuda'

    test_ds = SHREC23_Rings_RenderOnly_ImageQuery(
        'data/csv/test_skt.csv', 'data/SketchANIMAR2023/3D_Model_References/generated_models', 'data/SketchANIMAR2023/Train/SketchQuery_Train', [1, 3, 5])

    dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=test_ds.collate_fn)

    obj_extractor = Base3DObjectRingsExtractor(
        nheads=4,
        dropout=0.1,
        nrings=3,
    )

    obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)
    query_extractor = ResNetExtractor()
    query_embedder = MLP(query_extractor, latent_dim=latent_dim).to(device)
    test(obj_embedder, query_embedder, 'query_ims', latent_dim, dl, device=device)

if __name__ == '__main__':
    # test_txt()
    test_image()