# https://github.com/KevinMusgrave/pytorch-metric-learning/issues/373
import torch
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory
from pytorch_metric_learning.reducers import BaseReducer
from torch.utils.data import DataLoader
import time
from dataset import SHREC23_Rings_RenderOnly_TextQuery
from models import Base3DObjectRingsExtractor, BertExtractor, MLP

class PerModality(BaseReducer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def pos_pair_reduction(self, losses, loss_indices, *args):
        a1, _ = loss_indices
        halfway = self.batch_size // 2
        obj_loss = torch.mean(losses[a1 < halfway])
        txt_loss = torch.mean(losses[a1 >= halfway])
        return (obj_loss + txt_loss) / 2

batch_size = 2
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set optimizers
obj_extractor = Base3DObjectRingsExtractor(
    nheads=4,
    dropout=0.1,
    nrings=3,
)
obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)
query_extractor = BertExtractor(is_frozen=True) # OOM, so freeze for baseline
query_embedder = MLP(query_extractor, latent_dim=latent_dim).to(device)

contra_loss = NTXentLoss()
cbm_prompt = CrossBatchMemory(contra_loss, latent_dim, 128)
cbm_object = CrossBatchMemory(contra_loss, latent_dim, 128)

optimizer1 = torch.optim.Adam(obj_embedder.parameters(), lr=0.00001, weight_decay=0.0001)
optimizer2 = torch.optim.Adam(query_embedder.parameters(), lr=0.00001, weight_decay=0.0001)

train_ds = SHREC23_Rings_RenderOnly_TextQuery(
        'data/csv/train_tex.csv', 'data/SketchANIMAR2023/3D_Model_References/generated_models', None, [1, 3, 5])

dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_ds.collate_fn)
start_time = time.time()
for step, batch in enumerate(dl):
    obj_emb = obj_embedder(batch['object_ims'].to(device))
    txt_emb = query_embedder(batch['tokens'].to(device))
    emb = torch.cat([obj_emb, txt_emb])
    emb_len = emb.shape[0] # batch_size * 2 (text + object)
    
    labels = torch.cat([torch.arange(emb_len//2), torch.arange(emb_len//2)])
    if step > 2:
        enqueue_idx = torch.arange(emb_len//2, emb_len)
        # enqueue_mask: A boolean tensor where enqueue_mask[i] is True
        enqueue_mask = torch.zeros(emb_len, dtype=torch.bool)
        enqueue_mask[enqueue_idx] = True
        # enqueue_mask = enqueue_mask[:min(enqueue_mask.shape[0], emb.shape[0]),]
        loss1 = cbm_object(emb, labels, enqueue_mask=enqueue_mask)

        # This is the text-to-video loss
        enqueue_idx = torch.arange(emb_len//2, emb_len)

        enqueue_mask = torch.zeros(emb_len, dtype=torch.bool)
        enqueue_mask[enqueue_idx] = True
        # trim mask

        loss2 = cbm_prompt(emb, labels, enqueue_mask=enqueue_mask)
        loss = (loss1 + loss2) / 2
    else:
        contra_loss = NTXentLoss(reducer=PerModality(emb_len))
        loss = contra_loss(emb, labels)
    
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()
    optimizer2.step()

    print('Step: {}/{}, Loss: {}, Elapsed: {:02f}'.format(step, len(train_ds)//batch_size, loss.item(), time.time() - start_time))
