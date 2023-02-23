import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory

from ringnet.dataset import SHREC23_Rings_RenderOnly_ImageQuery
from ringnet.models import Base3DObjectRingsExtractor
from common.models import ResNetExtractor, MLP

from common.test import test_loop
from common.train import train_loop

batch_size = 2
latent_dim = 128
epoch = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obj_extractor = Base3DObjectRingsExtractor(
    nheads=4,
    dropout=0.1,
    nrings=3,
)
obj_embedder = MLP(obj_extractor,latent_dim=latent_dim).to(device)
query_extractor = ResNetExtractor()
query_embedder = MLP(query_extractor,latent_dim=latent_dim).to(device)

train_ds = SHREC23_Rings_RenderOnly_ImageQuery(
        'data/csv/train_skt.csv', 'data/SketchANIMAR2023/3D_Model_References/generated_models', 'data/SketchANIMAR2023/Train/SketchQuery_Train', [1, 3, 5])

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_ds.collate_fn)

test_ds = SHREC23_Rings_RenderOnly_ImageQuery(
        'data/csv/test_skt.csv', 'data/SketchANIMAR2023/3D_Model_References/generated_models', 'data/SketchANIMAR2023/Train/SketchQuery_Train', [1, 3, 5])

test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=test_ds.collate_fn)

contra_loss = NTXentLoss()
cbm_query = CrossBatchMemory(contra_loss, latent_dim, 128)
cbm_object = CrossBatchMemory(contra_loss, latent_dim, 128)

# Set optimizers
optimizer1 = torch.optim.Adam(obj_embedder.parameters(), lr=0.00001, weight_decay=0.0001)
optimizer2 = torch.optim.Adam(query_embedder.parameters(), lr=0.00001, weight_decay=0.0001)

for e in range(epoch):
    print(f'Epoch {e+1}/{epoch}:')
    loss = train_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
               obj_input='object_ims', query_input='query_ims',
               cbm_query=cbm_query, cbm_object=cbm_object,
               obj_optimizer=optimizer1, query_optimizer=optimizer2,
               dl=train_dl,
               device=device)
    
    print(f'Loss: {loss:.4f}')

    test_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
              obj_input='object_ims', query_input='query_ims',
              dl=test_dl,
              dimension=latent_dim,
              device=device)