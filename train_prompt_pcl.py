import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory

from curvenet.dataset import SHREC23_PointCloudData_TextQuery
from curvenet.models import CurveNet

from common.models import BertExtractor, MLP
from common.test import test_loop
from common.train import train_loop

batch_size = 2
latent_dim = 128
epoch = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obj_extractor = CurveNet(device=device)

obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)
query_extractor = BertExtractor() # OOM, so freeze for baseline
query_embedder = MLP(query_extractor,latent_dim=latent_dim).to(device)

train_ds = SHREC23_PointCloudData_TextQuery(obj_data_path='data/SketchANIMAR2023/3D_Model_References/References',
                                             csv_data_path='data/csv/train_tex.csv')

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_ds.collate_fn)

test_ds = SHREC23_PointCloudData_TextQuery(obj_data_path='data/SketchANIMAR2023/3D_Model_References/References',
                                             csv_data_path='data/csv/test_tex.csv')

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
               obj_input='pointclouds', query_input='tokens',
               cbm_query=cbm_query, cbm_object=cbm_object,
               obj_optimizer=optimizer1, query_optimizer=optimizer2,
               dl=train_dl,
               device=device)
    
    print(f'Loss: {loss:.4f}')

    test_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
              obj_input='pointclouds', query_input='tokens',
              dl=test_dl,
              dimension=latent_dim,
              device=device)