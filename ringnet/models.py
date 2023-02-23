import torch
import torch.nn as nn
from common.models import Extractor, BertExtractor, ResNetExtractor

__all__ = [ 'BaseRingExtractor',
            'Base3DObjectRingsExtractor']


class BaseRingExtractor(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.cnn = ResNetExtractor()
        self.cnn_feature_dim = self.cnn.feature_dim  # D
        self.feature_dim = 2 * hidden_dim  # D'
        self.lstm = nn.LSTM(self.cnn_feature_dim,
                            hidden_dim,
                            batch_first=True,
                            bidirectional=True)

    def get_embedding(self, x):
        # x: [B, V, C, H, W]
        B, V, C, H, W = x.size()
        x = x.reshape(B*V, C, H, W)  # B*V, C, H, W
        x = self.cnn.get_embedding(x)  # B*V, D
        x = x.reshape(B, V, self.cnn_feature_dim)  # B, V, D
        x, _ = self.lstm(x)  # B, V, D'
        x = x.mean(1)  # B, D'
        return x
    

class Base3DObjectRingsExtractor(nn.Module):
    def __init__(self, nrings, nheads, dropout=0.0, reverse=False):
        super().__init__()
        self.reverse = reverse
        if reverse:
            nrings = 12
        self.ring_exts = nn.ModuleList([
            BaseRingExtractor()
            for _ in range(nrings)
        ])
        self.view_feature_dim = self.ring_exts[0].feature_dim  # D
        self.feature_dim = self.view_feature_dim  # D'
        self.attn = nn.MultiheadAttention(self.feature_dim, nheads, dropout)

    def forward(self, x):
        # x: B, R, V, C, H, W
        if self.reverse:
            x = x.transpose(1, 2)
        x = torch.cat([
            ring_ext.get_embedding(x[:, i]).unsqueeze(1)
            for i, ring_ext in enumerate(self.ring_exts)
        ], dim=1)  # B, R, D
        x = x.transpose(0, 1)  # R, B, D
        x, _ = self.attn(x, x, x)  # R, B, D
        x = x.mean(0)  # B, D
        return x

    def get_embedding(self, x):
        # x: B, R, V, C, H, W
        return self.forward(x)

    
if __name__ == "__main__":
    from dataset import *
    ds = SHREC23_Rings_RenderOnly_ImageQuery(
        'data/train.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    ds2 = SHREC23_Rings_RenderOnly_TextQuery(
        'data/train.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    dl = data.DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    dl2 = data.DataLoader(ds2, batch_size=2, collate_fn=ds2.collate_fn)


    obj_extractor = Base3DObjectRingsExtractor(
        nheads=4,
        dropout=0.1,
        nrings=3,
    )
    obj_embedder = MLP(obj_extractor)
    img_extractor = ResNetExtractor()
    img_embedder = MLP(img_extractor)
    txt_extractor = BertExtractor()
    txt_embedder = MLP(txt_extractor)

    batch = next(iter(dl))
    ring_inputs = batch['object_ims']
    img_query = batch['query_ims']

    ring_outputs = obj_embedder(ring_inputs)
    print(ring_outputs.shape)

    img_query_outputs = img_embedder(img_query)
    print(img_query_outputs.shape)

    batch = next(iter(dl2))
    txt_query = batch['tokens']
    txt_query_outputs = txt_embedder(txt_query)
    print(txt_query_outputs.shape)