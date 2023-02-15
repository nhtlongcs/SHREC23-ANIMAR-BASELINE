import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from transformers import AutoModel

__all__ = [ 'MLP',
            'Extractor',
            'LanguageExtractor',
            'ImageExtractor',
            'BertExtractor',
            'ResNetExtractor',
            'BaseRingExtractor',
            'Base3DObjectRingsExtractor']

class Extractor(nn.Module):
    def freeze(self):
        for p in self.extractor.parameters():
            p.requires_grad = False

    def get_embedding(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.get_embedding(x)

class LanguageExtractor(Extractor):
    def get_embedding(self, x):
        x = self.get_feature_map(x)
        feature = torch.mean(x, dim=1)
        return feature

class BertExtractor(LanguageExtractor):
    def __init__(self, version='bert-base-uncased', use_pretrained=True, is_frozen=False):
        super().__init__()
        self.extractor = AutoModel.from_pretrained(version)
        self.feature_dim = self.extractor.config.hidden_size
        if is_frozen:
            self.freeze()

    def get_feature_map(self, x):
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        transformer_out = self.extractor(
            input_ids=input_ids, attention_mask=attention_mask
        )
        feature = transformer_out.last_hidden_state
        return feature

    
class ImageExtractor(Extractor):
    def get_feature_map(self, x):
        raise NotImplementedError

    def get_embedding(self, x):
        x = self.get_feature_map(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

class ResNetExtractor(ImageExtractor):
    arch = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    def __init__(self, version='resnet50', use_pretrained=True, is_frozen=False, drop=0):
        super().__init__()
        assert version in ResNetExtractor.arch, \
            f'Invalid version [{version}].'
        cnn = ResNetExtractor.arch[version](pretrained=use_pretrained)
        self.extractor = nn.Sequential(*list(cnn.children())[:-2-drop])
        self.feature_dim = cnn.fc.in_features // 2 ** drop
        if is_frozen:
            self.freeze()

    def get_feature_map(self, x):
        return self.extractor(x)

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

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, extractor: Extractor, latent_dim=128):
        super().__init__()
        self.extractor = extractor
        self.feature_dim = extractor.feature_dim
        # mlp = 4 layers, feature_dim -> feature_dim / 2 -> feature_dim / 4 -> latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, latent_dim),
        )
    def forward(self, x):
        x = self.extractor.get_embedding(x)
        x = self.mlp(x)
        return x
        
    
if __name__ == "__main__":
    from dataset import *
    ds = SHREC22_Rings_RenderOnly_ImageQuery(
        'data/train.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    ds2 = SHREC22_Rings_RenderOnly_TextQuery(
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