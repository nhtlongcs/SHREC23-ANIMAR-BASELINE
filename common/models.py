import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel

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
        