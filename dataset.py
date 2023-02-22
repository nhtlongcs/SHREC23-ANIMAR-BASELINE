from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np
import pandas as pd 
import os
from transformers import AutoTokenizer

from transformers.tokenization_utils_base import BatchEncoding
os.environ[
    "TOKENIZERS_PARALLELISM"
] = "true"  # https://github.com/huggingface/transformers/issues/5486

__all__ = [
    'SHREC23_Rings_RenderOnly_ImageQuery',
    'SHREC23_Rings_RenderOnly_TextQuery',
]

def DatasetItemInfo(item, indent=0):
    if isinstance(item, dict):
        for k, v in item.items():
            print(' ' * indent, k)
            DatasetItemInfo(v, indent + 4)
    elif isinstance(item, BatchEncoding):
        for k, v in item.items():
            print(' ' * indent, k)
            DatasetItemInfo(v, indent + 4)
    elif isinstance(item, list):
        for v in item:
            DatasetItemInfo(v, indent + 4)
    elif isinstance(item, torch.Tensor):
        print(' ' * indent, item.shape, item.dtype, item.device)
    else:
        print(' ' * indent, item)

def DatasetBatchInfo(batch):
    for k, v in batch.items():
        print(k)
        DatasetItemInfo(v, 4)
    print()

class BaseShrecDataset(data.Dataset):
    def __init__(self,
                 csv_path,
                 root,
                 skt_root,
                 ring_ids,
                 is_train=True,
                 vis=False):
        super().__init__()
        csv_data = pd.read_csv(csv_path)
        print(csv_data.head())

        self.obj_ids = csv_data['obj_filename']
        self.obj_ids = [x.split('.')[0] for x in self.obj_ids]
        self.skt_filenames = None if 'sket_filename' not in csv_data.columns else csv_data['sket_filename']
        self.tex = None if 'tex' not in csv_data.columns else csv_data['tex']

        assert self.skt_filenames is not None or self.tex is not None, 'Must provide either sketch or text'

        self.obj_root = root
        self.skt_root = skt_root

        self.data = [
            {
                mode:
                [
                    [
                        os.path.join(self.obj_root,
                                     f'ring{ring_id}',
                                     f'{obj_id}',
                                     mode,
                                     f'Image{view_id:04d}.png')
                        for view_id in range(1, 13)
                    ]
                    for ring_id in ring_ids
                ]
                for mode in ['depth', 'mask', 'render']
            }
            for obj_id in self.obj_ids
        ]

        if is_train:
            self.render_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])

            self.mask_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
            ])
        else:
            self.render_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])

            self.mask_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
            ])

        if vis:
            self.render_transforms = tvtf.Compose([
                tvtf.ToTensor(),
            ])

            self.mask_transforms = tvtf.Compose([
            ])

        self.is_train = is_train
        print('Dataset length: ',len(self))

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        raise NotImplementedError

class SHREC23_Rings_RenderOnly_ImageQuery(BaseShrecDataset):
    def __getitem__(self, i):
        data = self.data[i]['render']
        query_impath = self.skt_filenames[i]
        query_im = Image.open(os.path.join(self.skt_root, query_impath)).convert('RGB')
        query_im = self.render_transforms(query_im)
        ims = torch.cat([
            torch.cat([
                self.render_transforms(
                    Image.open(x).convert('RGB')
                ).unsqueeze(0)
                for x in views
            ]).unsqueeze(0)
            for views in data
        ])
        return {
            "object_im": ims,
            "query_im": query_im,
        }

    def collate_fn(self, batch):
        batch_dict = {
            "object_ims": torch.stack([x['object_im'] for x in batch]),
            "query_ims": torch.stack([x['query_im'] for x in batch]),
        }
        return batch_dict



class SHREC23_Rings_RenderOnly_TextQuery(BaseShrecDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
                 
    def __getitem__(self, i):
        query_text = self.tex[i]
        data = self.data[i]['render']
        ims = torch.cat([
            torch.cat([
                self.render_transforms(
                    Image.open(x).convert('RGB')
                ).unsqueeze(0)
                for x in views
            ]).unsqueeze(0)
            for views in data
        ])

        return {
            "object_im": ims,
            "query_text": query_text,
        }
    
    def collate_fn(self, batch):
        batch_dict = {
            "object_ims": torch.stack([x['object_im'] for x in batch]),
            "query_texts": [x['query_text'] for x in batch],
        }

        batch_dict["tokens"] = self.tokenizer.batch_encode_plus(
            batch_dict["query_texts"], padding="longest", return_tensors="pt"
        )
        return batch_dict
        

if __name__ == '__main__':

    print('='*80)
    ds = SHREC23_Rings_RenderOnly_ImageQuery(
        'data/train.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    print('='*80)
    ds2 = SHREC23_Rings_RenderOnly_TextQuery(
        'data/train.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    print('='*80)
    ds = SHREC23_Rings_RenderOnly_ImageQuery(
        'data/test.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    print('='*80)
    ds2 = SHREC23_Rings_RenderOnly_TextQuery(
        'data/test.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    
    print('='*80)
    DatasetItemInfo(ds[0])
    print('='*80)
    DatasetItemInfo(ds2[0])
    
    dl = data.DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    dl2 = data.DataLoader(ds2, batch_size=2, collate_fn=ds2.collate_fn)

    print('='*80)
    DatasetBatchInfo(next(iter(dl)))
    print('='*80)
    DatasetBatchInfo(next(iter(dl2)))
    print('='*80)