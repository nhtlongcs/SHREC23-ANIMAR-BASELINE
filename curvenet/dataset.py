import os 
from PIL import Image
import random
from pathlib import Path
import pandas as pd 
import numpy as np
import torch
from torchvision import transforms as tvtf
from torch.utils import data
from torchvision import transforms
from transformers import AutoTokenizer


class SimpleSampler:
    """
    Randomly sample N points from a point cloud.
    If the number of points is not enough, duplicate the points up to N points.
    """
    def __init__(self, N):
        self.N = N

    def __call__(self, points):
        assert len(points.shape) == 2
        n_points = points.shape[0]
        while points.shape[0] < self.N:
            points = np.concatenate([points, points[:n_points, :]], axis=0)
        indices = np.array(sorted(random.sample(range(0, points.shape[0]), self.N)))
        try:
            return points[indices, :]
        except:
            print(points.shape)
            print(indices.shape)
            raise


class PointCloudTranslate:
    def __call__(self, pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
          
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud    


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        pointcloud = np.transpose(pointcloud, (1, 0))
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([
                                SimpleSampler(1024),
                                PointCloudTranslate(),
                                ToTensor(),
                              ])

def read_obj(filepath: Path): 
    with open(filepath) as f:
        lines = f.readlines()

    verts = [tuple(map(float, line.split()[1:]))
             for line in lines 
             if line.startswith("v ")]
    return verts
    
class PointCloudData(data.Dataset):
    def __init__(self, data_path, ids=None, to_return_ids=False, is_test=False):
        self.is_test = is_test
        self.to_return_ids = to_return_ids
        data_path = Path(data_path)

        files = list(data_path.glob("*.obj"))
        self.data = {
            file.stem : {
                "verts": read_obj(file)
            }
            for file in files
        }

        if ids is None:    
            ids = self.data.keys()

        self.ids = ids
        self.index2id = {}
        for index, id in enumerate(self.ids):
            self.index2id[index] = id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.index2id[idx]
        points = np.array(self.data[id]['verts'])
        # if self.is_test:
        #     label = 0
        # else:
        #     label = int(self.h5[id]['label'][...])
        if not self.to_return_ids:
            return {
                'pointcloud': default_transforms()(points), 
                # 'category': label,
            }
        else:
            return {
                'pointcloud': default_transforms()(points), 
                # 'category': label,
                'id': id,
            }
class SHREC23_PointCloudData_ImageQuery(data.Dataset):
    def __init__(self, obj_data_path, csv_data_path, skt_root, ids=None):
        self.csv_data = pd.read_csv(csv_data_path)
        self.skt_root = skt_root
        self.ids = self.csv_data.index
        self.obj_data_path = obj_data_path
        self.render_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        obj_id = self.csv_data.iloc[idx]['obj_id']
        skt_id = self.csv_data.iloc[idx]['sketch_id']
        points = read_obj(os.path.join(self.obj_data_path, obj_id + '.obj'))
        points = np.array(points)
        points = default_transforms()(points)
        
        query_impath = self.csv_data.iloc[idx]['sket_filename']
        query_im = Image.open(os.path.join(self.skt_root, query_impath)).convert('RGB')
        query_im = self.render_transforms(query_im)
        
        return {
            "pointcloud": points,
            "query_im": query_im,
            "gallery_id": obj_id,
            "query_id": skt_id,
        }
    def collate_fn(self, batch):
        batch = {
            'pointclouds': torch.stack([item['pointcloud'] for item in batch]),
            'query_ims': torch.stack([item['query_im'] for item in batch]),
            'gallery_ids': [item['gallery_id'] for item in batch],
            'query_ids': [item['query_id'] for item in batch],
        }
        return batch

class SHREC23_PointCloudData_TextQuery(data.Dataset):
    def __init__(self, obj_data_path, csv_data_path, ids=None):
        self.csv_data = pd.read_csv(csv_data_path)
        self.ids = self.csv_data.index
        self.obj_data_path = obj_data_path
        self.render_transforms = tvtf.Compose([
                tvtf.CenterCrop((352, 352)),
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            ])
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        obj_id = self.csv_data.iloc[idx]['obj_id']
        txt_id = self.csv_data.iloc[idx]['text_id']
        query_text = self.csv_data.iloc[idx]['tex']

        points = read_obj(os.path.join(self.obj_data_path, obj_id + '.obj'))
        points = np.array(points)
        points = default_transforms()(points)
        
        return {
            "pointcloud": points,
            "query_text": query_text,
            "gallery_id": obj_id,
            "query_id": txt_id,
        }
    
    def collate_fn(self, batch):
        batch = {
            'pointclouds': torch.stack([item['pointcloud'] for item in batch]),
            "query_texts": [x['query_text'] for x in batch],
            'gallery_ids': [item['gallery_id'] for item in batch],
            'query_ids': [item['query_id'] for item in batch],
        }

        batch["tokens"] = self.tokenizer.batch_encode_plus(
            batch["query_texts"], padding="longest", return_tensors="pt"
        )
        return batch

def DatasetItemInfo(item, indent=0):
    if isinstance(item, dict):
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

if __name__ == "__main__":
    dataset = SHREC23_PointCloudData_ImageQuery(obj_data_path='data/SketchANIMAR2023/3D_Model_References/References',
                                                csv_data_path='data/csv/train_skt.csv',
                                                skt_root='data/SketchANIMAR2023/Train/SketchQuery_Train')
    print(len(dataset))
    DatasetItemInfo(dataset[0])
