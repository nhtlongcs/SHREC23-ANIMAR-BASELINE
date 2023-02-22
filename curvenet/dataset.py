import random
from pathlib import Path

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm


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
        
if __name__ == "__main__":
    dataset = PointCloudData("data/ANIMAR_Preliminary_Data/3D_Models")