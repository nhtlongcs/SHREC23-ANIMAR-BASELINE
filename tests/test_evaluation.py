from common.test import test_loop

def test_txt_ringview():
    from torch.utils.data import DataLoader
    from ringnet.dataset import SHREC23_Rings_RenderOnly_TextQuery
    from ringnet.models import Base3DObjectRingsExtractor
    from common.models import BertExtractor, MLP
    
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

    test_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
              obj_input='object_ims', query_input='tokens',
              dl=dl,
              dimension=latent_dim,
              device=device)

def test_image_ringview():
    from torch.utils.data import DataLoader
    from ringnet.dataset import SHREC23_Rings_RenderOnly_ImageQuery
    from ringnet.models import Base3DObjectRingsExtractor
    from common.models import ResNetExtractor, MLP
    
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

    test_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
              obj_input='object_ims', query_input='query_ims',
              dl=dl,
              dimension=latent_dim,
              device=device)
        
def test_txt_pcl():

    from torch.utils.data import DataLoader
    from curvenet.dataset import SHREC23_PointCloudData_TextQuery
    from curvenet.models import CurveNet
    from common.models import BertExtractor, MLP
    import torch 

    batch_size = 4
    latent_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    test_ds = SHREC23_PointCloudData_TextQuery(obj_data_path='data/SketchANIMAR2023/3D_Model_References/References',
                                             csv_data_path='data/csv/test_tex.csv')

    dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=test_ds.collate_fn)
    

    obj_extractor = CurveNet(device=device)

    obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)
    query_extractor = BertExtractor(is_frozen=True) # OOM, so freeze for baseline
    query_embedder = MLP(query_extractor, latent_dim=latent_dim).to(device)

    test_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
              obj_input='pointclouds', query_input='tokens',
              dl=dl,
              dimension=latent_dim,
              device=device)


def test_image_pcl():

    from torch.utils.data import DataLoader
    from curvenet.dataset import SHREC23_PointCloudData_ImageQuery
    from curvenet.models import CurveNet
    from common.models import ResNetExtractor, MLP
    import torch 

    batch_size = 4
    latent_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    test_ds = SHREC23_PointCloudData_ImageQuery(obj_data_path='data/SketchANIMAR2023/3D_Model_References/References',
                                             csv_data_path='data/csv/test_skt.csv',
                                             skt_root='data/SketchANIMAR2023/Train/SketchQuery_Train')


    dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=test_ds.collate_fn)
    

    obj_extractor = CurveNet(device=device)

    obj_embedder = MLP(obj_extractor, latent_dim=latent_dim).to(device)
    query_extractor = ResNetExtractor(is_frozen=True) # OOM, so freeze for baseline
    query_embedder = MLP(query_extractor, latent_dim=latent_dim).to(device)

    test_loop(obj_embedder = obj_embedder, query_embedder = query_embedder,
              obj_input='pointclouds', query_input='query_ims',
              dl=dl,
              dimension=latent_dim,
              device=device)


if __name__ == '__main__':
    test_txt_pcl()
    test_image_pcl()
    test_txt_ringview()
    test_image_ringview()