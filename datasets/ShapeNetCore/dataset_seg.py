import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(work_dir)
import glob
import torch
import random
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import datasets.ShapeNetCore.transform as aug_transform
from config.config import cfg

def fps_sampling(pcd, num_samples, return_indices=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pcd_tensor = torch.from_numpy(pcd).to(device)

    N, _ = pcd_tensor.shape
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * float('inf')

    sampled_indices[0] = torch.randint(0, N, (1,), device=device)
    for i in range(1, num_samples):
        current_point = pcd_tensor[sampled_indices[i - 1]].unsqueeze(0)
        dist = torch.cdist(current_point, pcd_tensor, p=2).squeeze(0)
        distances = torch.min(distances, dist)
        sampled_indices[i] = torch.argmax(distances)

    if return_indices:
        return sampled_indices.cpu().numpy()
    else:
        sampled_pcd_tensor = pcd_tensor[sampled_indices]
        sampled_pcd = sampled_pcd_tensor.cpu().numpy()
        return sampled_pcd


class ShapeNetCore(Dataset):
    def __init__(self, pcds, seg_labels, cfg):
        self.cfg = cfg
        self.pcds = pcds
        self.seg_labels = seg_labels
    
        self.NormalizeCoord = aug_transform.NormalizeCoord()
        self.CenterShift = aug_transform.CenterShift(apply_z=True)
        self.RandomRotate_z = aug_transform.RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0)
        self.RandomRotate_y = aug_transform.RandomRotate(angle=[-1, 1], axis="y", p=1.0)
        self.RandomRotate_x = aug_transform.RandomRotate(angle=[-1, 1], axis="x", p=1.0)

        self.train_aug_compose = aug_transform.Compose([self.CenterShift, self.RandomRotate_z, self.RandomRotate_y, self.RandomRotate_x,
                                                        self.NormalizeCoord])
        self.test_aug_compose = aug_transform.Compose([self.CenterShift, self.NormalizeCoord])
        
    def __len__(self):
        return len(self.pcds)
    
    
    def __getitem__(self, idx):
        
        pcd = self.pcds[idx]
        seg = self.seg_labels[idx]
        if cfg.mode == 'train':
            pcd = self.train_aug_compose(pcd)
        elif cfg.mode == 'test':
            pcd = self.test_aug_compose(pcd)
            
        return torch.tensor(pcd, dtype=torch.float32), torch.tensor(seg, dtype=torch.long)

    def worker_init_fn(self, worker_id):
        torch_seed = torch.initial_seed() 
        np_seed = torch_seed // 2 ** 32 - 1 - worker_id
        np.random.seed(np_seed)
        random.seed(np_seed)

batch_size = cfg.batch_size
dataset_workers = cfg.num_works

synsetoffset2category = np.loadtxt('datasets/ShapeNetCore/shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt', dtype=str)
synsetoffset2category = dict(synsetoffset2category)

npy_files = [os.path.basename(f) for f in glob.glob(f'datasets/ShapeNetCore/ShapeNetCoreNPY_{cfg.downsample_number}/*.npy')]
train_dataloaders = {}
test_dataloaders = {}

output_file = os.path.join(work_dir, 'datasets/ShapeNetCore/num_seg_classes.txt')
with open(output_file, 'w') as f:
    for ID in tqdm(synsetoffset2category.values(), desc="Processing Categories"):
        if f'{ID}_pcds.npy' not in npy_files:
            pcds = []
            seg_labels = []
            pts_files = sorted(glob.glob(f'datasets/ShapeNetCore/shapenetcore_partanno_segmentation_benchmark_v0/{ID}/points/*.pts'))
            seg_files = sorted(glob.glob(f'datasets/ShapeNetCore/shapenetcore_partanno_segmentation_benchmark_v0/{ID}/points_label/*.seg'))
            
            max_seg_classes = 0
            for pts_file, seg_file in tqdm(zip(pts_files, seg_files), desc='processing files'):
                pcd = np.loadtxt(pts_file, delimiter=' ', skiprows=0)
                seg = np.loadtxt(seg_file, dtype=int)

                # 更新当前类别的最大分割类数
                unique_labels = len(np.unique(seg))
                if unique_labels > max_seg_classes:
                    max_seg_classes = unique_labels

                # 使用 fps_sampling 返回采样的点以及对应的采样索引
                sampled_indices = fps_sampling(pcd, cfg.downsample_number, return_indices=True)
                sampled_pcd = pcd[sampled_indices]  
                sampled_seg = seg[sampled_indices]  # 根据采样的索引获取对应的分割标签

                pcds.append(sampled_pcd)
                seg_labels.append(sampled_seg)
                
            pcds = np.array(pcds)
            seg_labels = np.array(seg_labels) - 1
            
            print(f"Category {ID}: {max_seg_classes} unique segmentation classes")
            f.write(f"{ID}\t{max_seg_classes}\n")
                
            save_dir = f'datasets/ShapeNetCore/ShapeNetCoreNPY_{cfg.downsample_number}'
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f'{ID}_pcds.npy'), np.array(pcds))
            np.save(os.path.join(save_dir, f'{ID}_seg_labels.npy'), np.array(seg_labels))
        else:
            pcds = np.load(f'datasets/ShapeNetCore/ShapeNetCoreNPY_{cfg.downsample_number}/{ID}_pcds.npy', allow_pickle=True)
            seg_labels = np.load(f'datasets/ShapeNetCore/ShapeNetCoreNPY_{cfg.downsample_number}/{ID}_seg_labels.npy', allow_pickle=True)
            seg_labels = seg_labels 
            unique_labels = len(np.unique(seg_labels[0]))
            print(f"Category {ID}: {unique_labels} unique segmentation classes")
            f.write(f"{ID}\t{unique_labels}\n")
    
        train_ratio = 0.95
        train_num = int(len(pcds) * train_ratio)
        train_index = np.random.choice(len(pcds), train_num, replace=False)

        train_pcds = pcds[train_index]
        train_seg_labels = seg_labels[train_index]

        train_dataset = ShapeNetCore(train_pcds, train_seg_labels, cfg) #这有问题
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, 
                                                num_workers=cfg.num_works,
                                                shuffle=True, sampler=None,
                                                drop_last=True, pin_memory=False,
                                                worker_init_fn=train_dataset.worker_init_fn)
        train_dataloaders[ID] = train_dataloader
        
        test_pcds = np.delete(pcds, train_index, axis=0)
        test_seg_labels = np.delete(seg_labels, train_index, axis=0)
        test_dataset = ShapeNetCore(test_pcds, test_seg_labels, cfg) #这有问题
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, 
                                                num_workers=cfg.num_works,
                                                shuffle=False, sampler=None,
                                                drop_last=False, pin_memory=False,
                                                worker_init_fn=train_dataset.worker_init_fn)
        test_dataloaders[ID] = test_dataloader

# 仅当单独测试本文件时，才会运行以下代码
if __name__ == '__main__':
    # 检查数据加载器
    print("Testing train dataloader...")
    for batch_idx, (pcd_batch, seg_batch) in enumerate(train_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Point Cloud Batch Shape: {pcd_batch.shape}")
        print(f"  Segmentation Label Batch Shape: {seg_batch.shape}")

        # 检查点云和标签的形状是否一致
        assert pcd_batch.shape[0] == seg_batch.shape[0], "Batch size mismatch between point cloud and segmentation labels"
        assert pcd_batch.shape[1] == cfg.downsample_number, "Number of points per sample does not match downsampled size"
        assert pcd_batch.shape[1] == seg_batch.shape[1], "Mismatch in number of points and labels per sample"

        # 只测试前几个批次
        if batch_idx >= 2:
            break

    print("Testing test dataloader...")
    for batch_idx, (pcd_batch, seg_batch) in enumerate(test_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Point Cloud Batch Shape: {pcd_batch.shape}")
        print(f"  Segmentation Label Batch Shape: {seg_batch.shape}")

        # 检查点云和标签的形状是否一致
        assert pcd_batch.shape[0] == seg_batch.shape[0], "Batch size mismatch between point cloud and segmentation labels"
        assert pcd_batch.shape[1] == cfg.downsample_number, "Number of points per sample does not match downsampled size"
        assert pcd_batch.shape[1] == seg_batch.shape[1], "Mismatch in number of points and labels per sample"

        # 只测试前几个批次
        if batch_idx >= 2:
            break

    print("All tests passed successfully.")