from config.config import cfg
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pointnet.model import PointNetDenseCls
from datasets.ShapeNetCore.dataset_seg import train_dataloaders, test_dataloaders

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.makedirs(cfg.outf_Pointnet, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载类别和分割部分信息
def load_seg_classes(file_path):
    seg_classes = {}
    with open(file_path, 'r') as f:
        for line in f:
            category, num_parts = line.strip().split()
            seg_classes[category] = int(num_parts)
    return seg_classes

# 加载 num_seg_classes.txt 文件
seg_classes = load_seg_classes('datasets/ShapeNetCore/num_seg_classes.txt')

# 为每个类别初始化独立的模型实例
models = {}
optimizers = {}
schedulers = {}

for category, num_parts in seg_classes.items():
    model = PointNetDenseCls(k=num_parts).to(device)  # 使用类别的分割部分数
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    models[category] = model
    optimizers[category] = optimizer
    schedulers[category] = scheduler

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练单个类别模型的函数
def train_one_epoch_for_category(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for pcd_batch, seg_batch in dataloader:
        pcd_batch, seg_batch = pcd_batch.to(device), seg_batch.to(device)

        # 检查标签是否在范围内
        assert seg_batch.max() < model.k, f"Label value {seg_batch.max()} out of range for model with {model.k} classes"
        assert seg_batch.min() >= 0, f"Label value {seg_batch.min()} is negative"

        optimizer.zero_grad()
        outputs = model(pcd_batch)
        # 检查下这里的形状
        loss = criterion(outputs.view(-1, model.k), seg_batch.view(-1))  # 使用模型对应的 k
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# 主训练循环
for category in seg_classes.keys():
    print(f"Training category {category} with {seg_classes[category]} parts")

    model = models[category]
    optimizer = optimizers[category]
    scheduler = schedulers[category]

    for epoch in range(cfg.nepoch):
        print(f"Epoch {epoch + 1}/{cfg.nepoch}")
        # 训练当前类别的模型
        train_loss = train_one_epoch_for_category(model, train_dataloaders[category], criterion, optimizer, device)
        print(f"Category {category} Train Loss: {train_loss:.4f}")

        # 更新学习率调度器
        scheduler.step()

    # 保存当前类别的模型
    save_path = f"checkpoints/pointnet_seg_{category}_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model for category {category} saved at {save_path}")

print("Training complete.")


######以下代码用于测试#######
save_dir = 'results'
os.makedirs(os.path.join(save_dir,'points'), exist_ok=True)
os.makedirs(os.path.join(save_dir,'points_label'), exist_ok=True)

# 测试单个类别模型的函数
def test_one_epoch_for_category(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total_points = 0

    with torch.no_grad():
        sample_idx = 0  # 初始化全局样本索引

        for pcd_batch, seg_batch in dataloader:
            pcd_batch, seg_batch = pcd_batch.to(device), seg_batch.to(device)

            outputs = model(pcd_batch)  # [B, N, num_parts]

            # 获取预测结果并计算准确率
            preds = outputs.max(dim=-1)[1]  # 获取每个点的预测标签 [B, N]
            correct += (preds == seg_batch).sum().item()
            total_points += seg_batch.numel()

            # 遍历 batch 内的每个样本，保存点云和预测标签
            for i in range(pcd_batch.shape[0]):
                points = pcd_batch[i].cpu().numpy()  # 获取点云 [N, 3]
                pred_labels = preds[i].cpu().numpy()  # 获取预测标签 [N]

                # 保存点云为 .npy 文件
                np.save(os.path.join(save_dir, 'points', f"{category}_sample{sample_idx}_points.npy"), points)

                # 保存预测标签为 .seg 文件
                np.savetxt(os.path.join(save_dir,'points_label', f"{category}_sample{sample_idx}_pred.seg"), pred_labels, fmt='%d')

                sample_idx += 1  # 增加样本索引以确保唯一

        accuracy = correct / total_points if total_points > 0 else 0
        return accuracy

# 主测试循环
for category, num_parts in seg_classes.items():
    print(f"Testing category {category} with {num_parts} parts")

    # 加载模型
    model = PointNetDenseCls(k=num_parts).to(device)
    model_path = f"checkpoints/pointnet_seg_{category}_epoch_{cfg.nepoch}.pth"
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found for category {category}. Skipping.")
        continue

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 测试当前类别的模型
    test_dataloader = test_dataloaders[category]
    test_accuracy = test_one_epoch_for_category(model, test_dataloader, criterion, device)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("Testing complete.")