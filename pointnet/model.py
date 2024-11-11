from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# 编码器
class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        # 用于特征提取的MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.global_feat = global_feat

    def forward(self, x):
        batch_size, num_points, num_dims = x.size()

        x = x.view(-1, num_dims)  # [B * N, 3]
        x = self.mlp1(x)          # [B * N, 64]
        pointwise_feat = x.view(batch_size, num_points, 64)  # 重塑以便后续拼接

        # 继续使用MLP提取高维特征
        x = self.mlp2(x)          # [B * N, 128]
        x = self.mlp3(x)          # [B * N, 1024]
        x = x.view(batch_size, num_points, 1024)  # 重塑以便最大池化

        # 使用最大池化提取全局特征
        x = torch.max(x, 1, keepdim=True)[0]  # [B, 1, 1024]
        x = x.view(batch_size, 1024)  # 全局特征 [B, 1024]

        if self.global_feat:
            return x
        else:
            # 为每个点重复全局特征并与逐点特征拼接
            x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)  # [B, 1024, N]
            x = torch.cat([x, pointwise_feat.transpose(1, 2)], 1)     # 拼接为 [B, 1088, N]
            return x

# 用于点云分类任务
class PointNetCls(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 用于点云分割任务
class PointNetDenseCls(nn.Module):
    def __init__(self, k=2):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=False)
        # 用于预测每个点的分割标签的MLP
        self.mlp = nn.Sequential(
            nn.Linear(1088, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, k)
        )

    def forward(self, x):
        batch_size, num_points, num_dims = x.size()
        x = self.feat(x)  # 输出为 [B, 1088, N]
        x = x.transpose(1, 2).contiguous().view(-1, 1088)  # 将输入展平为MLP输入 [B * N, 1088]
        x = self.mlp(x)
        x = x.view(batch_size, num_points, self.k)
        return x


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    
    cls = PointNetCls(k=11)
    out = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k=3)
    out = seg(sim_data)
    print('seg', out.size())