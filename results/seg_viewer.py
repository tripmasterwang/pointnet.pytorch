import os
import numpy as np
import open3d as o3d

# 设置文件路径
points_dir = 'points'
labels_dir = 'points_label'

# 列出文件并排序
point_files = sorted([f for f in os.listdir(points_dir) if f.endswith('.npy')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.seg')])

# 检查文件数量是否匹配
assert len(point_files) == len(label_files), "Error: Mismatch between points and labels file count."

# 设置颜色映射，每个分割部分分配一个唯一颜色
num_parts = 10  # 根据具体的分割类别数设置或动态计算
colors = np.random.rand(num_parts, 3)  # 随机生成颜色

# 可视化函数
def visualize_point_cloud(points, labels, colors):
    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 应用颜色
    label_colors = colors[labels]  # 根据标签分配颜色
    point_cloud.colors = o3d.utility.Vector3dVector(label_colors)

    # 显示点云
    o3d.visualization.draw_geometries([point_cloud])

# 遍历成对的点云文件和标签文件
for point_file, label_file in zip(point_files, label_files):
    # 读取点云和标签数据
    points = np.load(os.path.join(points_dir, point_file))
    labels = np.loadtxt(os.path.join(labels_dir, label_file), dtype=int)

    # 可视化点云和标签
    visualize_point_cloud(points, labels, colors)
