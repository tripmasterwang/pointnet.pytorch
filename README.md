# about copyright
This project is only intended to help students quickly get started using deep learning to process point cloud tasks. After completing the course tasks, this project will be deleted.

# PointNet.pytorch
这个仓库是PointNet在pytorch中的实现。这个项目只是为了帮助学生快速开始点云深度学习的预训练项目。因此，大部分代码已经简化或删除。如果你对PointNet有进一步的兴趣，可以从以下链接获取完整版本的项目代码：https://github.com/fxia22/pointnet.pytorch

# 项目文件介绍
项目的参数写在 `/config/config.py` 中，里面主要放置项目参数，以提高便捷性。不至于想变更一些关键参数的时候翻很久的代码

模型在 `/pointnet/model.py` 中。MLP是通过Linear函数实现的。然而，如果你阅读作者的代码，你会发现MLP是通过1d卷积实现的。这可能看起来很奇怪，但表示的意义是相同的。

数据相关的文件在 `/datasets/ShapeNetCore/`中。`transform.py`具体定义了点云的归一化，旋转等操作；`dataset_seg.py` 包含了对shapenetcore_partanno_segmentation_benchmark_v0数据集的读取，预处理，以及dataloader迭代器的构建。

训练的结果存储在 `/checkpoints` 中，在训练完成后，这个文件夹会包含3个path文件，因为实际上对于每个类都要重新训练一个模型，即相同的模型结构下，不同类对应了不同的权重参数。

最后在测试过程中产生的结果放置在 `/results/seg_viewer` 中， 用于可视化分割结果，如果你使用远程服务器训练模型，记得把results文件夹下载到本地来运行，因为远程服务器运行该程序时无法看到可视化窗口。

# 下载项目
git clone https://github.com/tripmasterwang/pointnet.pytorch.git

# 环境配置
环境中最难配置的是pytorch，如果已经有可以使用pytorch的环境，就使用那个环境。然后
cd 项目目录
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
即可

如果没有可以使用pytorch的环境。先添加清华镜像源，然后创建一个python版本3.7的环境：
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda create -n pointnet python=3.7 -y
conda activate pointnet
cd 项目目录
pip install -e .
然后参照链接内的pdf文件安装pytorch：
通过百度网盘分享的文件：Python深度学习(1)：安装Anaconda、PyTorch（GP...
链接：https://pan.baidu.com/s/12ohIqsO4ACIfDfkJLaZ8Ww?pwd=sewh 
提取码：sewh 
--来自百度网盘超级会员V7的分享



# 下载数据

通过百度网盘分享的文件：shapenetcore_partanno_segmentation_...
链接：https://pan.baidu.com/s/1jmRjhqiz_v1A2tM3YTRO3w?pwd=tz47 
提取码：tz47 
--来自百度网盘超级会员V7的分享

数据集放在datasets/ShapeNetCore中，范例：
-/pointnet.pytorch
    --/datasets
    --/ShapeNetCore
        --/shapenetcore_partanno_segmentation_benchmark_v0
        --/02691156
        --/02773838
        --/02954340
        --/synsetoffset2category
        --/dataset_seg.py
        --/transform.py

# 分割性能

Testing category 02691156 with 4 parts
Test Accuracy: 75.34%

Testing category 02773838 with 2 parts
Test Accuracy: 85.84%

Testing category 02954340 with 2 parts
Test Accuracy: 88.93%

