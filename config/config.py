import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(work_dir)
import argparse

# 训练使用的相关参数
def get_parser():

    # Arguments
    parser = argparse.ArgumentParser()
    
    # PointNet
    parser.add_argument('--mode', type=str, default='train', help='run mode') #model  
    parser.add_argument('--outf_Pointnet', type=str, default='./checkpoints', help='output folder for the Pointnet_para') #model  
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")#model
    parser.add_argument('--nepoch', type=int, default='100', help='number of epoch of training')#dataset
    parser.add_argument('--batch_size', type=int, default=50, help='batch_size for single GPU') #dataset
    parser.add_argument('--num_works', type=int, default=1, help='num_works for dataset')#dataset
    parser.add_argument('--downsample_number', type=int, default='2048', help='downsample to a certain number')#dataset

    args = parser.parse_args()
    return args

cfg = get_parser()

