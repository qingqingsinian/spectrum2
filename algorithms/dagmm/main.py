import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
import numpy as np
import random
def str2bool(v):
    return v.lower() in ('true')
# 添加input_dim参数到配置中
def main(config):
    # For fast training
    cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)

    # 根据选择的数据集设置数据路径和输入维度
    dataset_config = {
        'smd': {'path': './processed_data/smd_standardized.npz', 'dim': 38},
        'smap': {'path': './processed_data/smap_standardized.npz', 'dim': 25},
        'psm': {'path': './processed_data/psm_standardized.npz', 'dim': 25},
        'msl': {'path': './processed_data/msl_standardized.npz', 'dim': 55},
        'swat': {'path': './processed_data/swat_standardized.npz', 'dim': 51},
        'kpi': {'path': './processed_data/kpi_standardized.npz', 'dim': 1}  # 添加KPI数据集配置
    }
    
    # 设置数据路径和输入维度
    if config.dataset in dataset_config:
        config.data_path = dataset_config[config.dataset]['path']
        config.input_dim = dataset_config[config.dataset]['dim']
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}. Supported datasets: {list(dataset_config.keys())}")

    data_loader = get_loader(config.data_path, batch_size=config.batch_size, mode=config.mode)
    
    # Solver
    solver = Solver(data_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver

# 在参数解析部分添加input_dim参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input_dim', type=int, default=118, help='输入数据的维度')

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gmm_k', type=int, default=3)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--pretrained_model', type=str, default='200_4')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    
    # 数据集选择参数
    parser.add_argument('--dataset', type=str, default='smd', 
                        choices=['smd', 'smap', 'psm', 'msl', 'swat','kpi'],
                        help='选择要使用的数据集')

    # Path
    parser.add_argument('--data_path', type=str, default='kdd_cup.npz')
    parser.add_argument('--log_path', type=str, default='./dagmm/logs')
    parser.add_argument('--model_save_path', type=str, default='./dagmm/models')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=50)
    parser.add_argument('--model_save_step', type=int, default=50)

    config = parser.parse_args()
 
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)