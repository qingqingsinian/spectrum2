import os
import torch
import argparse
import sensitive_hue
import numpy as np
import torch.optim as optim
import pandas as pd
import time
from config.parser import YAMLParser
from base.dataset import ADataset, split_dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

def process_smd_dataset(data_dir, output_dir):
    """
    处理SMD数据集，将测试标签合并到测试数据中
    
    Args:
        data_dir: SMD原始数据目录路径
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    train_data = np.load(os.path.join(data_dir, 'SMD_train.npy'))
    test_data = np.load(os.path.join(data_dir, 'SMD_test.npy'))
    test_labels = np.load(os.path.join(data_dir, 'SMD_test_label.npy'))
    
    # 保存训练数据
    np.save(os.path.join(output_dir, 'train.npy'), train_data)
    
    # 将测试标签合并到测试数据中，保存为npz文件
    np.savez(os.path.join(output_dir, 'test.npz'), x=test_data, y=test_labels)
    
    print(f"Processed SMD dataset:")
    print(f"  Train shape: {train_data.shape}")
    print(f"  Test shape: {test_data.shape}")
    print(f"  Label shape: {test_labels.shape}")


def process_smap_dataset(data_dir, output_dir):
    """
    处理SMAP数据集，将测试标签合并到测试数据中
    
    Args:
        data_dir: SMAP原始数据目录路径
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    train_data = np.load(os.path.join(data_dir, 'SMAP_train.npy'))
    test_data = np.load(os.path.join(data_dir, 'SMAP_test.npy'))
    test_labels = np.load(os.path.join(data_dir, 'SMAP_test_label.npy'))
    
    # 保存训练数据
    np.save(os.path.join(output_dir, 'train.npy'), train_data)
    
    # 将测试标签合并到测试数据中，保存为npz文件
    np.savez(os.path.join(output_dir, 'test.npz'), x=test_data, y=test_labels)
    
    print(f"Processed SMAP dataset:")
    print(f"  Train shape: {train_data.shape}")
    print(f"  Test shape: {test_data.shape}")
    print(f"  Label shape: {test_labels.shape}")


def process_msl_dataset(data_dir, output_dir):
    """
    处理MSL数据集，将测试标签合并到测试数据中
    
    Args:
        data_dir: MSL原始数据目录路径
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    train_data = np.load(os.path.join(data_dir, 'MSL_train.npy'))
    test_data = np.load(os.path.join(data_dir, 'MSL_test.npy'))
    test_labels = np.load(os.path.join(data_dir, 'MSL_test_label.npy'))
    
    # 保存训练数据
    np.save(os.path.join(output_dir, 'train.npy'), train_data)
    
    # 将测试标签合并到测试数据中，保存为npz文件
    np.savez(os.path.join(output_dir, 'test.npz'), x=test_data, y=test_labels)
    
    print(f"Processed MSL dataset:")
    print(f"  Train shape: {train_data.shape}")
    print(f"  Test shape: {test_data.shape}")
    print(f"  Label shape: {test_labels.shape}")


def process_psm_dataset(data_dir, output_dir):
    """
    处理PSM数据集，将其转换为项目所需的格式
    
    Args:
        data_dir: PSM原始数据目录路径
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test_labels = pd.read_csv(os.path.join(data_dir, 'test_label.csv'))
    
    # 移除第一列（时间戳）
    train_values = train_data.values[:, 1:].astype(np.float32)
    test_values = test_data.values[:, 1:].astype(np.float32)
    train_values = np.nan_to_num(train_values, nan=0.0, posinf=1e6, neginf=-1e6)
    test_values = np.nan_to_num(test_values, nan=0.0, posinf=1e6, neginf=-1e6)
    test_label_values = test_labels.values[:, 1:].astype(np.int32)
    test_label_values = test_label_values.squeeze()  # 转换为一维数组
    

    
    # 保存训练数据
    np.save(os.path.join(output_dir, 'train.npy'), train_values)
    
    # 保存测试数据和标签到npz文件
    np.savez(os.path.join(output_dir, 'test.npz'), x=test_values, y=test_label_values)
    
    print(f"Processed PSM dataset:")
    print(f"  Train shape: {train_values.shape}")
    print(f"  Test shape: {test_values.shape}")
    print(f"  Label shape: {test_label_values.shape}")


def process_swat_dataset(data_dir, output_dir):
    """
    处理SWaT数据集
    
    Args:
        data_dir: SWaT原始数据目录路径
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    train_data = pd.read_csv(os.path.join(data_dir, 'swat_train2.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'swat2.csv'))
    
    # 分离特征和标签（最后一列是标签）
    train_values = train_data.values[:, :-1]
    test_values = test_data.values[:, :-1]
    test_labels = test_data.values[:, -1:]  # 保持二维数组
 
    # 保存训练数据
    np.save(os.path.join(output_dir, 'train.npy'), train_values)
    
    # 保存测试数据和标签到npz文件
    np.savez(os.path.join(output_dir, 'test.npz'), x=test_values, y=test_labels)
    
    print(f"Processed SWaT dataset:")
    print(f"  Train shape: {train_values.shape}")
    print(f"  Test shape: {test_values.shape}")
    print(f"  Label shape: {test_labels.shape}")


def prepare_dataset(data_args, data_name):
    """
    根据数据集名称准备相应的数据
    
    Args:
        data_args: 数据集配置参数
        data_name: 数据集名称
    """
    # 检查是否需要处理数据
    train_file = os.path.join(data_args.data_dir, 'train.npy')
    test_file = os.path.join(data_args.data_dir, 'test.npz')
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"Dataset {data_name} already processed. Skipping...")
        return
    
    # 根据数据集名称调用相应的处理函数
    if data_name == 'SMD' and hasattr(data_args, 'raw_data_dir'):
        process_smd_dataset(data_args.raw_data_dir, data_args.data_dir)
    elif data_name == 'SMAP' and hasattr(data_args, 'raw_data_dir'):
        process_smap_dataset(data_args.raw_data_dir, data_args.data_dir)
    elif data_name == 'MSL' and hasattr(data_args, 'raw_data_dir'):
        process_msl_dataset(data_args.raw_data_dir, data_args.data_dir)
    elif data_name == 'PSM' and hasattr(data_args, 'raw_data_dir'):
        process_psm_dataset(data_args.raw_data_dir, data_args.data_dir)
    elif data_name == 'SWaT' and hasattr(data_args, 'raw_data_dir'):
        process_swat_dataset(data_args.raw_data_dir, data_args.data_dir)
    else:
        print(f"No preprocessing function found for dataset {data_name} or raw_data_dir not specified")


def get_data_loaders(data_args, args, scaler):
    dataset_names = ('train.npy', 'test.npz')
    keys = (None, ('x', 'y'))

    data_loaders = []
    for name, key in zip(dataset_names, keys):
        # 使用配置文件中的stride参数，如果不存在则默认为1
        stride = getattr(data_args, 'stride', 1) if name.startswith('train') else 1
        dataset = ADataset(os.path.join(data_args.data_dir, name), data_args.step_num_in, stride, keys=key)
        if name.startswith('train'):
            if scaler is not None:
                dataset.fit_transform(scaler.fit_transform)
            random_val = getattr(data_args, 'random_val', False)
            for d_set in split_dataset(dataset, data_args.val_ratio, random_val):
                data_loader = DataLoader(d_set, batch_size=data_args.batch_size, shuffle=True)
                data_loaders.append(data_loader)
        else:
            dataset.fit_transform(scaler.transform)
            data_loader = DataLoader(dataset, batch_size=data_args.batch_size)
            data_loaders.append(data_loader)

    return data_loaders


def save_results_to_csv(results, data_name, csv_file='results.csv'):
    """
    将结果保存到CSV文件中
    
    Args:
        results: 包含评估指标和时间的字典
        data_name: 数据集名称
        csv_file: CSV文件路径
    """
    # 创建包含结果的DataFrame
    df = pd.DataFrame([results])
    df.insert(0, 'dataset', data_name)  # 在第一列插入数据集名称
    
    # 如果文件存在，则追加，否则创建新文件
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)
    
    print(f"Results saved to {csv_file}")


def train_single_entity(args, data_name: str, only_test=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ----------------------------- data ---------------------------------
    data_args = getattr(args, data_name)
    
    # 预处理数据集
    prepare_dataset(data_args, data_name)
    
    scaler = StandardScaler()
    data_loaders = get_data_loaders(data_args, args, scaler)
    
    # ------------------------ Trainer Setting ----------------------------
    model = sensitive_hue.SensitiveHUE(
        data_args.step_num_in, data_args.f_in, data_args.dim_model, args.head_num,
        data_args.dim_hidden_fc, data_args.encode_layer_num, 0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    trainer = sensitive_hue.Trainer(
        model, optimizer, data_args.alpha, args.max_epoch, data_args.model_save_dir,
        scheduler, use_prob=True)

    # 记录训练时间
    train_start_time = time.time()
    if not only_test:
        trainer.train(data_loaders[0], data_loaders[1])
    train_time = time.time() - train_start_time
    
    test_start_time = time.time()
    ignore_dims = getattr(data_args, 'ignore_dims', None)
    print(f"Testing {data_name}...")
    test_results = trainer.test(data_loaders[-1], ignore_dims, data_args.select_pos)
    test_time = time.time() - test_start_time
    
    # 提取评估指标（假设test_results包含precision, recall, f1score）
    # 根据Trainer的实现可能需要调整这部分代码
    preds, preds_adjust, labels = test_results[0], test_results[1], test_results[2]
    
    # 计算评估指标
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(labels, preds_adjust)
    recall = recall_score(labels, preds_adjust)
    f1score = f1_score(labels, preds_adjust)
    
    # 准备结果字典
    results = {
        'precision': precision,
        'recall': recall,
        'f1score': f1score,
        'train_time': train_time,
        'test_time': test_time
    }
    
    # 保存结果到CSV文件
    save_results_to_csv(results, data_name, 'results.csv')
    

    
    return results


def train_multi_entity(args, data_name: str, only_test=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ----------------------------- data ---------------------------------
    data_args = getattr(args, data_name)
    scaler = StandardScaler()
    start, end = data_args.range
    data_dir = data_args.data_dir

    ignore_dims = getattr(data_args, 'ignore_dims', dict())
    ignore_entities = getattr(data_args, 'ignore_entities', tuple())

    all_results = []
    for i in range(start, end + 1):
        if i in ignore_entities:
            continue
        data_args.data_dir = os.path.join(data_dir, f'{data_name}_{i}')
        
        # 预处理数据集
        prepare_dataset(data_args, f'{data_name}_{i}')
        
        data_loaders = get_data_loaders(data_args, args, scaler)

        model = sensitive_hue.SensitiveHUE(
            data_args.step_num_in, data_args.f_in, data_args.dim_model, args.head_num,
            data_args.dim_hidden_fc, data_args.encode_layer_num, 0.1
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
        trainer = sensitive_hue.Trainer(
            model, optimizer, data_args.alpha, args.max_epoch, data_args.model_save_dir,
            scheduler, use_prob=True, model_save_suffix=f'_{i}'
        )

        trainer.logger.info(f'entity {i}')

        # 记录训练时间
        train_start_time = time.time()
        if not only_test:
            trainer.train(data_loaders[0], data_loaders[1])
        train_time = time.time() - train_start_time
        
        # 记录测试时间和获取测试结果
        test_start_time = time.time()
        cur_ignore_dim = ignore_dims[i] if i in ignore_dims else None
        test_results = trainer.test(data_loaders[-1], cur_ignore_dim, data_args.select_pos)
        test_time = time.time() - test_start_time
        
        # 提取评估指标
        precision, recall, f1score = test_results[0], test_results[1], test_results[2]
        
        # 准备结果字典
        results = {
            'entity': i,
            'precision': precision,
            'recall': recall,
            'f1score': f1score,
            'train_time': train_time,
            'test_time': test_time
        }
        
        all_results.append(results)
        
        # 保存每个实体的结果到CSV文件
        save_results_to_csv(results, f'{data_name}_{i}', f'{data_name}_results.csv')

    # 计算平均结果
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_f1score = np.mean([r['f1score'] for r in all_results])
    avg_train_time = np.mean([r['train_time'] for r in all_results])
    avg_test_time = np.mean([r['test_time'] for r in all_results])
    
    avg_results = {
        'entity': 'average',
        'precision': avg_precision,
        'recall': avg_recall,
        'f1score': avg_f1score,
        'train_time': avg_train_time,
        'test_time': avg_test_time
    }
    
    # 保存平均结果
    save_results_to_csv(avg_results, data_name, f'{data_name}_results.csv')
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/star.yaml')
    parser.add_argument('--data_name', type=str, default='SWaT', help='Data set to train.')
    parser.add_argument('--test', action='store_true', help='Test mode.')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    args = parser.parse_args()

    configs = YAMLParser(args.config_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_args = getattr(configs, args.data_name, None)
    if data_args is None:
        raise KeyError(f'{args.data_name} not found in configs.')
    if hasattr(data_args, 'range'):
        train_multi_entity(configs, args.data_name, args.test)
    else:
        train_single_entity(configs, args.data_name, args.test)