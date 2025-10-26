# test_usad.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import time
import csv
from usad import *
from utils import *

# 在 main.py 中添加新的数据加载函数
def load_dataset(dataset_name, input_dir):
    """加载不同数据集"""
    if dataset_name == 'SWAT':
        # 原有的 SWAT 数据加载逻辑
        normal_path = os.path.join(input_dir, "swat_train2.csv")
        attack_path = os.path.join(input_dir, "swat2.csv")
        
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"Normal dataset not found: {normal_path}")
        if not os.path.exists(attack_path):
            raise FileNotFoundError(f"Attack dataset not found: {attack_path}")
        
        # 加载正常数据
        normal = pd.read_csv(normal_path)
        normal = normal.drop(["Normal/Attack"], axis=1)
        
        # 转换所有列为float
        for i in list(normal): 
            normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
        normal = normal.astype(float)
        
        # 加载攻击数据
        attack = pd.read_csv(attack_path, sep=",")
        labels = [float(label != 0) for label in attack["Normal/Attack"].values]
        attack = attack.drop(["Normal/Attack"], axis=1)
        
        # 转换所有列为float
        for i in list(attack):
            attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
        attack = attack.astype(float)
        
        return normal, attack, labels
    
    elif dataset_name == 'MSL':
        # MSL 数据集加载逻辑
        train_data = np.load(os.path.join(input_dir, "MSL_train.npy"))
        test_data = np.load(os.path.join(input_dir, "MSL_test.npy"))
        labels = np.load(os.path.join(input_dir, "MSL_test_label.npy"))
        
        # 转换为 DataFrame 格式以保持一致性
        normal = pd.DataFrame(train_data)
        attack = pd.DataFrame(test_data)
        
        return normal, attack, labels.tolist()
    
    elif dataset_name == 'PSM':
    # PSM 数据集加载逻辑
        normal_path = os.path.join(input_dir, "train.csv")
        attack_path = os.path.join(input_dir, "test.csv")
        label_path = os.path.join(input_dir, "test_label.csv")
        
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"Normal dataset not found: {normal_path}")
        if not os.path.exists(attack_path):
            raise FileNotFoundError(f"Attack dataset not found: {attack_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label dataset not found: {label_path}")
        
        # 加载正常数据
        normal = pd.read_csv(normal_path)
        # 移除第一列（时间戳）
        
        normal = normal.drop(normal.columns[0], axis=1)
        
        # 转换所有列为float
        #for i in list(normal): 
        #    normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
        normal = normal.astype(float)
        
        # 加载攻击数据
        attack = pd.read_csv(attack_path)
        # 移除第一列（时间戳）
        
        attack = attack.drop(attack.columns[0], axis=1)
        
        # 转换所有列为float
        #for i in list(attack):
         #   attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
        attack = attack.astype(float)
        
        # 加载标签数据
        label_data = pd.read_csv(label_path)
        # 移除第一列（时间戳）
        
        label_data = label_data.drop(label_data.columns[0], axis=1)
        print(len(label_data))
        # 处理标签数据 - 确保是二进制标签
        labels = []
        label_values = label_data.values.flatten()
        print(len(label_values))
        for label in label_values:
            labels.append(float(label != 0))  # 转换为二进制标签
        print(len(normal),len(attack),len(labels))
        return normal, attack, labels
    
    elif dataset_name == 'SMD':
        # SMD 数据集加载逻辑
        train_data = np.load(os.path.join(input_dir, "SMD_train.npy"))
        test_data = np.load(os.path.join(input_dir, "SMD_test.npy"))
        labels = np.load(os.path.join(input_dir, "SMD_test_label.npy"))
        
        # 转换为 DataFrame 格式
        normal = pd.DataFrame(train_data)
        attack = pd.DataFrame(test_data)
        return normal, attack, labels.tolist()
    elif dataset_name == 'SMAP':
        # SMAP 数据集加载逻辑
        train_data = np.load(os.path.join(input_dir, "SMAP_train.npy"))
        test_data = np.load(os.path.join(input_dir, "SMAP_test.npy"))
        labels = np.load(os.path.join(input_dir, "SMAP_test_label.npy"))
        
        # 转换为 DataFrame 格式
        normal = pd.DataFrame(train_data)
        attack = pd.DataFrame(test_data)
        
      
        return normal, attack, labels.tolist()
    elif dataset_name == 'kpi':
        # KPI 数据集加载逻辑
        normal_path = os.path.join(input_dir, "kpi_train.csv")
        attack_path = os.path.join(input_dir, "kpi_test.csv")
        
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"Normal dataset not found: {normal_path}")
        if not os.path.exists(attack_path):
            raise FileNotFoundError(f"Attack dataset not found: {attack_path}")
        
        # 加载正常数据 (前两列是时间戳和值)
        normal = pd.read_csv(normal_path)
        # 移除时间戳列，只保留值列
        normal = normal.iloc[:, 1:2]  # 只保留第二列(值)
        normal.columns = [0]  # 重命名列以匹配其他数据集格式
        
        # 加载攻击数据 (前三列是时间戳、值、标签)
        attack = pd.read_csv(attack_path)
        # 提取标签 (第三列)
        labels = attack.iloc[:, 2].tolist()  # 第三列是标签
        # 移除时间戳和标签列，只保留值列
        attack = attack.iloc[:, 1:2]  # 只保留第二列(值)
        attack.columns = [0]  # 重命名列以匹配其他数据集格式
        
        return normal, attack, labels
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def normalize_data(normal, attack):
    """按照notebook中的方法进行标准化"""
    # 对正常数据进行MinMax标准化
    min_max_scaler = preprocessing.MinMaxScaler()
    
    x_normal = normal.values
    x_normal_scaled = min_max_scaler.fit_transform(x_normal)
    normal = pd.DataFrame(x_normal_scaled)
    
    # 对攻击数据使用相同的scaler进行标准化
    x_attack = attack.values 
    x_attack_scaled = min_max_scaler.transform(x_attack)
    attack = pd.DataFrame(x_attack_scaled)
    
    return normal, attack, min_max_scaler

class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, window_size, transform=None):
        self.data = torch.from_numpy(dataframe.values).float()
        self.window_size = window_size
        self.transform = transform
        self.length = len(self.data) - window_size + 1
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError()
            
        # 实时创建窗口
        window = self.data[idx:idx + self.window_size]
        # 展平窗口
        window_flat = window.reshape(-1)
        
        if self.transform:
            window_flat = self.transform(window_flat)
            
        return window_flat

# 使用示例
def create_data_loaders(normal_df, attack_df, labels, window_size):
    # 创建训练和验证数据集
    total_normal_length = len(normal_df)
    train_end_idx = int(0.8 * (total_normal_length - window_size))
    
    train_dataset = SlidingWindowDataset(normal_df.iloc[:train_end_idx + window_size], window_size)
    val_dataset = SlidingWindowDataset(normal_df.iloc[train_end_idx:], window_size)
    test_dataset = SlidingWindowDataset(attack_df, window_size)
    
    # 创建标签
    y_test = []
    for i in range(len(labels) - window_size+1):
        window_labels = labels[i:i + window_size]
        y_test.append(1.0 if np.sum(window_labels) > 0 else 0.0)
    
    # 数据加载器
    BATCH_SIZE = 2048
    w_size = window_size * normal_df.shape[1]
    z_size = window_size * 100
    print(window_size,normal_df.shape[1],z_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_test, w_size, z_size
def evaluate_metrics(y_true, y_pred, threshold=None):
    """计算评估指标"""
    if threshold is not None:
        y_pred_binary = (y_pred > threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    return precision, recall, f1

def main(dataset_name='SWAT'):
    # 设置设备
    device = get_default_device()
    print(f"Using device: {device}")
    
    # 创建数据集特定的输入目录
    input_dir = f"input/{dataset_name.lower()}"
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")
    
    # 加载和预处理数据
    normal, attack, labels = load_dataset(dataset_name, input_dir)
    normal, attack, scaler = normalize_data(normal, attack)
    import pickle
    with open(f'{dataset_name.lower()}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    # 创建数据加载器
    window_size = 128
    train_loader, val_loader, test_loader, y_test, w_size, z_size = create_data_loaders(
        normal, attack, labels, window_size)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # 初始化模型
    dropout_rate = 0.1  # 设置dropout率
    use_batch_norm = True  # 是否使用批归一化
    model = UsadModel(w_size, z_size, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)
    model = to_device(model, device)
    
    # 训练模型并统计时间
    print("Starting training...")
    weight_decay = 1e-5  # L2正则化系数
    train_start_time = time.time()
    history = training(10, model, train_loader, val_loader, learning_rate=0.0004, weight_decay=weight_decay)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # 保存模型
    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder1': model.decoder1.state_dict(),
        'decoder2': model.decoder2.state_dict()
    }, f"{dataset_name.lower()}_model.pth")
    
    # 加载模型
    print("Loading trained model...")
    checkpoint = torch.load(f"{dataset_name.lower()}_model.pth")
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])
    print(model.decoder1)
    # 测试模型并统计时间
    print("Starting testing...")
    test_start_time = time.time()
    results = testing(model, test_loader)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f"Testing completed in {test_time:.2f} seconds")
    
    # 处理测试结果
    y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                             results[-1].flatten().detach().cpu().numpy()])
    scores_df = pd.DataFrame({
        'anomaly_score': y_pred,
        'true_label': y_test
    })
    scores_df.to_csv(f'{dataset_name.lower()}_test_scores.csv', index=False)
    print(f"Anomaly scores saved to {dataset_name.lower()}_test_scores.csv")
    #print(y_pred.shape,y_test.shape)
    
    # 绘制ROC曲线并获取最优阈值
    print(len(y_test))
    print("Evaluating model performance...")
    threshold = ROC(y_test, y_pred)
    print(f"Optimal threshold: {threshold}")
    
    # 使用最优阈值计算评估指标
    precision, recall, f1 = evaluate_metrics(y_test, y_pred, threshold)
    print(f"\nEvaluation Metrics (Optimal Threshold = {threshold[0]:.6f}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("Testing completed.")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold[0] if isinstance(threshold, np.ndarray) else threshold,
        'train_time': 0,
        'test_time': test_time
    }
# 修改主程序入口
if __name__ == "__main__":
    # 支持评测多个数据集
    datasets = [ 'SMAP',  'PSM','SMD','SWAT']
    results = {}
    
    # 准备CSV文件
    csv_filename = "experiment_results.csv"
    csv_headers = ['Dataset', 'Precision', 'Recall', 'F1-Score', 'Threshold', 'Train_Time(seconds)', 'Test_Time(seconds)']
    
    # 创建或清空CSV文件并写入头部

    
    for dataset in datasets:
       
            print(f"\n{'='*50}")
            print(f"Processing dataset: {dataset}")
            print(f"{'='*50}")
            result = main(dataset)
            results[dataset] = result
            
            # 将结果写入CSV文件
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    dataset,
                    f"{result['precision']:.4f}",
                    f"{result['recall']:.4f}",
                    f"{result['f1']:.4f}",
                    f"{result['threshold']:.6f}",
                    f"{result['train_time']:.2f}",
                    f"{result['test_time']:.2f}"
                ])

    
    # 打印所有数据集的结果汇总
    print(f"\n{'='*50}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*50}")
    for dataset, result in results.items():
        if result:
            print(f"{dataset}:")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  F1-Score:  {result['f1']:.4f}")
            print(f"  Threshold: {result['threshold']:.6f}")
            print(f"  Train Time: {result['train_time']:.2f} seconds")
            print(f"  Test Time:  {result['test_time']:.2f} seconds")
        else:
            print(f"{dataset}: Failed to process")
    
    print(f"\nResults have been saved to {csv_filename}")