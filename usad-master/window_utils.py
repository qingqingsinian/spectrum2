# window_utils.py
import os
import pandas as pd
import numpy as np

def load_dataset(dataset_name, input_dir, window_size=None):
    """加载不同数据集并可选择性地进行窗口化标签处理"""
    if dataset_name == 'SWAT':
        # 原有的 SWAT 数据加载逻辑
        attack_path = os.path.join(input_dir, "swat2.csv")
        
        if not os.path.exists(attack_path):
            raise FileNotFoundError(f"Attack dataset not found: {attack_path}")
        
        # 加载攻击数据
        attack = pd.read_csv(attack_path, sep=",")
        labels = [float(label != 0) for label in attack["Normal/Attack"].values]
        attack = attack.drop(["Normal/Attack"], axis=1)
        
        # 转换所有列为float
        for i in list(attack):
            attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
        attack = attack.astype(float)
        
        # 如果指定了窗口大小，则进行窗口化处理
        if window_size is not None:
            labels = window_based_label_processing(labels, window_size)
        
        return attack, labels
    
    elif dataset_name == 'MSL':
        # MSL 数据集加载逻辑
        test_data = np.load(os.path.join(input_dir, "MSL_test.npy"))
        labels = np.load(os.path.join(input_dir, "MSL_test_label.npy"))
        
        # 转换为 DataFrame 格式以保持一致性
        attack = pd.DataFrame(test_data)
        
        # 如果指定了窗口大小，则进行窗口化处理
        if window_size is not None:
            labels = window_based_label_processing(labels.tolist(), window_size)
        
        return attack, labels
    
    elif dataset_name == 'PSM':
        # PSM 数据集加载逻辑
        attack_path = os.path.join(input_dir, "test.csv")
        label_path = os.path.join(input_dir, "test_label.csv")
        
        if not os.path.exists(attack_path):
            raise FileNotFoundError(f"Attack dataset not found: {attack_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label dataset not found: {label_path}")
        
        # 加载攻击数据
        attack = pd.read_csv(attack_path)
        # 移除第一列（时间戳）
        attack = attack.drop(attack.columns[0], axis=1)
        attack = attack.astype(float)
        
        # 加载标签数据
        label_data = pd.read_csv(label_path)
        # 移除第一列（时间戳）
        label_data = label_data.drop(label_data.columns[0], axis=1)
        
        # 处理标签数据 - 确保是二进制标签
        labels = []
        label_values = label_data.values.flatten()
        for label in label_values:
            labels.append(float(label != 0))  # 转换为二进制标签
            
        # 如果指定了窗口大小，则进行窗口化处理
        if window_size is not None:
            labels = window_based_label_processing(labels, window_size)
            
        return attack, labels
    
    elif dataset_name == 'SMD':
        # SMD 数据集加载逻辑
        test_data = np.load(os.path.join(input_dir, "SMD_test.npy"))
        labels = np.load(os.path.join(input_dir, "SMD_test_label.npy"))
        
        # 转换为 DataFrame 格式
        attack = pd.DataFrame(test_data)
        
        # 如果指定了窗口大小，则进行窗口化处理
        if window_size is not None:
            labels = window_based_label_processing(labels.tolist(), window_size)
        
        return attack, labels
        
    elif dataset_name == 'SMAP':
        # SMAP 数据集加载逻辑
        test_data = np.load(os.path.join(input_dir, "SMAP_test.npy"))
        labels = np.load(os.path.join(input_dir, "SMAP_test_label.npy"))
        
        # 转换为 DataFrame 格式
        attack = pd.DataFrame(test_data)
        
        # 如果指定了窗口大小，则进行窗口化处理
        if window_size is not None:
            labels = window_based_label_processing(labels.tolist(), window_size)
        
        return attack, labels
        
    elif dataset_name == 'kpi':
        # KPI 数据集加载逻辑
        attack_path = os.path.join(input_dir, "kpi_test.csv")
        
        if not os.path.exists(attack_path):
            raise FileNotFoundError(f"Attack dataset not found: {attack_path}")
        
        # 加载攻击数据 (前三列是时间戳、值、标签)
        attack = pd.read_csv(attack_path)
        # 提取标签 (第三列)
        labels = attack.iloc[:, 2].tolist()  # 第三列是标签
        # 移除时间戳和标签列，只保留值列
        attack = attack.iloc[:, 1:2]  # 只保留第二列(值)
        attack.columns = [0]  # 重命名列以匹配其他数据集格式
        
        # 如果指定了窗口大小，则进行窗口化处理
        if window_size is not None:
            labels = window_based_label_processing(labels, window_size)
        
        return attack, labels
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def preprocess_labels(labels):
    """
    预处理标签，确保它们是数值型的 0 和 1
    """
    processed_labels = []
    for label in labels:
        if label is None:
            processed_labels.append(0)  # 将 None 转换为 0
        elif isinstance(label, str):
            # 处理字符串类型的标签
            if label.lower() in ['1', 'true', 'yes', 'positive']:
                processed_labels.append(1)
            elif label.lower() in ['0', 'false', 'no', 'negative']:
                processed_labels.append(0)
            else:
                try:
                    # 尝试转换为数值
                    processed_labels.append(int(float(label)))
                except:
                    processed_labels.append(0)  # 默认为 0
        else:
            # 处理数值类型的标签
            try:
                processed_labels.append(int(float(label)))
            except:
                processed_labels.append(0)  # 默认为 0
                
    return processed_labels

def window_based_label_processing(labels: list, window_size: int) -> list:
    """
    基于窗口的标签处理函数
    只要窗口内有一个数据点是异常的，则将整个窗口标记为异常
    
    Args:
        labels: 原始标签列表
        window_size: 窗口大小
    
    Returns:
        处理后的标签列表
    """
    processed_labels = preprocess_labels(labels)
    
    for i in range(0, len(processed_labels), window_size):
        window_end = min(i + window_size, len(processed_labels))
        window_labels = processed_labels[i:window_end]
        if any(label == 1 for label in window_labels): 
            processed_labels[i:window_end] = [1] * (window_end - i)
    
    
    return processed_labels