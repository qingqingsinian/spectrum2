import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
    

def process_smd_dataset(data_path, output_path):
    """
    处理SMD数据集 (进行标准化)
    文件: SMD_train.npy, SMD_test.npy, SMD_test_label.npy
    输出: smd_standardized.npz (合并训练集和测试集)
    """
    # 加载数据
    train_data = np.load(os.path.join(data_path, "SMD_train.npy"))
    test_data = np.load(os.path.join(data_path, "SMD_test.npy"))
    test_labels = np.load(os.path.join(data_path, "SMD_test_label.npy"))
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)
    
    # 对所有数据进行标准化 (使用训练数据拟合标准化参数)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # SMD数据集标签处理：0表示正常，1表示异常
    # 我们需要翻转标签以匹配DAGMM的约定：1表示正常，0表示异常
    test_labels_flipped = 1 - test_labels
    
    # 为训练数据添加标签(训练数据都是正常的，标签为1)
    train_labels = np.ones((train_data_scaled.shape[0], 1))
    
    # 为测试数据添加标签
    test_labels_flipped = test_labels_flipped.reshape(-1, 1)
    
    # 合并训练和测试数据
    all_data = np.concatenate([train_data_scaled, test_data_scaled], axis=0)
    all_labels = np.concatenate([train_labels, test_labels_flipped], axis=0)
    
    # 合并特征和标签
    final_data = np.concatenate([all_data, all_labels], axis=1)
    
    # 保存合并后的数据
    output_file = os.path.join(output_path, "smd_standardized.npz")
    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(output_file, kdd=final_data)
    print(f"SMD 标准化数据已保存到 {output_file}，形状: {final_data.shape}")
# 在 process_data.py 中添加以下函数

def process_kpi_dataset(data_path, output_path):
    """
    处理KPI数据集 (进行标准化)
    文件: kpi_train.csv, kpi_test.csv
    输出: kpi_standardized.npz (合并训练集和测试集)
    """
    # 加载数据
    train_df = pd.read_csv(os.path.join(data_path, "kpi_train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "kpi_test.csv"))
    
    # 提取特征 (时间戳列通常不需要作为特征)
    # kpi_train: 第一列是时间戳，第二列是值
    train_data = train_df.values[:, 1:2]  # 只取值列，并保持二维数组
    # kpi_test: 第一列是时间戳，第二列是值，第三列是标签
    test_data = test_df.values[:, 1:2]    # 只取值列
    
    # 提取测试集标签 (第三列)
    test_labels = test_df.values[:, 2].reshape(-1, 1)
    print(train_data,test_data,test_labels)
    # 处理NaN值
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)
    
    # 对所有数据进行标准化 (使用训练数据拟合标准化参数)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 翻转标签以匹配DAGMM的约定 (KPI中1表示异常，0表示正常)
    test_labels_flipped = 1 - test_labels
    
    # 为训练数据添加标签(训练数据都是正常的，标签为1)
    train_labels = np.ones((train_data_scaled.shape[0], 1))
    
    # 合并训练和测试数据
    all_data = np.concatenate([train_data_scaled, test_data_scaled], axis=0)
    all_labels = np.concatenate([train_labels, test_labels_flipped], axis=0)
    
    # 合并特征和标签
    final_data = np.concatenate([all_data, all_labels], axis=1)
    
    # 保存合并后的数据
    output_file = os.path.join(output_path, "kpi_standardized.npz")
    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(output_file, kdd=final_data)
    print(f"KPI 标准化数据已保存到 {output_file}，形状: {final_data.shape}")

def process_smap_dataset(data_path, output_path):
    """
    处理SMAP数据集 (进行标准化)
    文件: SMAP_train.npy, SMAP_test.npy, SMAP_test_label.npy
    输出: smap_standardized.npz (合并训练集和测试集)
    """
    # 加载数据
    train_data = np.load(os.path.join(data_path, "SMAP_train.npy"))
    test_data = np.load(os.path.join(data_path, "SMAP_test.npy"))
    test_labels = np.load(os.path.join(data_path, "SMAP_test_label.npy"))
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)
    
    # 对所有数据进行标准化 (使用训练数据拟合标准化参数)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 翻转标签以匹配DAGMM的约定
    test_labels_flipped = 1 - test_labels
    
    # 为训练数据添加标签(训练数据都是正常的，标签为1)
    train_labels = np.ones((train_data_scaled.shape[0], 1))
    
    # 为测试数据添加标签
    test_labels_flipped = test_labels_flipped.reshape(-1, 1)
    
    # 合并训练和测试数据
    all_data = np.concatenate([train_data_scaled, test_data_scaled], axis=0)
    all_labels = np.concatenate([train_labels, test_labels_flipped], axis=0)
    
    # 合并特征和标签
    final_data = np.concatenate([all_data, all_labels], axis=1)
    
    # 保存合并后的数据
    output_file = os.path.join(output_path, "smap_standardized.npz")
    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(output_file, kdd=final_data)
    print(f"SMAP 标准化数据已保存到 {output_file}，形状: {final_data.shape}")

def process_psm_dataset(data_path, output_path):
    """
    处理PSM数据集 (进行标准化)
    文件: train.csv, test.csv, test_label.csv
    输出: psm_standardized.npz (合并训练集和测试集)
    """
    # 加载数据
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_label_df = pd.read_csv(os.path.join(data_path, "test_label.csv"))
    
    # 移除第一列(通常为时间戳)
    train_data = train_df.values[:, 1:]
    test_data = test_df.values[:, 1:]
    test_labels = test_label_df.values[:, 1:]
    
    # 处理NaN值
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)
    
    # 对所有数据进行标准化 (使用训练数据拟合标准化参数)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 翻转标签以匹配DAGMM的约定
    test_labels_flipped = 1 - test_labels
    
    # 为训练数据添加标签(训练数据都是正常的，标签为1)
    train_labels = np.ones((train_data_scaled.shape[0], 1))
    
    # 为测试数据添加标签
    test_labels_flipped = test_labels_flipped.reshape(-1, 1)
    
    # 合并训练和测试数据
    all_data = np.concatenate([train_data_scaled, test_data_scaled], axis=0)
    all_labels = np.concatenate([train_labels, test_labels_flipped], axis=0)
    
    # 合并特征和标签
    final_data = np.concatenate([all_data, all_labels], axis=1)
    
    # 保存合并后的数据
    output_file = os.path.join(output_path, "psm_standardized.npz")
    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(output_file, kdd=final_data)
    print(f"PSM 标准化数据已保存到 {output_file}，形状: {final_data.shape}")

def process_msl_dataset(data_path, output_path):
    """
    处理MSL数据集 (进行标准化)
    文件: MSL_train.npy, MSL_test.npy, MSL_test_label.npy
    输出: msl_standardized.npz (合并训练集和测试集)
    """
    # 加载数据
    train_data = np.load(os.path.join(data_path, "MSL_train.npy"))
    test_data = np.load(os.path.join(data_path, "MSL_test.npy"))
    test_labels = np.load(os.path.join(data_path, "MSL_test_label.npy"))
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)
    
    # 对所有数据进行标准化 (使用训练数据拟合标准化参数)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 翻转标签以匹配DAGMM的约定
    test_labels_flipped = 1 - test_labels
    
    # 为训练数据添加标签(训练数据都是正常的，标签为1)
    train_labels = np.ones((train_data_scaled.shape[0], 1))
    
    # 为测试数据添加标签
    test_labels_flipped = test_labels_flipped.reshape(-1, 1)
    
    # 合并训练和测试数据
    all_data = np.concatenate([train_data_scaled, test_data_scaled], axis=0)
    all_labels = np.concatenate([train_labels, test_labels_flipped], axis=0)
    
    # 合并特征和标签
    final_data = np.concatenate([all_data, all_labels], axis=1)
    
    # 保存合并后的数据
    output_file = os.path.join(output_path, "msl_standardized.npz")
    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(output_file, kdd=final_data)
    print(f"MSL 标准化数据已保存到 {output_file}，形状: {final_data.shape}")

def process_swat_dataset(data_path, output_path):
    """
    处理SWAT数据集 (进行标准化)
    文件: swat_train2.csv, swat2.csv
    输出: swat_standardized.npz (合并训练集和测试集)
    """
    # 加载数据
    train_df = pd.read_csv(os.path.join(data_path, "swat_train2.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "swat2.csv"))
    
    # 分离特征和标签(标签在测试数据的最后一列)
    train_data = train_df.values[:, :-1]  # SWAT训练数据不包含标签列
    test_data = test_df.values[:, :-1]    # 测试数据特征
    test_labels = test_df.values[:, -1].reshape(-1, 1)  # 最后一列是标签
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)
    
    # 对所有数据进行标准化 (使用训练数据拟合标准化参数)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 翻转标签以匹配DAGMM的约定 (SWAT中1表示异常，0表示正常)
    test_labels_flipped = 1 - test_labels
    
    # 为训练数据添加标签(训练数据都是正常的，标签为1)
    train_labels = np.ones((train_data_scaled.shape[0], 1))
    
    # 合并训练和测试数据
    all_data = np.concatenate([train_data_scaled, test_data_scaled], axis=0)
    all_labels = np.concatenate([train_labels, test_labels_flipped], axis=0)
    
    # 合并特征和标签
    final_data = np.concatenate([all_data, all_labels], axis=1)
    
    # 保存合并后的数据
    output_file = os.path.join(output_path, "swat_standardized.npz")
    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(output_file, kdd=final_data)
    print(f"SWAT 标准化数据已保存到 {output_file}，形状: {final_data.shape}")

if __name__ == "__main__":
    # 定义数据集路径
    datasets = {
        "SMD": "./data/SMD",
        "SMAP": "./data/SMAP",
        "PSM": "./data/PSM",
        "MSL": "./data/MSL",
        "SWAT": "./data/SWAT",
        "kpi": "./data/kpi"  # 添加KPI数据集路径
    }
    
    # 处理每个数据集
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"警告: 路径 {path} 不存在，跳过 {name} 数据集")
            continue
            
        try:
            output_dir = "./processed_data"
            if name == "SMD":
                process_smd_dataset(path, output_dir)
            elif name == "SMAP":
                process_smap_dataset(path, output_dir)
            elif name == "PSM":
                process_psm_dataset(path, output_dir)
            elif name == "MSL":
                process_msl_dataset(path, output_dir)
            elif name == "SWAT":
                process_swat_dataset(path, output_dir)
            elif name == "kpi":
                process_kpi_dataset(path, output_dir)  # 添加KPI数据集处理
            
            print(f"{name} 标准化数据集处理完成\n")
            
        except Exception as e:
            print(f"处理 {name} 数据集时出错: {e}")