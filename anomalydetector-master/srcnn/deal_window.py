# preprocess_data.py
import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple
import os

def window_based_preprocessing(data: pd.DataFrame, 
                              window_size: int, 
                              threshold: int) -> pd.DataFrame:
    """
    基于时间窗口的数据预处理函数
    
    Args:
        data: 包含'timestamp', 'value', 'label'列的DataFrame
        window_size: 时间窗口大小
        threshold: 异常点阈值，超过该数量则整个窗口标记为异常
    
    Returns:
        处理后的DataFrame
    """
    # 确保数据按时间戳排序
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # 创建结果DataFrame的副本
    processed_data = data.copy()
    
    # 按窗口大小分组处理数据
    for i in range(0, len(data), window_size):
        # 获取当前窗口的数据
        window_end = min(i + window_size, len(data))
        window_data = data.iloc[i:window_end]
        
        # 计算当前窗口中异常点的数量
        anomaly_count = window_data['label'].sum()
        
        # 如果异常点数量大于阈值，则将整个窗口标记为异常
        if anomaly_count > threshold:
            processed_data.iloc[i:window_end, processed_data.columns.get_loc('label')] = 1
        else:
            # 可选：也可以将整个窗口标记为正常（根据需求而定）
            processed_data.iloc[i:window_end, processed_data.columns.get_loc('label')] = 0
            pass  # 保持原始标签
    
    return processed_data

def process_multiple_files(input_dir: str, 
                          output_dir: str, 
                          window_size: int, 
                          threshold: int):
    """
    处理目录中的多个CSV文件
    
    Args:
        input_dir: 输入文件目录
        output_dir: 输出文件目录
        window_size: 时间窗口大小
        threshold: 异常点阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理目录中的所有CSV文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            
            # 读取数据
            try:
                data = pd.read_csv(input_path)
                
                # 检查必要列是否存在
                required_columns = ['timestamp', 'value', 'label']
                if not all(col in data.columns for col in required_columns):
                    print(f"警告: 文件 {filename} 缺少必要列，跳过处理")
                    continue
                
                # 处理数据
                processed_data = window_based_preprocessing(data, window_size, threshold)
                
                # 保存处理后的数据
                processed_data.to_csv(output_path, index=False)
                print(f"已处理文件: {filename} -> {output_path}")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='基于时间窗口的数据预处理')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径或目录')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径或目录')
    parser.add_argument('--window_size', type=int, default=5, help='时间窗口大小')
    parser.add_argument('--threshold', type=int, default=1, help='异常点阈值')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], default='single', 
                        help='处理模式: single(单文件) 或 multiple(多文件目录)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # 处理单个文件
        data = pd.read_csv(args.input)
        
        # 检查必要列
        required_columns = ['timestamp', 'value', 'label']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"CSV文件必须包含以下列: {required_columns}")
        
        # 预处理数据
        processed_data = window_based_preprocessing(data, args.window_size, args.threshold)
        
        # 保存结果
        processed_data.to_csv(args.output, index=False)
        print(f"预处理完成，结果已保存到: {args.output}")
        
    else:
        # 处理多个文件
        process_multiple_files(args.input, args.output, args.window_size, args.threshold)
        print("多文件预处理完成")

if __name__ == '__main__':
    main()