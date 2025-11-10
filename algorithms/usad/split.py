import pandas as pd
import os

def split_kpi_datasets_by_id(input_dir="input/kpi", output_dir="input/kpi_split"):
    """
    按照 KPI ID 将 KPI 数据集分割成多个独立的 CSV 文件
    
    参数:
    input_dir: 包含原始 KPI 数据集的目录
    output_dir: 保存分割后文件的目录
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练数据集
    train_path = os.path.join(input_dir, "kpi_train.csv")
    if os.path.exists(train_path):
        print(f"处理训练数据集: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"训练数据集列名: {list(train_df.columns)}")
        print(f"训练数据集形状: {train_df.shape}")
        
        # 根据您提供的信息，KPI ID 在第4列 (索引为3)
        # 列顺序为: timestamp, value, label, KPI ID, source_file
        if len(train_df.columns) >= 4:
            kpi_id_col = train_df.columns[2]  # KPI ID 列
            print(f"使用 '{kpi_id_col}' 列作为 KPI ID")
            
            # 按 KPI ID 分组并保存
            for kpi_id, group in train_df.groupby(kpi_id_col):
                # 创建文件名（移除可能存在的非法字符）
                safe_kpi_id = str(kpi_id).replace('/', '_').replace('\\', '_').replace(':', '_').replace('.csv', '_csv')
                output_file = os.path.join(output_dir, f"{safe_kpi_id}_train.csv")
                
                # 去除 source_file 列（如果存在）
                if 'source_file' in group.columns:
                    group = group.drop(columns=['source_file'])
                
                # 保存数据
                group.to_csv(output_file, index=False)
                print(f"已保存 {output_file}，包含 {len(group)} 行数据")
        else:
            print("错误: 训练数据集列数不足")
            return
    
    # 处理测试数据集
    test_path = os.path.join(input_dir, "kpi_test.csv")
    if os.path.exists(test_path):
        print(f"\n处理测试数据集: {test_path}")
        test_df = pd.read_csv(test_path)
        print(f"测试数据集列名: {list(test_df.columns)}")
        print(f"测试数据集形状: {test_df.shape}")
        
        # 根据您提供的信息，KPI ID 在第4列 (索引为3)
        # 列顺序为: timestamp, value, source_file, KPI ID
        if len(test_df.columns) >= 4:
            kpi_id_col = test_df.columns[3]  # KPI ID 列（根据您提供的数据格式，应该是第4列）
            print(f"使用 '{kpi_id_col}' 列作为 KPI ID")
            
            # 按 KPI ID 分组并保存
            for kpi_id, group in test_df.groupby(kpi_id_col):
                # 创建文件名（移除可能存在的非法字符）
                safe_kpi_id = str(kpi_id).replace('/', '_').replace('\\', '_').replace(':', '_').replace('.csv', '_csv')
                output_file = os.path.join(output_dir, f"{safe_kpi_id}_test.csv")
                
                # 去除 source_file 列（如果存在）
                if 'source_file' in group.columns:
                    group = group.drop(columns=['source_file'])
                
                # 保存数据
                group.to_csv(output_file, index=False)
                print(f"已保存 {output_file}，包含 {len(group)} 行数据")
        else:
            print("错误: 测试数据集列数不足")
            return
            
    print(f"\n数据分割完成！分割后的文件保存在: {output_dir}")

def verify_split_files(split_dir="input/kpi_split"):
    """
    验证分割后的文件
    """
    if not os.path.exists(split_dir):
        print(f"目录 {split_dir} 不存在")
        return
        
    files = os.listdir(split_dir)
    print(f"\n分割后的文件列表:")
    for file in sorted(files):
        if file.endswith('.csv'):
            file_path = os.path.join(split_dir, file)
            try:
                df = pd.read_csv(file_path)
                print(f"  {file}: {df.shape[0]} 行, {df.shape[1]} 列")
            except Exception as e:
                print(f"  {file}: 读取错误 - {e}")

# 使用示例
if __name__ == "__main__":
    print("开始按 KPI ID 分割数据集...")
    split_kpi_datasets_by_id()
    
    # 验证分割结果
    verify_split_files()
    
    print("\n任务完成!")