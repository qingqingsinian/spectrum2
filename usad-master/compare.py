# compare.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_first_file(filename):
    """
    加载第一个CSV文件的数据（第一列是异常分数）
    """
    df = pd.read_csv(filename)
    scores = df.iloc[:, 0].values  # 读取第一列作为异常分数
    return scores

def compare_anomaly_scores(file1_path, file2_path):
    """
    比较两个文件中的异常分数
    """
    # 加载第一个文件（CSV格式，第一列是异常分数）
    scores1 = load_first_file(file1_path)
    
    # 加载第二个文件 (usad_realtime_results.csv)
    df2 = pd.read_csv(file2_path)
    scores2 = df2['score'].values
    
    # 对齐数据长度
    min_length = min(len(scores1), len(scores2))
    scores1_aligned = scores1[:min_length]
    scores2_aligned = scores2[:min_length]
    
    # 计算差异
    differences = scores1_aligned - scores2_aligned
    
    # 计算统计信息
    stats = {
        'mean_diff': np.mean(differences),
        'std_diff': np.std(differences),
        'max_diff': np.max(differences),
        'min_diff': np.min(differences),
        'correlation': np.corrcoef(scores1_aligned, scores2_aligned)[0, 1]
    }
    
    # 可视化比较结果
    plt.figure(figsize=(15, 12))
    
    # 时间序列比较
    plt.subplot(2, 2, 1)
    plt.plot(scores1_aligned[:min_length], label='File 1', alpha=0.7)
    plt.plot(scores2_aligned[:min_length], label='File 2 (USAD)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores Comparison (First 500 points)')
    plt.legend()
    
    # 差异随时间变化
    plt.subplot(2, 2, 2)
    plt.plot(differences[:min_length])
    plt.xlabel('Time')
    plt.ylabel('Difference (File1 - File2)')
    plt.title('Differences Between Files (First 500 points)')
    
    # 绘制散点图
    plt.subplot(2, 2, 3)
    plt.scatter(scores1_aligned[:min_length], scores2_aligned[:min_length], alpha=0.5)
    plt.xlabel('File 1 Anomaly Score')
    plt.ylabel('File 2 Anomaly Score')
    plt.title('Scatter Plot Comparison')
    
    # 添加对角线
    min_val = min(scores1_aligned.min(), scores2_aligned.min())
    max_val = max(scores1_aligned.max(), scores2_aligned.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    # 差异直方图
    plt.subplot(2, 2, 4)
    plt.hist(differences, bins=50, alpha=0.7)
    plt.xlabel('Difference (File1 - File2)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Differences')
    
    plt.tight_layout()
    plt.show()
    
    return stats, differences


stats, differences = compare_anomaly_scores('msl_test_scores.csv', 'usad_realtime_results.csv')
print("统计信息:")
for key, value in stats.items():
    print(f"{key}: {value}")