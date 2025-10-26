# realtime_detection.py

import os
import time
import torch
import pandas as pd
import numpy as np
from source import IntroVAE
from source import KPISeries
from evaluation_metric import range_lift_with_delay

class RealTimeAnomalyDetector:
    """
    实时异常检测器，累积数据形成批次后进行异常检测
    """
    
    def __init__(self, model_path, window_size=128, latency=1.0, batch_size=256):
        """
        初始化实时检测器
        
        Args:
            model_path (str): 训练好的模型路径
            window_size (int): 滑动窗口大小
            latency (float): 数据采集间隔（秒）
            batch_size (int): 批次大小
        """
        # 设置随机种子以确保结果可重现，与main.py中保持一致
        seed = 2021
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
        self.window_size = window_size
        self.latency = latency
        self.batch_size = batch_size
        self.data_buffer = []  # 存储窗口数据
        self.timestamp_buffer = []
        self.label_buffer = []
        self.truth_buffer = []
        self.missing_buffer = []
        self.train_mean = None
        self.train_std = None
        self.delay = 7
        
        # 新增：用于累积批次数据
        self.batch_windows = []  # 存储完整的窗口
        self.batch_timestamps = []  # 存储对应的时间戳
        self.batch_labels = []
        self.batch_truths = []
        self.batch_missings = []
        
        # 加载模型
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """
        从指定路径加载模型
        
        Args:
            model_path (str): 模型文件路径
            
        Returns:
            IntroVAE: 加载的模型实例
        """
        
        # 确保与SaVAE初始化参数一致
        model = IntroVAE(
            cuda=torch.cuda.is_available(),
            window_size=self.window_size,
            latent_dims=3,  # 与main.py中保持一致
            margin=15.0,    # 与main.py中保持一致
            batch_size=self.batch_size
        )
        stats = np.load('normalization_stats.npz')
        self.train_mean = stats['mean']
        self.train_std = stats['std']
        model.load(model_path)
        model.model.eval()
        
        return model
    
    def add_data_point(self, value, timestamp, label=0, truth=0, missing=0):
        """
        添加单个数据点到缓冲区
        
        Args:
            value (float): 数据值
            timestamp (int): 时间戳
            label (int): 预测标签（默认0）
            truth (int): 真实标签（默认0）
            missing (int): 是否为缺失值（默认0）
            
        Returns:
            list or None: 如果有累积的检测结果则返回，否则返回None
        """
        self.data_buffer.append(value)
        self.timestamp_buffer.append(timestamp)
        self.label_buffer.append(label)
        self.truth_buffer.append(truth)
        self.missing_buffer.append(missing)
        
        # 如果缓冲区满了，则形成一个完整的窗口
        if len(self.data_buffer) >= self.window_size:
            # 保存完整的窗口数据
            self.batch_windows.append(self.data_buffer.copy())
            self.batch_timestamps.append(self.timestamp_buffer.copy())
            self.batch_labels.append(self.label_buffer.copy())
            self.batch_truths.append(self.truth_buffer.copy())
            self.batch_missings.append(self.missing_buffer.copy())
            
            # 清空缓冲区，为下一个窗口做准备
            self.data_buffer.pop(0)
            self.timestamp_buffer.pop(0)
            self.label_buffer.pop(0)
            self.truth_buffer.pop(0)
            self.missing_buffer.pop(0)
            
            # 检查是否累积了足够的窗口形成一个批次
            if len(self.batch_windows) >= self.batch_size:
                # 执行批次检测
                results = self._detect_batch()
                
                # 清空批次缓冲区
                self.batch_windows = []
                self.batch_timestamps = []
                self.batch_labels = []
                self.batch_truths = []
                self.batch_missings = []
                
                return results
        
        return None
    
    def _detect_batch(self):
        """
        对累积的批次数据进行异常检测
        
        Returns:
            list: 检测结果列表
        """
        try:
            # 创建KPI序列集合
            all_values = np.array(self.batch_windows)
            all_timestamps = np.array(self.batch_timestamps)
            all_labels = np.array(self.batch_labels)
            all_truths = np.array(self.batch_truths)
            all_missings = np.array(self.batch_missings)
            
            # 创建KPI序列对象
            kpi_values = []
            kpi_timestamps = []
            kpi_labels = []
            kpi_truths = []
            kpi_missings = []
            
            for i in range(len(self.batch_windows)):
                kpi_series = KPISeries(
                    value=all_values[i],
                    timestamp=all_timestamps[i],
                    label=all_labels[i],
                    truth=all_truths[i],
                    missing=all_missings[i]
                )
                kpi_values.extend(all_values[i])
                kpi_timestamps.extend(all_timestamps[i])
                kpi_labels.extend(all_labels[i])
                kpi_truths.extend(all_truths[i])
                kpi_missings.extend(all_missings[i])
            
            # 创建批次KPI序列
            batch_kpi = KPISeries(
                value=np.array(kpi_values),
                timestamp=np.array(kpi_timestamps),
                label=np.array(kpi_labels),
                truth=np.array(kpi_truths),
                missing=np.array(kpi_missings)
            )
            
            # 归一化数据
            normalized_kpi = KPISeries(
                value=(batch_kpi.value - self.train_mean) / np.clip(self.train_std, 1e-4, None),
                timestamp=batch_kpi.timestamp,
                label=batch_kpi.label,
                truth=batch_kpi.truth,
                missing=batch_kpi.missing
            )

            # 使用模型进行批次预测
            anomaly_scores = self.model.predict(normalized_kpi, indicator_name="indicator_erf")

           
            # 根据SaVAE的实现，前window_size-1个数据是填充的，需要去掉
            if len(anomaly_scores) >= self.window_size - 1:
                # 去掉前面window_size-1个填充的数据点
                valid_anomaly_scores = anomaly_scores[self.window_size - 1:]

            else:
                valid_anomaly_scores = anomaly_scores
            # 处理结果，为每个窗口的最后一个点生成结果
            # 处理结果，为每个窗口的最后一个点生成结果
            results = []
            num_windows_to_process = min(len(self.batch_windows), len(valid_anomaly_scores))

            # 新增：窗口大小（根据您的要求设置为8）
            window_size = 8

            # 按窗口大小分组处理
            for i in range(0, num_windows_to_process, window_size):
                # 获取当前窗口组
                window_group_end = min(i + window_size, num_windows_to_process)
                
                # 检查这个窗口组中是否有任何一个点超过阈值
                group_has_anomaly = False
                group_scores = []
                
                for j in range(i, window_group_end):
                    score_idx = j
                    if score_idx < len(valid_anomaly_scores):
                        score = valid_anomaly_scores[score_idx]
                        group_scores.append(score)
                        if score > 0.5967797636985779:  # 使用与原来相同的阈值
                            group_has_anomaly = True
                
                # 如果组中有任何一个点是异常，则整个组都标记为异常
                for j in range(i, window_group_end):
                    score_idx = j
                    if score_idx < len(valid_anomaly_scores):
                        score = valid_anomaly_scores[score_idx]
                        is_anomaly = group_has_anomaly  # 整个组使用相同的异常标记
                        
                        result = {
                            'timestamp': self.batch_timestamps[j][-1],  # 最后一个点的时间戳
                            'value': self.batch_windows[j][-1],         # 最后一个点的值
                            'anomaly_score': float(score),
                            'is_anomaly': bool(is_anomaly),
                            'truth': self.batch_truths[j][-1]
                        }
                        results.append(result)
                        
            return results
            
        except Exception as e:
            print(f"Error during batch anomaly detection: {e}")
            return []
    # 将fill_missing_timestamps作为类方法
    def fill_missing_timestamps(self, df, interval=60):
        """
        遍历df中的时间戳，如果两个时间戳的差不是60，则补全中间数据，并标记为异常
        
        Args:
            df: 包含时间戳和数据的DataFrame
            interval: 期望的时间间隔（默认60秒）
        
        Returns:
            补全后的DataFrame
        """
        # 确保时间戳列存在
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame中必须包含'timestamp'列")
        
        # 按时间戳排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 创建新的DataFrame来存储结果
        filled_data = []
        
        # 遍历相邻的时间戳对
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            current_timestamp = current_row['timestamp']
            next_timestamp = next_row['timestamp']
            
            # 添加当前行数据
            current_dict = current_row.to_dict()
            if 'missing' not in current_dict:
                current_dict['missing'] = 0  # 原始数据点不是缺失点
            filled_data.append(current_dict)
            
            # 计算时间差
            time_diff = next_timestamp - current_timestamp
            
            # 如果时间差不是interval的整数倍，则需要补全
            if time_diff != interval:
                # 计算需要补全的点数
                missing_points = int(time_diff // interval) - 1
                
                # 补全缺失的时间点
                for j in range(1, missing_points + 1):
                    missing_timestamp = current_timestamp + j * interval
                    
                    # 创建补全的数据点
                    missing_row = current_row.copy()
                    missing_row['timestamp'] = missing_timestamp
                    
                    # 将数值型数据设置为0（与KPISeries中处理缺失值的方式保持一致）
                    missing_row['value'] = 0.0  # 将缺失点的值设为0
                    
                    # 标记为缺失数据点
                    missing_row['missing'] = 1  # 1表示这是补全的缺失点
                    if 'missing_flag' in missing_row:
                        missing_row['missing_flag'] = 1
                    
                    # 对于标签列，设置为0表示缺失数据
                    if 'label' in missing_row:
                        missing_row['label'] = 0
                    
                    if 'truth' in missing_row:
                        missing_row['truth'] = 0
                    
                    filled_data.append(missing_row)
        
        # 添加最后一行数据
        last_row = df.iloc[-1].to_dict()
        if 'missing' not in last_row:
            last_row['missing'] = 0  # 最后一个原始数据点不是缺失点
        filled_data.append(last_row)
        
        # 转换为DataFrame
        filled_df = pd.DataFrame(filled_data)
        
        return filled_df.reset_index(drop=True)

# 更新simulate_realtime_detection函数，使用类中的fill_missing_timestamps方法
def simulate_realtime_detection(csv_file_path, model_path, window_size=128, latency=1.0, batch_size=220):
    """
    模拟实时数据采集和异常检测过程
    
    Args:
        csv_file_path (str): 包含测试数据的CSV文件路径
        model_path (str): 训练好的模型路径
        window_size (int): 滑动窗口大小
        latency (float): 数据采集间隔（秒）
        batch_size (int): 批次大小
    """
    # 读取测试数据

    # 初始化实时检测器
    detector = RealTimeAnomalyDetector(model_path, window_size, latency, batch_size)
    df = pd.read_csv(csv_file_path,header=0,index_col=None)
    
    # 初始化实时检测器以使用其fill_missing_timestamps方法

    df = detector.fill_missing_timestamps(df, interval=60)
    
    print(len(df))
    df = df.iloc[len(df)//2:]
    print(len(df))
    print(df.iloc[0]['timestamp'])
    print(f"开始模拟实时数据采集与异常检测...")
    print(f"数据源: {csv_file_path}")
    print(f"模型路径: {model_path}")
    print(f"窗口大小: {window_size}")
    print(f"批次大小: {batch_size}")
    print(f"采样间隔: {latency} 秒")
    print("-" * 50)
    
    # 用于存储预测结果和真实标签
    predictions = []
    ground_truths = []
    timestamps = []
    values = []
    scores = []
    
    anomaly_count = 0
    total_points = 0

    # 逐条读取数据并进行检测
    for idx, row in df.iterrows():
    
        value = row['value']
        timestamp = row['timestamp']
        label = row.get('pred', 0)   # 预测标签（如果存在）
        truth = row.get('label', 0)  # 真实标签（如果存在）
        missing = row.get('missing', 0)  # 缺失标记（如果存在）
        
        # 添加数据点并检查是否有检测结果
        result = detector.add_data_point(value, timestamp, label, truth, missing)
        
        if result:
            for res in result:
                total_points += 1
                if res['is_anomaly']:
                    anomaly_count += 1
                
                # 保存预测结果和真实标签
                predictions.append(int(res['is_anomaly']))
                ground_truths.append(int(res['truth']))  # 注意：这里可能需要根据实际情况调整truth的获取方式
                timestamps.append(res['timestamp'])
                values.append(res['value'])
                scores.append(res['anomaly_score'])
                
                # 打印检测结果
                status = "ANOMALY" if res['is_anomaly'] else "NORMAL"
                if total_points % 1000 == 0:
                    print(f"[{status}] "
                        f"Timestamp: {res['timestamp']}, "
                        f"Value: {res['value']:.4f}, "
                        f"Score: {res['anomaly_score']:.4f}, "
                        f"Anomalies: {anomaly_count}/{total_points}")

        # 模拟数据采集延迟
        #time.sleep(latency)
    
    # 处理剩余未满一个批次的数据
    if len(detector.batch_windows) > 0:
        print(f"处理最后 {len(detector.batch_windows)} 个窗口...")
        # 可以选择处理剩余数据，这里为了简化不处理
        
    print("-" * 50)
    print(f"模拟完成. 总共处理 {total_points} 个数据点，检测到 {anomaly_count} 个异常.")

    # 计算评估指标
    if predictions and ground_truths:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 1)
        tn = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 0)
        
        # 计算派生指标
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        
        # 计算F1-score
        best_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        
        print("\n评估结果:")
        
        print(f"True Positives: {tp}, False Positives: {fp}")
        print(f"False Negatives: {fn}, True Negatives: {tn}")
        print(f"FPR (False Positive Rate): {fpr:.4f}")
        print(f"FNR (False Negative Rate): {fnr:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {best_f1:.4f}")
        
        # 保存结果到CSV文件
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'anomaly_score': scores,
            'prediction': predictions,
            'ground_truth': ground_truths
        })
        
        # 保存到文件
        output_file = "realtime_detection_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")
        
        # 保存评估指标摘要
        summary_df = pd.DataFrame([{
            'FPR': fpr,
            'FNR': fnr,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': best_f1,
            'Total_Points': total_points,
            'Anomalies_Detected': anomaly_count
        }])
        
        summary_file = "realtime_detection_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"评估摘要已保存到: {summary_file}")
    
    return predictions, ground_truths

def main():
    """
    主函数 - 演示实时异常检测
    sr-vae是用一整个时间窗口的数据对该时间窗口最后一个数据做预测
    """
    # 配置参数
    data_file = "./data/kpi_test_sr.csv"  # 测试数据文件
    model_file = "./saved_models/kpi_test_sr_final_model111.pth"  # 模型文件路径
    window_size = 128
    batch_size = 256  # 批次大小
    latency = 0.1  # 100ms采集间隔
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 数据文件 {data_file} 不存在")
        return
        
    if not os.path.exists(model_file):
        print(f"错误: 模型文件 {model_file} 不存在")
        print("请先运行训练程序生成模型文件")
        return
    
    # 开始模拟实时检测
    simulate_realtime_detection(data_file, model_file, window_size, latency, batch_size)

if __name__ == "__main__":
    main()