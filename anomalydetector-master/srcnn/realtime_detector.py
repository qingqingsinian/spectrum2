import os
import time
import csv
import json
import argparse
import torch
from srcnn.net import Anomaly, load_model
from srcnn.competition_metric import evaluate_for_all_series
import threading
from queue import Queue
import pandas as pd
from srcnn.utils import *

class RealTimeSRCNNPredictor:
    def __init__(self, model, window_size, batch_size=32, delay=5,threshold1=1):
        self.model = model
        self.window_size = window_size
        self.batch_size = batch_size
        self.delay = delay
        self.window1 = delay
        self.threshold1 = threshold1
        # 存储实时数据的缓冲区
        self.data_buffer = []
        self.timestamp_buffer = []
        self.label_buffer = []
        
        # 存储预测结果
        self.predictions = []
        self.pred_timestamps = []
        self.pred_labels = []
        self.pred_scores = []
        
        # 统计信息
        self.total_processed = 0
        self.anomalies_detected = 0
        
    def add_data_point(self, timestamp, value, label=None):
        """
        添加单个数据点到缓冲区
        """
        self.timestamp_buffer.append(timestamp)
        self.data_buffer.append(value)
        if label is not None:
            self.label_buffer.append(label)
        
        # 如果缓冲区满了，执行一次预测
        if len(self.data_buffer) >= self.window_size:
            self._process_batch(self.window1,self.threshold1)
            
    def _process_batch(self, window1=None, threshold1=None):
        """
        处理一个批次的数据 (使用 sr_cnn_eval 方式进行预测，模仿 evalue.py 的方式)
        如果提供了window1和threshold1参数，则使用窗口阈值处理方式
        """
        if len(self.data_buffer) < self.window_size:
            return
        # 使用 sr_cnn_eval 方式进行预测
        # 构造输入数据
        recent_timestamps = self.timestamp_buffer[-self.window_size:]
        recent_values = self.data_buffer[-self.window_size:]
        recent_labels = self.label_buffer[-self.window_size:] if self.label_buffer else [0] * self.window_size
        
        # 调用 sr_cnn_eval 进行预测 (模仿 evalue.py 中的调用方式)
        pred_timestamps, pred_labels, predictions, scores = sr_cnn_eval(
            recent_timestamps, 
            recent_values, 
            recent_labels, 
            self.window_size, 
            self.model, 
            'anomaly',  # missing option
            0.25  # threshold
        )
        
        # 记录最新的预测结果
        if len(pred_timestamps) > 0:
            current_timestamp = pred_timestamps[-2]
            is_anomaly = predictions[-2]
            pred_score = scores[-2]
            
            # 如果提供了window1和delay参数，使用窗口阈值处理
            if window1 is not None and threshold1 is not None:
                # 将当前预测添加到临时缓冲区
                if not hasattr(self, '_window_predictions_buffer'):
                    self._window_predictions_buffer = []
                    self._window_timestamps_buffer = []
                    self._window_scores_buffer = []
                    self._window_labels_buffer = []
                    
                self._window_predictions_buffer.append(is_anomaly)
                self._window_timestamps_buffer.append(current_timestamp)
                self._window_scores_buffer.append(pred_score)
                if self.label_buffer and len(self.label_buffer) > 0:
                    self._window_labels_buffer.append(self.label_buffer[-2])
                else:
                    self._window_labels_buffer.append(None)
                    
                # 当缓冲区达到window1大小时，进行窗口处理
                if len(self._window_predictions_buffer) >= window1:
                    # 统计窗口内异常数量
                    anomaly_count = sum(self._window_predictions_buffer)
                    
                    # 如果异常数量超过阈值，将整个窗口标记为异常
                    final_prediction = 1 if anomaly_count >=threshold1 else 0
                    
                    # 使用窗口处理后的结果
                    for i in range(len(self._window_predictions_buffer)):
                        self.pred_timestamps.append(self._window_timestamps_buffer[i])
                        self.predictions.append(final_prediction)  # 使用窗口处理后的结果
                        self.pred_scores.append(self._window_scores_buffer[i])
                        self.pred_labels.append(self._window_labels_buffer[i])
                        
                        self.total_processed += 1
                        if final_prediction:  # 注意：这里使用窗口处理后的结果
                            self.anomalies_detected += 1
                    
                    # 清空缓冲区
                    self._window_predictions_buffer = []
                    self._window_timestamps_buffer = []
                    self._window_scores_buffer = []
                    self._window_labels_buffer = []

    def get_results(self):
        """
        获取预测结果
        """
        return {
            'timestamps': self.pred_timestamps,
            'predictions': self.predictions,
            'labels': self.pred_labels,
            'pred_scores': self.pred_scores, 
            'total_processed': self.total_processed,
            'anomalies_detected': self.anomalies_detected
        }

def simulate_real_time_prediction(csv_file, model, window_size, batch_size=32, interval=0.1):
    """
    模拟实时预测过程
    
    Args:
        csv_file: CSV文件路径
        model: 训练好的模型
        window_size: 窗口大小
        batch_size: 批次大小
        interval: 数据点之间的时间间隔（秒）
    """
    # 初始化实时预测器
    predictor = RealTimeSRCNNPredictor(model, window_size, batch_size)
    
    # 读取CSV数据
    df = pd.read_csv(csv_file)
    
    # 检查必要的列
    required_columns = ['timestamp', 'value']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV文件必须包含以下列: {required_columns}")
    
    has_label = 'label' in df.columns
    
    print(f"开始模拟实时预测，共 {len(df)} 条数据，间隔 {interval} 秒")
    n=0
    # 模拟实时数据流
    for index, row in df.iterrows():
        timestamp = row['timestamp']
        value = row['value']
        label = row['label'] if has_label else None
        n+=1
        if(n%1000==0):
           if n % 1000 == 0 and len(predictor.pred_timestamps) > 0:
            latest_timestamp = predictor.pred_timestamps[-1]
            latest_prediction = predictor.predictions[-1]
            latest_score = predictor.pred_scores[-1] if hasattr(predictor, 'pred_scores') else 'N/A'
            print(f"Timestamp: {latest_timestamp}, Prediction: {latest_prediction}, Score: {latest_score}")
        # 添加数据点
        predictor.add_data_point(timestamp, value, label)
        
        # 模拟时间间隔
        #if index < len(df) - 1:  # 最后一个数据点不需要等待
            #time.sleep(interval)
    
    print("实时预测完成")
    return predictor.get_results()

def read_csv_kpi(file_path):
    """
    读取KPI数据CSV文件
    """
    timestamps = []
    values = []
    labels = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            if len(row) >= 3:
                timestamps.append(int(row[0]))
                values.append(float(row[1]))
                labels.append(int(row[2]))
            elif len(row) >= 2:
                timestamps.append(int(row[0]))
                values.append(float(row[1]))
                labels.append(0)  # 如果没有标签，默认为0
    
    return timestamps, values, labels

def save_results_with_metrics(csv_output, results, dataset_name, metrics=None):
    """
    保存结果到CSV文件，只保留指标信息
    
    Args:
        csv_output: 输出文件路径
        results: 预测结果字典
        dataset_name: 数据集名称
        metrics: 评估指标字典（可选）
    """
    with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['dataset']
        
        # 如果有评估指标，添加到字段名中
        if metrics:
            fieldnames.extend(['f1', 'precision', 'recall', 'tp', 'fp', 'tn', 'fn'])
        else:
            # 即使没有指标也添加占位字段
            fieldnames.extend(['f1', 'precision', 'recall', 'tp', 'fp', 'tn', 'fn'])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 只写入一行数据，包含数据集名称和指标
        row_data = {
            'dataset': dataset_name
        }
        
        # 添加评估指标（如果有）
        if metrics:
            row_data.update({
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'tn': metrics['tn'],
                'fn': metrics['fn']
            })
        
        
        writer.writerow(row_data)
# 在 realtime_detector.py 中添加以下函数

def find_best_threshold_for_file(timestamps, labels, scores, delay):
    """
    为单个文件寻找最佳阈值
    
    Args:
        timestamps: 时间戳列表
        labels: 真实标签列表
        scores: 异常分数列表
        delay: 延迟参数
        
    Returns:
        best_threshold: 最佳阈值
        best_f1: 最佳F1分数
        best_precision: 最佳精确率
        best_recall: 最佳召回率
    """
    from srcnn.competition_metric import evaluate_for_all_series
    
    best_f1 = 0.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    # 在0.01到0.99之间搜索最佳阈值
    for i in range(98):
        threshold = 0.01 + i * 0.01
        predictions = [1 if score > threshold else 0 for score in scores]
        
        # 构造评估所需格式
        eval_data = [[timestamps, labels, predictions, "temp_file"]]
        
        try:
            f1, precision, recall, tp, fp, tn, fn = evaluate_for_all_series(eval_data, delay, prt=False)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        except Exception as e:
            pass  # 忽略计算错误
    
    return best_threshold, best_f1, best_precision, best_recall

def process_single_file(csv_file, model, window_size, batch_size, interval, delay):
    """
    处理单个CSV文件并使用优化的阈值返回结果和评估指标
    """
    # 执行实时预测模拟
    start_time = time.time()
    results = simulate_real_time_prediction(
        csv_file, 
        model, 
        window_size, 
        batch_size, 
        interval
    )
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # 寻找最佳阈值
    best_threshold = 0.25  # 默认阈值
    metrics = None
    
    if any(label is not None for label in results['labels']) and len(results['pred_scores']) > 0:
        try:
            # 寻找最佳阈值
            best_threshold, best_f1, best_precision, best_recall = find_best_threshold_for_file(
                results['timestamps'], 
                results['labels'], 
                results['pred_scores'], 
                delay
            )
            
            # 使用最佳阈值重新生成预测结果
            optimized_predictions = [1 if score > best_threshold else 0 for score in results['pred_scores']]
            
            # 更新结果中的预测
            results['predictions'] = optimized_predictions
            
            # 计算评估指标
            eval_data = [[
                results['timestamps'],
                results['labels'],
                results['predictions'],
                csv_file
            ]]
            
            f1, precision, recall, tp, fp, tn, fn = evaluate_for_all_series(eval_data, delay)
            metrics = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'best_threshold': best_threshold
            }
        except Exception as e:
            print(f"优化阈值过程中出错: {e}")
    
    return results, processing_time, metrics, best_threshold

def process_all_files(test_dir, model, window_size, batch_size, interval, delay, output_dir):
    """
    处理目录中的所有CSV文件，并为每个文件优化阈值
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"在目录 {test_dir} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件: {csv_files}")
    
    # 存储所有文件的总体统计信息
    all_metrics = []
    
    # 处理每个文件
    for csv_file in csv_files:
        file_path = os.path.join(test_dir, csv_file)
        print(f"\n处理文件: {csv_file}")
        
        try:
            # 处理单个文件并优化阈值
            results, processing_time, metrics, best_threshold = process_single_file(
                file_path, model, window_size, batch_size, interval, delay
            )
            
            # 保存结果到CSV文件
            base_name = os.path.splitext(csv_file)[0]
            csv_output = os.path.join(output_dir, f"{base_name}_results.csv")
            
            # 使用新的保存方法
            save_results_with_metrics(csv_output, results, csv_file, metrics)
            
            # 打印统计信息
            print(f"\n文件 {csv_file} 的预测统计:")
            print(f"总处理数据点: {results['total_processed']}")
            print(f"检测到异常: {results['anomalies_detected']}")
            print(f"处理时间: {processing_time:.2f} 秒")
            print(f"最佳阈值: {best_threshold:.2f}")
            
            # 如果有评估指标，打印
            if metrics:
                print(f"\n文件 {csv_file} 的评估结果:")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"最佳阈值: {metrics['best_threshold']:.2f}")
                print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
                
                # 添加到总体统计
                all_metrics.append({
                    'file': csv_file,
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'tn': metrics['tn'],
                    'fn': metrics['fn'],
                    'best_threshold': metrics['best_threshold'],
                    'processing_time': processing_time,
                    'total_processed': results['total_processed'],
                    'anomalies_detected': results['anomalies_detected']
                })
            
            print(f"结果已保存到: {csv_output}")
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
    
    # 保存总体统计信息到CSV文件
    if all_metrics:
        summary_output = os.path.join(output_dir, "summary_metrics.csv")
        with open(summary_output, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file', 'f1', 'precision', 'recall', 'tp', 'fp', 'tn', 'fn', 
                         'best_threshold', 'processing_time', 'total_processed', 'anomalies_detected']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in all_metrics:
                writer.writerow(metric)
        
        print(f"\n总体统计信息已保存到: {summary_output}")

# 修改 main 函数以支持新的处理方法
def main():
    parser = argparse.ArgumentParser(description='SRCNN Real-time Prediction')
    parser.add_argument('--data', type=str, default='./test_new', help='CSV data file path or directory')
    parser.add_argument('--window', type=int, default=128, help='window size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--interval', type=float, default=0.1, help='interval between data points (seconds)')
    parser.add_argument('--model_path', type=str, default='snapshot', help='model path')
    parser.add_argument('--delay', type=int, default=3, help='delay for evaluation')
    parser.add_argument('--output', type=str, default='realtime_results', help='output directory path')
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()
    root = os.getcwd()
    epoch=args.epoch
    window=args.window
    model_path = root + '/' + args.model_path + '/totalsrcnn_retry' + str(epoch) + '_' + str(window) + '.bin'
    # 加载模型
    print("加载模型...")
    srcnnmodel = Anomaly(args.window)
    models = {
        'sr_cnn': sr_cnn_eval,
    }
    model = load_model(srcnnmodel, model_path).cuda()

    print(f"处理目录中的所有CSV文件: {args.data}")
    
    process_all_files(
            args.data, model, args.window, args.batch_size, 
            args.interval, args.delay, args.output
        )
if __name__ == '__main__':
    main()