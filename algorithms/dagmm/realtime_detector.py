# file: realtime_detector.py
import os
import time
import csv
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from model import DaGMM
import pickle

class RealTimeDaGMMPredictor:
    def __init__(self, model, scaler, window_size, input_dim, batch_size=32, gmm_k=3, threshold=None):
        """
        实时DaGMM模型预测器
        
        Args:
            model: 训练好的DaGMM模型
            scaler: 数据标准化器
            window_size: 窗口大小
            input_dim: 输入数据维度
            batch_size: 批处理大小
            gmm_k: GMM组件数量
            threshold: 异常检测阈值
        """
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.gmm_k = gmm_k
        
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
        
        # 阈值
        self.threshold = threshold
        
    def add_data_point(self, timestamp, values, label=None):
        """
        添加单个数据点到缓冲区
        Args:
            timestamp: 时间戳
            values: 特征值 (list or array)
            label: 标签 (可选)
        """
        self.timestamp_buffer.append(timestamp)
        self.data_buffer.append(values)
        if label is not None:
            self.label_buffer.append(label)
        
        # 如果缓冲区满了，执行一次预测
        if len(self.data_buffer) >= self.window_size:
            self._process_batch()
            
    def _process_batch(self):
        """
        处理一个批次的数据
        """
        if len(self.data_buffer) < self.window_size:
            return
            
        # 获取最近的窗口数据
        recent_data = np.array(self.data_buffer[-self.window_size:])  # shape: (window_size, n_features)
        recent_timestamps = self.timestamp_buffer[-self.window_size:]
        recent_labels = self.label_buffer[-self.window_size:] if self.label_buffer else [0] * self.window_size
        
        # 如果是单特征数据，确保形状正确
        if len(recent_data.shape) == 1:
            recent_data = recent_data.reshape(-1, 1)
            
        # 数据标准化
        if self.scaler is not None:
            data_scaled = self.scaler.transform(recent_data)
        else:
            data_scaled = recent_data
            
        # 取最后一个数据点进行预测
        data_point = data_scaled[-1]  # 取最后一个点
        
        # 转换为tensor
        data_tensor = torch.from_numpy(data_point).float()
        if torch.cuda.is_available():
            data_tensor = data_tensor.cuda()
            
        # 使用模型进行预测
        with torch.no_grad():
            self.model.eval()
            enc, dec, z, gamma = self.model(data_tensor.unsqueeze(0))  # 添加batch维度
            
            # 计算异常分数
            if self.model.phi is not None and self.model.mu is not None and self.model.cov is not None:
                
                sample_energy, cov_diag = self.model.compute_energy(
                    z, 
                    phi=self.model.phi, 
                    mu=self.model.mu, 
                    cov=self.model.cov, 
                    size_average=False
                )
            else:
                # 计算当前点的GMM参数
                sample_energy, cov_diag = self.model.compute_energy(z, size_average=False)
            
            # 获取异常分数
  
            energy_score = sample_energy.item()
            
            # 进行异常判断
            is_anomaly = 0
            if self.threshold is not None:
                is_anomaly = 1 if energy_score > self.threshold else 0
            
            # 记录预测结果
            current_timestamp = recent_timestamps[-1]
            
            self.pred_timestamps.append(current_timestamp)
            self.predictions.append(is_anomaly)
            self.pred_scores.append(energy_score)
            self.pred_labels.append(recent_labels[-1] if recent_labels else 0)
            
            self.total_processed += 1
            if is_anomaly:
                self.anomalies_detected += 1
                
        # 移除最早的数据点以维持滑动窗口
        if len(self.data_buffer) >= self.window_size:
            self.data_buffer.pop(0)
            self.timestamp_buffer.pop(0)
            if self.label_buffer:
                self.label_buffer.pop(0)
    
    def get_results(self):
        """
        获取预测结果
        """
        return {
            'timestamps': self.pred_timestamps,
            'predictions': self.predictions,
            'labels': self.pred_labels,
            'scores': self.pred_scores,
            'total_processed': self.total_processed,
            'anomalies_detected': self.anomalies_detected,
            'threshold': self.threshold
        }

def load_npz_data(data_path):
    """
    加载.npz格式的数据文件
    """
    data = np.load(data_path, allow_pickle=True)
    features = data["kdd"][:,:-1]
    labels = data["kdd"][:,-1]
    
    # 翻转标签以匹配实时检测的约定（1表示异常，0表示正常）
    labels = 1 - labels
    
    # 生成时间戳
    timestamps = np.arange(len(features))
    
    return features, labels, timestamps

def simulate_real_time_prediction(data_file, model, scaler, window_size, input_dim, 
                                 batch_size=32, gmm_k=3, threshold=None):
    """
    模拟实时预测过程
    
    Args:
        data_file: .npz数据文件路径
        model: 训练好的模型
        scaler: 数据标准化器
        window_size: 窗口大小
        input_dim: 输入数据维度
        batch_size: 批次大小
        gmm_k: GMM组件数量
        threshold: 异常检测阈值
    """
    # 加载数据
    print("加载数据...")
    features, labels, timestamps = load_npz_data(data_file)
    
    # 初始化实时预测器
    predictor = RealTimeDaGMMPredictor(model, scaler, window_size, input_dim, batch_size, gmm_k, threshold)
    
    print(f"开始模拟实时预测，共 {len(features)} 条数据")
    print(f"特征数量: {input_dim}")
    
    # 处理数据
    for i in range(len(features)):
        timestamp = timestamps[i]
        values = features[i]
        label = labels[i]
        
        # 添加数据点
        predictor.add_data_point(timestamp, values, label)
        
        if i % 1000 == 0 and len(predictor.pred_timestamps) > 0:
            latest_timestamp = predictor.pred_timestamps[-1]
            latest_score = predictor.pred_scores[-1]
            print(f"处理进度: {i}/{len(features)}, 最新分数: {latest_score:.6f}")
    
    print("实时预测完成")
    return predictor

def load_dagmm_model(model_path, input_dim=38, gmm_k=3):
    """
    加载训练好的DaGMM模型（使用与solver中保存路径一致的格式）
    """
    # 初始化模型
    model = DaGMM(n_gmm=gmm_k, input_dim=input_dim)
    
    # 加载模型权重
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    return model

def find_optimal_threshold(y_true, y_scores):
    """
    使用ROC曲线找到最佳阈值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # 计算约登指数 (Youden's J statistic = sensitivity + specificity - 1)
    # sensitivity = tpr, specificity = 1 - fpr
    youden_j = tpr - fpr  # 等价于 (tpr + (1-fpr) - 1)
    
    # 找到约登指数最大的索引
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

def evaluate_more_metrics(y_true, y_pred):
    """计算更多评估指标"""
    from sklearn.metrics import confusion_matrix
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # 计算各种指标
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # 同时也是TPR
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 计算额外指标
    tpr = recall  # True Positive Rate (敏感度)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (特异性)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tpr': tpr,
        'fnr': fnr,
        'tnr': tnr,
        'fpr': fpr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def main():
    parser = argparse.ArgumentParser(description='DaGMM Real-time Prediction')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='dataset name (smd, smap, psm, msl, swat, kpi)')
    parser.add_argument('--window', type=int, default=100, help='window size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--gmm_k', type=int, default=3, help='number of GMM components')
    parser.add_argument('--data', type=str, help='NPZ data file path (if not provided, will use default)')
    parser.add_argument('--model_path', type=str, default='', help='model path (if empty, use default)')
    parser.add_argument('--scaler_path', type=str, default='', help='scaler path (if empty, use default)')
    parser.add_argument('--output', type=str, default='dagmm_realtime_results.csv', help='output file path')
    parser.add_argument('--threshold', type=float, default=1.23, help='anomaly detection threshold')
    
    args = parser.parse_args()
    
    # 数据集配置
    dataset_config = {
        'smd': {'dim': 38},
        'smap': {'dim': 25},
        'psm': {'dim': 25},
        'msl': {'dim': 55},
        'swat': {'dim': 51},
        'kpi': {'dim': 1}
    }
    
    if args.dataset not in dataset_config:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets: {list(dataset_config.keys())}")
    
    input_dim = dataset_config[args.dataset]['dim']
    
    # 确定模型路径
    if not args.model_path:
        model_path = f"./dagmm/models/{args.dataset}/200_4_dagmm.pth"
    else:
        model_path = args.model_path
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 加载模型
    print("加载DaGMM模型...")
    model = load_dagmm_model(model_path, input_dim, args.gmm_k)
    
    
    # 确定数据文件路径
    if not args.data:
        # 使用.npz格式的数据文件
        data_file = f"./processed_data1/{args.dataset}_standardized.npz"
    else:
        data_file = args.data
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # 执行实时预测模拟
    start_time = time.time()
    predictor = simulate_real_time_prediction(
        data_file,
        model,
        None,
        args.window,
        input_dim,
        args.batch_size,
        args.gmm_k,
        args.threshold
    )
    end_time = time.time()
    
    # 获取预测结果
    results = predictor.get_results()
    
    # 提取预测结果和真实标签
    predictions = results['predictions']
    ground_truths = results['labels']
    scores = results['scores']
    
    # 过滤掉None值（如果有的话）
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    metrics_info = {}  # 初始化指标信息字典
    
    if valid_indices and any(label is not None for label in ground_truths):
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_ground_truths = [ground_truths[i] for i in valid_indices if ground_truths[i] is not None]
        valid_scores = [scores[i] for i in valid_indices]
        
        # 使用ROC找到最佳阈值
        optimal_threshold = find_optimal_threshold(valid_ground_truths, valid_scores)
        print(f"Optimal threshold (based on ROC): {optimal_threshold:.6f}")
        
        # 使用最佳阈值重新计算预测结果
        optimal_predictions = [1 if score > optimal_threshold else 0 for score in valid_scores]
        
        # 计算评估指标
        metrics_info = evaluate_more_metrics(valid_ground_truths, optimal_predictions)
        
        print(f"Evaluation Metrics:")
        print(f"  Precision: {metrics_info['precision']:.4f}")
        print(f"  Recall: {metrics_info['recall']:.4f}")
        print(f"  F1-Score: {metrics_info['f1']:.4f}")
        print(f"  TPR (True Positive Rate): {metrics_info['tpr']:.4f}")
        print(f"  FNR (False Negative Rate): {metrics_info['fnr']:.4f}")
        print(f"  TNR (True Negative Rate): {metrics_info['tnr']:.4f}")
        print(f"  FPR (False Positive Rate): {metrics_info['fpr']:.4f}")
    
    # 保存详细结果到CSV文件
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'prediction', 'label', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 写入数据
        for i in range(len(results['timestamps'])):
            writer.writerow({
                'timestamp': int(results['timestamps'][i]),  
                'prediction': int(results['predictions'][i]) if results['predictions'][i] is not None else 0,
                'label': int(results['labels'][i]) if results['labels'][i] is not None else 0,
                'score': float(results['scores'][i]) 
            })
    
    # 将评估指标写入另一个CSV文件
    metrics_output = args.output.replace('.csv', '_metrics.csv')
    with open(metrics_output, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['metric', 'value']
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'value'])
        if metrics_info:
            writer.writerow(['Precision', f"{metrics_info['precision']:.6f}"])
            writer.writerow(['Recall/TPR', f"{metrics_info['recall']:.6f}"])
            writer.writerow(['F1-Score', f"{metrics_info['f1']:.6f}"])
            writer.writerow(['FNR', f"{metrics_info['fnr']:.6f}"])
            writer.writerow(['TNR', f"{metrics_info['tnr']:.6f}"])
            writer.writerow(['FPR', f"{metrics_info['fpr']:.6f}"])
            writer.writerow(['TP', metrics_info['tp']])
            writer.writerow(['TN', metrics_info['tn']])
            writer.writerow(['FP', metrics_info['fp']])
            writer.writerow(['FN', metrics_info['fn']])
            # writer.writerow(['Threshold', optimal_threshold if 'optimal_threshold' in locals() else args.threshold])
            writer.writerow(['Total_Processed', results['total_processed']])
            writer.writerow(['Anomalies_Detected', results['anomalies_detected']])
            writer.writerow(['Processing_Time', f"{end_time - start_time:.2f}"])
    
    print(f"\n预测统计:")
    print(f"总处理数据点: {results['total_processed']}")
    print(f"检测到异常: {results['anomalies_detected']}")
    print(f"处理时间: {end_time - start_time:.2f} 秒")
    
    print(f"\n结果已保存到: {args.output}")
    print(f"评估指标已保存到: {metrics_output}")

if __name__ == '__main__':
    main()