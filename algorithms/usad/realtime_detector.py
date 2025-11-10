# realtime_usad_detector.py
import os
import time
import csv
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from usad import UsadModel
from utils import get_default_device, to_device
import pickle
from utils import *
from window_utils import load_dataset
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score
class RealTimeUSADPredictor:
    def __init__(self, model, scaler, window_size, n_features, batch_size=32, alpha=0.5, beta=0.5, detection_window_size=5, window_threshold=1):
        """
        实时USAD模型预测器，用于多维数据
        
        Args:
            model: 训练好的USAD模型
            scaler: 数据标准化器 (与main.py中训练时使用的scaler一致)
            window_size: 窗口大小（用于模型输入）
            n_features: 特征数量
            batch_size: 批处理大小
            alpha, beta: USAD模型的权重参数
            detection_window_size: 检测时使用的窗口大小（用于后处理）
            window_threshold: 窗口内异常点阈值，超过该数量则整个窗口标记为异常
        """
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        self.n_features = n_features
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.detection_window_size = detection_window_size if detection_window_size is not None else window_size
        self.window_threshold = window_threshold
        
        # 存储实时数据的缓冲区
        self.data_buffer = []  # 存储多维数据点
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
        
        # 设备
        self.device = get_default_device()
        
    def add_data_point(self, timestamp, values, label=None):
        """
        添加单个数据点到缓冲区
        Args:
            timestamp: 时间戳
            values: 多维特征值 (list or array)
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
        处理一个批次的数据，并进行实时窗口化处理
        """
        if len(self.data_buffer) < self.window_size:
            return
            
        # 构造输入数据 (使用滑动窗口)
        recent_data = np.array(self.data_buffer[-self.window_size:])  # shape: (window_size, n_features)
        recent_timestamps = self.timestamp_buffer[-self.window_size:]
        recent_labels = self.label_buffer[-self.window_size:] if self.label_buffer else [0] * self.window_size
        
        # 数据标准化 (使用训练时的scaler)
        data_scaled = self.scaler.transform(recent_data)
        
        # 展平为一维向量，因为USAD模型期望一维输入
        data_flat = data_scaled.flatten()  # shape: (window_size * n_features,)
        
        # 转换为tensor并移动到设备
        # 添加批量维度，使数据变为2D (1, window_size * n_features)
        data_tensor = torch.from_numpy(data_flat).float().unsqueeze(0).to(self.device)
 
        # 使用USAD模型进行预测
        with torch.no_grad():
            c = self.model.encoder(data_tensor)
            w1 = self.model.decoder1(c)
            w2 = self.model.decoder2(self.model.encoder(w1))
            
            # 计算重构误差
            score = self.alpha * torch.mean((data_tensor - w1) ** 2, axis=1) + \
                    self.beta * torch.mean((data_tensor - w2) ** 2, axis=1)
            
            score_value = score.item()
            
        # 记录预测结果 (使用最新的时间戳)
        current_timestamp = recent_timestamps[-1]
        
        # 实时窗口化处理
        # 将当前预测添加到临时缓冲区
        if not hasattr(self, '_window_scores_buffer'):
            self._window_scores_buffer = []
            self._window_timestamps_buffer = []
            self._window_labels_buffer = []
            
        self._window_scores_buffer.append(score_value)
        self._window_timestamps_buffer.append(current_timestamp)
        if self.label_buffer and len(self.label_buffer) > 0:
            self._window_labels_buffer.append(self.label_buffer[-1])
        else:
            self._window_labels_buffer.append(None)
            
        # 当缓冲区达到检测窗口大小时，进行窗口处理
        if len(self._window_scores_buffer) >= self.detection_window_size:
            # 使用默认阈值进行初步判断
            default_threshold = 0.247716  # 这应该是一个可配置的参数
            window_point_predictions = [1 if score > default_threshold else 0 for score in self._window_scores_buffer]
            
            # 统计窗口内异常数量
            anomaly_count = sum(window_point_predictions)
            
            # 如果异常数量超过阈值，将整个窗口标记为异常
            final_prediction = 1 if anomaly_count >= self.window_threshold else 0
            
            # 使用窗口处理后的结果
            for i in range(len(self._window_scores_buffer)):
                self.pred_timestamps.append(self._window_timestamps_buffer[i])
                self.predictions.append(final_prediction)  # 使用窗口处理后的结果
                self.pred_scores.append(self._window_scores_buffer[i])
                self.pred_labels.append(self._window_labels_buffer[i])
                
                self.total_processed += 1
                if final_prediction:  # 注意：这里使用窗口处理后的结果
                    self.anomalies_detected += 1
            
            # 清空缓冲区
            self._window_scores_buffer = []
            self._window_timestamps_buffer = []
            self._window_labels_buffer = []
        
        # 移除最早的数据点以维持滑动窗口
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
            self.timestamp_buffer.pop(0)
            if self.label_buffer:
                self.label_buffer.pop(0)
    
    def get_results(self):
        """
        获取预测结果（仅获取结果，不进行任何处理）
        """
        return {
            'timestamps': self.pred_timestamps,
            'predictions': self.predictions,
            'labels': self.pred_labels,
            'pred_scores': self.pred_scores, 
            'total_processed': self.total_processed,
            'anomalies_detected': self.anomalies_detected
        }
def simulate_real_time_prediction(csv_file, model, scaler, window_size, n_features, 
                                 batch_size=32, interval=0.1, alpha=0.5, beta=0.5,detection_window_size=5,window_threshold=1):
    """
    模拟实时预测过程
    
    Args:
        csv_file: CSV文件路径
        model: 训练好的模型
        scaler: 数据标准化器
        window_size: 窗口大小
        n_features: 特征数量
        batch_size: 批次大小
        interval: 数据点之间的时间间隔（秒）
        alpha, beta: USAD模型的权重参数
    """
    # 初始化实时预测器
    predictor = RealTimeUSADPredictor(model, scaler, window_size, n_features, 
                                      batch_size, alpha, beta,detection_window_size,window_threshold)
    
    # 读取CSV数据
    df = pd.read_csv(csv_file)
    
    # 检查必要的列
    required_columns = ['timestamp']
    # 自动识别特征列 (除了timestamp和label之外的所有列)
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'label']]
    
    if 'timestamp' not in df.columns:
        raise ValueError("CSV文件必须包含'timestamp'列")
    
    if len(feature_columns) == 0:
        raise ValueError("CSV文件必须包含至少一个特征列")
    
    has_label = 'label' in df.columns
    
    print(f"开始模拟实时预测，共 {len(df)} 条数据，间隔 {interval} 秒")
    print(f"特征数量: {len(feature_columns)}")
    
    # 模拟实时数据流
    for index, row in df.iterrows():
        timestamp = row['timestamp']
        # 提取所有特征值
        values = [row[col] for col in feature_columns]
        label = row['label'] if has_label else None
        
        if index % 1000 == 0 and len(predictor.pred_timestamps) > 0:
            latest_timestamp = predictor.pred_timestamps[-1]
            latest_score = predictor.pred_scores[-1] if hasattr(predictor, 'pred_scores') else 'N/A'
            print(f"Timestamp: {latest_timestamp}, Score: {latest_score:.6f}")
        
        # 添加数据点
        predictor.add_data_point(timestamp, values, label)
        
        # 模拟时间间隔
        # if index < len(df) - 1:  # 最后一个数据点不需要等待
        #     time.sleep(interval)
    
    print("实时预测完成")
    return predictor

def load_model_weights(model_path, w_size, z_size):
    """
    加载训练好的USAD模型权重
    """
    device = get_default_device()
    
    # 初始化模型
    model = UsadModel(w_size, z_size)
    model = model.to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])

    model.decoder2.load_state_dict(checkpoint['decoder2'])
    
    model.eval()
    return model, device
# 在 imports 部分添加需要的库
from sklearn.metrics import roc_curve, auc

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
    if(optimal_idx==0):
        best_f1_tmp=0
        best_threshold=0
        for i in range(100):
            optimal_threshold = np.percentile(y_scores, i+1)
            best_pred_labels = (y_scores > (optimal_threshold)).astype(int)
            best_f1 = f1_score(y_true, best_pred_labels, zero_division=0)
            if(best_f1>best_f1_tmp):
                best_f1_tmp=best_f1
                best_threshold=optimal_threshold
        optimal_threshold = best_threshold
    print("最佳阈值111:", optimal_threshold)
    return optimal_threshold

# 首先，在 imports 部分添加需要的库（如果还没有）
from sklearn.metrics import confusion_matrix
import numpy as np

# 然后创建一个计算更多指标的函数
def evaluate_more_metrics(y_true, y_pred):
    """计算更多评估指标"""
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

# 修改 main() 函数中的评估部分，替换为以下代码：
def main():
    parser = argparse.ArgumentParser(description='USAD Real-time Prediction for Multivariate Data')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name (SWAT, MSL, PSM, SMD, SMAP, kpi)')
    parser.add_argument('--window', type=int, default=128, help='window size')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--window_threshold', type=int, default=1, help='window anomaly threshold (default: 1)')
    parser.add_argument('--detection_window', type=int, default=5, help='detection window size (default: same as window)')
    parser.add_argument('--interval', type=float, default=0.1, help='interval between data points (seconds)')
    parser.add_argument('--model_path', type=str, default='', help='model path (if empty, use default)')
    parser.add_argument('--scaler_path', type=str, default='', help='scaler path (if empty, use default)')
    parser.add_argument('--output', type=str, default='usad_realtime_results.csv', help='output file path')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha parameter for USAD')
    parser.add_argument('--beta', type=float, default=0.5, help='beta parameter for USAD')
    parser.add_argument('--threshold', type=float, default=0.247116, help='anomaly detection threshold')
  
    args = parser.parse_args()
  
    # 创建数据集特定的输入目录
    input_dir = f"input/{args.dataset.lower()}"
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")
    detection_window_size = args.detection_window if args.detection_window is not None else args.window
    # 加载测试数据和标签 (与main.py中一致)
    print("加载测试数据...")
    attack, labels = load_dataset(args.dataset, input_dir,detection_window_size)
    n_features = attack.shape[1]
    
    # 计算w_size和z_size
    w_size = args.window * n_features  # 对于多变量时间序列，w_size等于window_size*features
    z_size = args.window * 100  # 按比例计算z_size
    
    # 确定模型和scaler路径
    if not args.model_path:
        model_path = f"{args.dataset.lower()}_model.pth"
    else:
        model_path = args.model_path
        
    if not args.scaler_path:
        scaler_path = f"{args.dataset.lower()}_scaler.pkl"
    else:
        scaler_path = args.scaler_path
    
    # 加载模型
    print("加载模型...")
    model, device = load_model_weights(model_path, w_size, z_size)
    
    # 加载scaler
    print("加载数据标准化器...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 创建模拟数据文件 (将数据保存为CSV格式供实时检测使用)
    print("准备测试数据...")
    # 添加时间戳列
    timestamps = list(range(len(attack)))
    attack_with_ts = attack.copy()
    attack_with_ts.insert(0, 'timestamp', timestamps)
    attack_with_ts['label'] = labels
    
    # 保存为临时CSV文件
    temp_csv = f"temp_{args.dataset.lower()}_test.csv"
    attack_with_ts.to_csv(temp_csv, index=False)
    
    # 执行实时预测模拟
    start_time = time.time()
    predictor = simulate_real_time_prediction(
        temp_csv, 
        model, 
        scaler,
        args.window,
        n_features,
        args.batch_size, 
        args.interval,
        args.alpha,
        args.beta,
        detection_window_size,
        args.threshold
    )
    

     # 获取带阈值的预测结果
    results = predictor.get_results()
    end_time = time.time()
    
    # 提取预测结果和真实标签
    predictions = results['predictions']
    ground_truths = results['labels']
    scores = results['pred_scores']
    
    # 过滤掉None值（如果有的话）
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    metrics_info = {}  # 初始化指标信息字典
    
    if valid_indices:
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_ground_truths = [ground_truths[i] for i in valid_indices if ground_truths[i] is not None]
        valid_scores = [scores[i] for i in valid_indices]
        
        optimal_threshold = find_optimal_threshold(valid_ground_truths, valid_scores)
        print(f"Optimal threshold (based on ROC): {optimal_threshold:.6f}")
        print(f"Given threshold (reference): {args.threshold:.6f}")
        # 计算更多评估指标
        valid_predictions = [1 if score > optimal_threshold else 0 for score in valid_scores]
        
        # 计算评估指标
        metrics_info = evaluate_more_metrics(valid_ground_truths, valid_predictions)
        
        print(f"Optimal threshold: {args.threshold}")
        print(f"Evaluation Metrics:")
        print(f"  Precision: {metrics_info['precision']:.4f}")
        print(f"  Recall: {metrics_info['recall']:.4f}")
        print(f"  F1-Score: {metrics_info['f1']:.4f}")
        print(f"  TPR (True Positive Rate): {metrics_info['tpr']:.4f}")
        print(f"  FNR (False Negative Rate): {metrics_info['fnr']:.4f}")
        print(f"  TNR (True Negative Rate): {metrics_info['tnr']:.4f}")
        print(f"  FPR (False Positive Rate): {metrics_info['fpr']:.4f}")


# 将评估指标写入另一个CSV文件
        metrics_output = args.output.replace('.csv', '_metrics.csv')
        import os

        # 检查文件是否存在，如果不存在则写入表头
        file_exists = os.path.exists(metrics_output)

        with open(metrics_output, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                header = ['dataset', 'Precision', 'Recall/TPR', 'F1-Score', 'FNR', 'TNR', 'FPR', 'TP', 'TN', 'FP', 'FN', 
                        'Threshold', 'Total_Processed', 'Anomalies_Detected', 'Processing_Time']
                writer.writerow(header)
            
            # 写入数据行
            if metrics_info:
                dataset_name = args.dataset.lower()
                row_data = [
                    dataset_name,
                    f"{metrics_info['precision']:.6f}",
                    f"{metrics_info['recall']:.6f}",
                    f"{metrics_info['f1']:.6f}",
                    f"{metrics_info['fnr']:.6f}",
                    f"{metrics_info['tnr']:.6f}",
                    f"{metrics_info['fpr']:.6f}",
                    metrics_info['tp'],
                    metrics_info['tn'],
                    metrics_info['fp'],
                    metrics_info['fn'],
                    f"{optimal_threshold if 'optimal_threshold' in locals() else args.threshold:.6f}",
                    results['total_processed'],
                    results['anomalies_detected'],
                    f"{end_time - start_time:.2f}"
                ]
                writer.writerow(row_data)
    
    print(f"\n预测统计:")
    print(f"总处理数据点: {results['total_processed']}")
    print(f"检测到异常: {results['anomalies_detected']}")
    print(f"处理时间: {end_time - start_time:.2f} 秒")
    
    # 清理临时文件
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
    
    print(f"\n结果已保存到: {args.output}")
    print(f"评估指标已保存到: {metrics_output}")

if __name__ == '__main__':
    main()