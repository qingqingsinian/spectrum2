import numpy as np
import pandas as pd
import os
import time
import glob
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn import preprocessing

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
        normal = normal.astype(float)
        
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
        
    elif dataset_name == 'your_dataset':  # 您当前的数据集
        # 加载训练数据
        train_path = os.path.join(input_dir, "train1.csv")
        test_path = os.path.join(input_dir, "test.csv")
        label_path = os.path.join(input_dir, "test_label.csv")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train dataset not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test dataset not found: {test_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label dataset not found: {label_path}")
        
        # 加载训练数据（正常数据）
        train_df = pd.read_csv(train_path)
        normal = pd.DataFrame(train_df.values[:, 1:])  # 去除时间戳列
        
        # 加载测试数据
        test_df = pd.read_csv(test_path)
        attack = pd.DataFrame(test_df.values[:, 1:])  # 去除时间戳列
        
        # 加载标签数据
        label_df = pd.read_csv(label_path)
        labels = label_df['label'].values.tolist()
        
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

def calculate_metrics(true_labels, predicted_labels):
    """
    计算异常检测的各种评估指标
    
    Args:
        true_labels: 真实标签
        predicted_labels: 预测标签
        
    Returns:
        dict: 包含各种指标的字典
    """
    # 基本指标
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    
    # 混淆矩阵元素
    cm = confusion_matrix(true_labels, predicted_labels)
    tn, fp, fn, tp = cm.ravel()
    
    # 计算FNR和FPR
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    
    # 其他有用指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fnr': fnr,
        'fpr': fpr,
        'accuracy': accuracy,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def fit_arima_model(data, order=(1, 1, 1)):
    """
    拟合ARIMA模型
    
    Args:
        data: 时间序列数据
        order: ARIMA模型的(p,d,q)参数
        
    Returns:
        fitted_model: 训练好的ARIMA模型
    """
    try:
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        return None

def compute_anomaly_scores_arima(data, window_size=32, order=(1, 1, 1)):
    """
    使用ARIMA模型计算异常分数（优化版本）
    
    Args:
        data: 时间序列数据 (numpy array)
        window_size: 用于训练ARIMA模型的窗口大小
        order: ARIMA模型参数(p,d,q)
        
    Returns:
        anomaly_scores: 异常分数列表
    """

    n = len(data)
    anomaly_scores = np.zeros(n)

    # 只在必要时重新拟合模型（每隔一定步数）
    retrain_interval = 1  # 每步重新训练一次模型
    model = None

    for i in range(window_size, n):
        # 检查是否需要重新训练模型

        if i % retrain_interval == 0 or model is None:
            # 获取训练窗口数据
            train_data = data[i-window_size:i]
           
            # 拟合ARIMA模型
            model = fit_arima_model(train_data, order)
        
        if model is not None:
            try:
                # 预测下一个值
                forecast = model.forecast(steps=1)
                predicted_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
                
                # 计算预测误差作为异常分数
                actual_value = data[i]
                anomaly_scores[i] = abs(actual_value - predicted_value)
            except Exception as e:
                anomaly_scores[i] = 0
        if(i % 100 == 0):
            print(i, "predict:{},ground_truth:{}".format(predicted_value, actual_value))
        
        else:
            anomaly_scores[i] = 0
            
    return anomaly_scores

def find_best_threshold(true_labels, anomaly_scores):
    """
    遍历anomaly_scores数组的1到99百分位数，找出F1分数最大的阈值
    
    Args:
        true_labels: 真实标签
        anomaly_scores: 异常分数
        
    Returns:
        tuple: (最佳阈值, 最佳F1分数, 所有阈值和对应的F1分数列表)
    """
    best_threshold = 0
    best_f1 = 0
    threshold_f1_list = []
    
    # 遍历1到99百分位数
    for i in range(10, 100):
        threshold = np.percentile(anomaly_scores, i)
        predicted_labels = (anomaly_scores > threshold).astype(int)
        
        # 计算F1分数
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        threshold_f1_list.append((threshold, f1, i))
        
        # 更新最佳阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, threshold_f1_list

from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries, max_d=2):
    """
    使用ADF检验判断时间序列的平稳性，并确定最佳差分次数
    
    Args:
        timeseries: 时间序列数据
        max_d: 最大差分次数
        
    Returns:
        int: 推荐的差分次数d
    """
    def adf_test(data, diff_order):
        """执行ADF检验"""
        if diff_order > 0:
            # 对数据进行差分
            diff_data = np.diff(data, n=diff_order)
        else:
            diff_data = data
            
        # 执行ADF检验
        result = adfuller(diff_data)
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        # 通常使用5%显著性水平
        is_stationary = p_value <= 0.05
        
        return adf_statistic, p_value, is_stationary
    
    print("ADF检验结果:")
    print("差分次数\tADF统计量\tP值\t\t是否平稳")
    print("-" * 50)
    
    for d in range(max_d + 1):
        try:
            adf_stat, p_val, is_stationary = adf_test(timeseries, d)
            status = "是" if is_stationary else "否"
            print(f"{d}\t\t{adf_stat:.4f}\t\t{p_val:.6f}\t\t{status}")
            
            # 如果序列平稳，返回当前差分次数
            if is_stationary and d > 0:
                print(f"\n推荐差分次数 d = {d}")
                return d
            elif is_stationary and d == 0:
                print(f"\n序列已经平稳，推荐差分次数 d = 0")
                return 0
                
        except Exception as e:
            print(f"d={d}: 计算出错 - {e}")
            continue
    
    # 如果所有差分次数都不能使序列平稳，选择p值最小的
    print(f"\n无法使序列完全平稳，建议使用 d = {max_d}")
    return max_d

def find_optimal_arima_params(data, max_p=3, max_q=3, max_d=50):
    """
    使用BIC准则寻找最佳ARIMA(p,d,q)参数，包括自动确定d值
    
    Args:
        data: 时间序列数据
        max_p: p参数的最大值
        max_q: q参数的最大值
        max_d: d参数的最大值
        
    Returns:
        tuple: (最佳(p,d,q)参数, BIC值字典)
    """
    # 首先确定最佳差分次数
    best_d = check_stationarity(data, max_d)
    
    best_bic = np.inf
    best_order = (0, best_d, 0)
    bic_dict = {}
    
    print(f"\n使用最佳差分次数 d = {best_d} 进行参数搜索")
    print(f"Searching for optimal ARIMA parameters with max_p={max_p}, max_q={max_q}")
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                order = (p, best_d, q)
                model = ARIMA(data, order=order)
                fitted_model = model.fit()
                bic = fitted_model.bic
                
                bic_dict[order] = bic
                
                if bic < best_bic:
                    best_bic = bic
                    best_order = order
                    
                print(f"ARIMA{order}: BIC={bic:.4f}")
                
            except Exception as e:
                bic_dict[(p, best_d, q)] = np.inf
                print(f"ARIMA{(p, best_d, q)}: Failed to fit - {str(e)[:50]}...")
                continue
    
    print(f"Best ARIMA order: {best_order} with BIC={best_bic:.4f}")
    return best_order, bic_dict

def process_kpi_test_files(input_dir="input"):
    """处理input/kpi/kpi_test目录下的所有数据集文件"""
    kpi_test_dir = os.path.join(input_dir, "kpi", "kpi_test")
    
    # 检查目录是否存在
    if not os.path.exists(kpi_test_dir):
        print(f"目录 {kpi_test_dir} 不存在")
        return {}
    
    # 获取所有csv文件
    csv_files = glob.glob(os.path.join(kpi_test_dir, "*.csv"))
    
    if not csv_files:
        print(f"在 {kpi_test_dir} 目录下未找到CSV文件")
        return {}
    
    print(f"找到 {len(csv_files)} 个数据集文件:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    results = {}
    
    # 处理每个数据集文件
    for file_path in csv_files:
        try:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"\n{'='*50}")
            print(f"Processing KPI test file with ARIMA: {dataset_name}")
            print(f"{'='*50}")
            
            # 加载数据
            data = pd.read_csv(file_path)
            
            # 按照data_utils的方式处理数据
            # 假设数据文件有'value'列和'label'列
            if 'value' not in data.columns or 'label' not in data.columns:
                print(f"文件 {file_path} 缺少'value'或'label'列，跳过处理")
                continue
                
            # 分离正常数据和测试数据（按照data_utils的方式）
            # 这里假设所有数据都是测试数据
            test_data_values = data['value'].values
            labels = data['label'].values
            
            print(f"Test data shape: {test_data_values.shape}")
            print(f"Labels length: {len(labels)}")
            
            # 数据标准化（如果需要）
            # 这里我们创建一个临时的DataFrame来使用normalize_data函数
            test_data_df = pd.DataFrame({'value': test_data_values})
            
            # 由于没有明确的normal数据，我们使用测试数据的一部分作为"normal"参考
            # 取前10%的数据作为"normal"数据进行标准化
            split_idx = max(100, len(test_data_values) // 10)
            normal_ref = test_data_df.iloc[:split_idx]
            attack_data = test_data_df  # 全部数据用于测试
            
            # 标准化数据
            normal_scaled, attack_scaled, scaler = normalize_data(normal_ref, attack_data)
            
            # 提取测试数据
            test_data = attack_scaled.values[:, 0] if attack_scaled.shape[1] > 0 else attack_scaled.values.flatten()
            
            # 确保标签长度与测试数据一致
            if len(labels) > len(test_data):
                labels = labels[:len(test_data)]
            elif len(labels) < len(test_data):
                test_data = test_data[:len(labels)]
            
            print(f"Processing data shape: {test_data.shape}")
            print(f"Labels length: {len(labels)}")
            
            # 使用部分数据进行参数搜索以节省时间
            search_data = test_data[-min(1000, len(test_data)):] if len(test_data) > 1000 else test_data
            print("Finding optimal ARIMA parameters using BIC...")
            best_order, bic_results = find_optimal_arima_params(search_data, max_p=3, max_q=3)
            print(f"Using ARIMA order: {best_order}")
            
            # 计算测试数据的异常分数
            print("Computing anomaly scores...")
            start_time = time.time()
            anomaly_scores = compute_anomaly_scores_arima(test_data, window_size=50, order=best_order)
            elapsed_time = time.time() - start_time
            print(f"计算用时: {elapsed_time:.2f}秒")
            
            # 确保分数和标签数量一致
            true_labels = np.array(labels)
            if len(anomaly_scores) > len(true_labels):
                anomaly_scores = anomaly_scores[:len(true_labels)]
            elif len(anomaly_scores) < len(true_labels):
                true_labels = true_labels[:len(anomaly_scores)]
            
            print(f"对齐后: 异常分数 {len(anomaly_scores)}, 标签 {len(true_labels)}")
            
            print("正在查找最佳阈值...")
            best_threshold, best_f1, threshold_f1_list = find_best_threshold(true_labels, anomaly_scores)
            
            # 使用最佳阈值进行预测
            predicted_labels = (anomaly_scores > best_threshold).astype(int)
            
            print(f"\n使用最佳阈值 {best_threshold:.6f}:")
            print(f"预测异常点数量: {np.sum(predicted_labels)}")
            print(f"实际异常点数量: {np.sum(true_labels)}")
            
            # 评估结果
            print("\n分类报告:")
            print(classification_report(true_labels, predicted_labels))
            
            # 计算详细指标
            metrics = calculate_metrics(true_labels, predicted_labels)
            metrics['cost_time'] = elapsed_time
            
            print(f"\n详细指标:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"False Negative Rate (FNR): {metrics['fnr']:.4f}")
            print(f"False Positive Rate (FPR): {metrics['fpr']:.4f}")
            print(f"True Positives: {metrics['tp']}")
            print(f"True Negatives: {metrics['tn']}")
            print(f"False Positives: {metrics['fp']}")
            print(f"False Negatives: {metrics['fn']}")
            
            # 保存指标到单独的文件
            metrics_df = pd.DataFrame([metrics]).T.reset_index()
            metrics_df.columns = ['metric', 'value']
            metrics_filename = f'{dataset_name}_arima_anomaly_detection_metrics.csv'
            metrics_df.to_csv(metrics_filename, index=False)
            print(f"指标已保存到 {metrics_filename}")
            
            results[dataset_name] = metrics
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            results[dataset_name] = None
    
    return results

def main():

    results = {}

    # 处理input/kpi/kpi_test目录下的所有文件
    kpi_test_results = process_kpi_test_files()
    results.update(kpi_test_results)
    
    # 打印所有数据集的结果汇总
    print(f"\n{'='*50}")
    print("SUMMARY OF ARIMA RESULTS")
    print(f"{'='*50}")
    for dataset, result in results.items():
        if result:
            print(f"{dataset}:")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  F1-Score:  {result['f1_score']:.4f}")
            print(f"  Accuracy:  {result['accuracy']:.4f}")
        else:
            print(f"{dataset}: Failed to process")

if __name__ == "__main__":
    main()