import os
from srcnn.competition_metric import get_variance, evaluate_for_all_series
import time
import json
import argparse
from msanomalydetector.spectral_residual import SpectralResidual
from srcnn.utils import *
import csv


def auto():
    path_auto = os.getcwd() + '/auto.json'
    with open(path_auto, 'r+') as f:
        store = json.load(f)
    window = store['window']
    epoch = store['epoch']
    return window, epoch


def getfid(path):
    return path.split('/')[-1]


def get_path(data_source):
    if data_source == 'kpi':
        dir_ = root + '/Test/'
        trainfiles = [dir_ + _ for _ in os.listdir(dir_)]
        files = trainfiles
    else:
        dir_ = root + '/' + data_source + '/'
        files = [dir_ + _ for _ in os.listdir(dir_)]
    return files


def get_score(data_source, files, thres, option):
    total_time = 0
    results = []
    savedscore = []
    prediction_times = {}  # 记录每个文件的预测时间
    for f in files:
        print('reading', f)
        if data_source == 'test' or data_source == 'test_kpi':
            in_timestamp, in_value, in_label = read_csv_kpi(f)
        else:
            tmp_data = read_pkl(f)
            in_timestamp, in_value, in_label = tmp_data['timestamp'], tmp_data['value'], tmp_data['label']
        length = len(in_timestamp)
        if model == 'sr_cnn' and len(in_value) < window:
            print("length is shorter than win_size", len(in_value), window)
            continue
        time_start = time.time()
        timestamp, label, pre, scores = models[model](in_timestamp, in_value, in_label, window, net, option, thres)
        time_end = time.time()
        file_prediction_time = time_end - time_start
        total_time += file_prediction_time
        prediction_times[f] = file_prediction_time  # 存储该文件的预测时间
        results.append([timestamp, label, pre, f])
        savedscore.append([label, scores, f, timestamp])
    return total_time, results, savedscore, prediction_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRCNN')
    parser.add_argument('--data', type=str, required=True, help='location of the data file')
    parser.add_argument('--window', type=int, default=128, help='window size')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--model_path', type=str, default='snapshot', help='model path')
    parser.add_argument('--delay', type=int, default=3, help='delay')
    parser.add_argument('--thres', type=float, default=0.95, help='initial threshold of SR')
    parser.add_argument('--auto', type=bool, default=False, help='Automatic filling parameters')
    parser.add_argument('--model', type=str, default='sr_cnn', help='model')
    parser.add_argument('--missing_option', type=str, default='anomaly',
                        help='missing data option, anomaly means treat missing data as anomaly')

    args = parser.parse_args()
    if args.auto:
        window, epoch = auto()
    else:
        window = args.window
        epoch = args.epoch
    data_source = args.data
    delay = args.delay
    model = args.model
    root = os.getcwd()
    print(data_source, window, epoch)
    models = {
        'sr_cnn': sr_cnn_eval,
    }

    model_path = root + '/' + args.model_path + '/totalsrcnn_retry' + str(epoch) + '_' + str(window) + '.bin'
    srcnn_model = Anomaly(window)
    net = load_model(srcnn_model, model_path).cuda()
    files = get_path(data_source)
    total_time, results, savedscore, prediction_times = get_score(data_source, files, args.thres, args.missing_option)
    print('\n***********************************************')
    print('data source:', data_source, '     model:', model)
    print('-------------------------------')
    total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(results, delay)
    with open(data_source + '_saved_scores.json', 'w') as f:
        json.dump(savedscore, f)
    print('time used for making predictions:', total_time, 'seconds')
    score_csv_filename = data_source + '_scores.csv'
    with open(score_csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['file', 'timestamp', 'label', 'score', 'prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 为每个文件写入分数数据
        for flabel, scores, filename, ftimestamp in savedscore:
            # 计算最佳阈值用于生成预测结果
            best_fscore = 0.0
            best_threshold = 0.0
            # 简单搜索最佳阈值
            for i in range(98):
                threshold = 0.01 + i * 0.01
                preds = [1 if score > threshold else 0 for score in scores]
                try:
                    fscore, _, _, _, _, _, _ = evaluate_for_all_series([[ftimestamp, flabel, preds, filename]], delay, prt=False)
                    if fscore > best_fscore:
                        best_fscore = fscore
                        best_threshold = threshold
                except:
                    pass
            
            # 使用最佳阈值生成预测结果
            predictions = [1 if score > best_threshold else 0 for score in scores]
            
            # 写入每条记录
            for ts, lbl, scr, pred in zip(ftimestamp, flabel, scores, predictions):
                writer.writerow({
                    'file': os.path.basename(filename),
                    'timestamp': ts,
                    'label': lbl,
                    'score': scr,
                    'prediction': pred
                })
    
    print(f'Anomaly scores saved to {score_csv_filename}')
    # 创建CSV文件并写入结果
    csv_filename = data_source + '_evaluation_results1.csv'
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['type', 'filename', 'f1_score', 'precision', 'recall', 'TP', 'FP', 'TN', 'FN', 'best_threshold', 'prediction_time', 'training_time', 'delay']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 写入总体结果
        writer.writerow({
            'type': 'overall',
            'filename': 'all_files',
            'f1_score': total_fscore,
            'precision': pre,
            'recall': rec,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'best_threshold': 'N/A',
            'prediction_time': total_time,
            'training_time': 'N/A',  # 总体结果中不适用
            'delay': delay
        })

        # 为每个文件计算最佳阈值和相应的指标
        for f, (flabel, cnnscores, _, ftimestamp) in zip(files, savedscore):
            best = 0.0
            bestthre = 0.0
            best_pre = 0.0
            best_rec = 0.0
            
            # 对每个文件搜索最佳阈值
            for i in range(98):
                threshold = 0.01 + i * 0.01
                pre = [1 if item > threshold else 0 for item in cnnscores]
                # 构造单个文件的结果
                file_result = [[ftimestamp, flabel, pre, f]]
                try:
                    fscore, precision, recall, tp, fp, tn, fn = evaluate_for_all_series(file_result, delay, prt=False)
                    if fscore > best:
                        best = fscore
                        bestthre = threshold
                        best_pre = precision
                        best_rec = recall
                except Exception as e:
                    pass  # 忽略计算错误
            
            # 使用最佳阈值生成最终预测结果
            final_pre = [1 if item > bestthre else 0 for item in cnnscores]
            file_result_final = [[ftimestamp, flabel, final_pre, f]]
            
            try:
                final_fscore, final_precision, final_recall, tp, fp, tn, fn = evaluate_for_all_series(file_result_final, delay, prt=False)
            except:
                final_fscore, final_precision, final_recall, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0, 0
            
            # 获取该文件的预测时间
            file_prediction_time = prediction_times.get(f, 0)
            
            # 写入单个文件的结果
            writer.writerow({
                'type': 'per_file',
                'filename': os.path.basename(f),
                'f1_score': final_fscore,
                'precision': final_precision,
                'recall': final_recall,
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'best_threshold': bestthre,
                'prediction_time': file_prediction_time,
                'training_time': 'N/A',  # 单个文件不涉及训练时间
                'delay': delay
            })
            
            print(f"File: {os.path.basename(f)} - Best F1: {final_fscore:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, Best Threshold: {bestthre:.2f}, Prediction Time: {file_prediction_time:.4f}s")

    print(f'Results saved to {csv_filename}')

    best = 0.
    bestthre = 0.
    print('delay :', delay)
    if data_source == 'yahoo':
        sru = {}
        rf = open(data_source + 'sr3.json', 'r')
        srres = json.load(rf)
        for (srtime, srl, srpre, srf) in srres:
            sru[getfid(srf)] = [srtime, srl, srpre]
        for i in range(98):
            newresults = []
            threshold = 0.01 + i * 0.01
            for f, (srtt, srlt, srpret, srft), (flabel, cnnscores, cnnf, cnnt) in zip(files, srres, savedscore):
                fid = getfid(cnnf)
                srtime = sru[fid][0]
                srl = sru[fid][1]
                srpre = sru[fid][2]
                srtime = [(srtime[0] - 3600 * (64 - j)) for j in range(64)] + srtime
                srl = [0] * 64 + srl
                srpre = [0] * 64 + srpre
                print(len(srl), len(flabel), '!!')
                assert (len(srl) == len(flabel))
                pre = [1 if item > threshold else 0 for item in cnnscores]
                newresults.append([srtime, srpre, pre, f])
            total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(newresults, delay, prt=False)
            if total_fscore > best:
                best = total_fscore
                bestthre = threshold
        results = []
        threshold = bestthre
        print('guided threshold :', threshold)
        for f, (flabel, cnnscores, _, ftimestamp) in zip(files, savedscore):
            pre = [1 if item > threshold else 0 for item in cnnscores]
            results.append([ftimestamp, flabel, pre, f])
        print('score\n')
        total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(results, delay)
        print(total_fscore)
    best = 0.
    for i in range(98):
        newresults = []
        threshold = 0.01 + i * 0.01
        for f, (flabel, cnnscores, _, ftimestamp) in zip(files, savedscore):
            pre = [1 if item > threshold else 0 for item in cnnscores]
            newresults.append([ftimestamp, flabel, pre, f])
        total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(newresults, delay, prt=False)
        if total_fscore > best:
            best = total_fscore
            bestthre = threshold
            print('tem best', best, threshold)
    threshold = bestthre
    print('best overall threshold :', threshold, 'best score :', best)