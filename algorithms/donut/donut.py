import os
from sklearn import metrics
import tensorflow as tf
import keras as K
import glob
import pandas as pd
import time
import numpy as np
from evaluation_metric import range_lift_with_delay

if not hasattr(np, "int"):
    np.int = np.int64
if not hasattr(np, "float"):
    np.float = np.float64

tf.compat.v1.disable_v2_behavior()

if not hasattr(tf, "GraphKeys"):
    tf.GraphKeys = tf.compat.v1.GraphKeys
if not hasattr(tf, "log"):
    tf.log = tf.math.log
if not hasattr(tf, "log1p"):
    tf.log1p = tf.math.log1p
if not hasattr(tf.train, "AdamOptimizer"):
    tf.train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer
if not hasattr(tf, "reset_default_graph"):
    tf.reset_default_graph = tf.compat.v1.reset_default_graph
if not hasattr(tf, "variable_scope"):
    tf.variable_scope = tf.compat.v1.variable_scope
if not hasattr(tf, "VariableScope"):
    tf.VariableScope = tf.compat.v1.VariableScope
if not hasattr(tf, "layers"):
    tf.layers = tf.compat.v1.layers
if not hasattr(tf, "get_variable"):
    tf.get_variable = tf.compat.v1.get_variable
if not hasattr(tf, "placeholder"):
    tf.placeholder = tf.compat.v1.placeholder
if not hasattr(tf, "get_default_graph"):
    tf.get_default_graph = tf.compat.v1.get_default_graph
if not hasattr(tf, "random_normal"):
    tf.random_normal = tf.compat.v1.random_normal

from donut import complete_timestamp, standardize_kpi
from donut import Donut
from donut import DonutTrainer, DonutPredictor
from tfsnippet.modules import Sequential

ROUND = 5
save_dir = "./model_save"
os.makedirs(save_dir, exist_ok=True)

results_summary = []

train_files = sorted(glob.glob("../dataset/kpi/train/*.csv"), key=os.path.getsize)

if not train_files:
    print("train_files not found")
    exit()

for train_file in train_files:
    kpi_id = os.path.basename(train_file).split(".")[0]
    print(f"Processing kpi: {kpi_id}")
    round_results = []

    for i in range(ROUND):
        print(f"Round {i+1}")
        try:
            tf.reset_default_graph()

            train_data = np.loadtxt(
                train_file, delimiter=",", skiprows=1, unpack=True, usecols=[0, 1]
            )
            train_timestamp, train_values = train_data[0], train_data[1]
            train_labels = np.zeros_like(train_values, dtype=np.int32)

            train_timestamp, train_missing, (train_values, train_labels) = (
                complete_timestamp(train_timestamp, (train_values, train_labels))
            )
            test_file = train_file.replace("train", "test")
            test_data = np.loadtxt(
                test_file, delimiter=",", skiprows=1, unpack=True, usecols=[0, 1, 2]
            )
            test_timestamp, test_values, test_labels = (
                test_data[0],
                test_data[1],
                test_data[2],
            )

            test_timestamp, test_missing, (test_values, test_labels) = (
                complete_timestamp(test_timestamp, (test_values, test_labels))
            )

            train_values, mean, std = standardize_kpi(
                train_values, excludes=np.logical_or(train_labels, train_missing)
            )
            test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

            with tf.variable_scope("model") as model_vs:
                model = Donut(
                    h_for_p_x=Sequential(
                        [
                            K.layers.Dense(
                                100,
                                kernel_regularizer=K.regularizers.l2(0.001),
                                activation=tf.nn.relu,
                            ),
                            K.layers.Dense(
                                100,
                                kernel_regularizer=K.regularizers.l2(0.001),
                                activation=tf.nn.relu,
                            ),
                        ]
                    ),
                    h_for_q_z=Sequential(
                        [
                            K.layers.Dense(
                                100,
                                kernel_regularizer=K.regularizers.l2(0.001),
                                activation=tf.nn.relu,
                            ),
                            K.layers.Dense(
                                100,
                                kernel_regularizer=K.regularizers.l2(0.001),
                                activation=tf.nn.relu,
                            ),
                        ]
                    ),
                    x_dims=128,
                    z_dims=5,
                )

            trainer = DonutTrainer(model=model, model_vs=model_vs, max_epoch=60)
            predictor = DonutPredictor(model)

            with tf.Session() as sess:
                train_start_time = time.time()
                trainer.fit(train_values, train_labels, train_missing, mean, std)
                train_end_time = time.time()
                train_time = train_end_time - train_start_time

                test_start_time = time.time()
                test_score = predictor.get_score(test_values, test_missing)
                test_end_time = time.time()
                test_time = test_end_time - test_start_time

                aligned_test_labels = test_labels[model.x_dims - 1 :]

                if len(aligned_test_labels) != len(test_score):
                    min_len = min(len(aligned_test_labels), len(test_score))
                    aligned_test_labels = aligned_test_labels[:min_len]
                    test_score = test_score[:min_len]
                
                test_score = range_lift_with_delay(test_score, aligned_test_labels, delay=7)

                best_accuracy = 0
                best_f1 = 0
                best_precision = 0
                best_recall = 0
                best_fnr = 0
                best_fpr = 0

                for percentile in range(1, 100):
                    threshold = np.percentile(test_score, percentile)
                    pred_labels = (test_score < threshold).astype(int)

                    tn, fp, fn, tp = metrics.confusion_matrix(
                        aligned_test_labels, pred_labels
                    ).ravel()
                    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
                    accuracy = (
                        (tp + tn) / (tp + tn + fp + fn)
                        if (tp + tn + fp + fn) > 0
                        else 0
                    )
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    if f1 > best_f1:
                        best_f1 = f1
                        best_accuracy = accuracy
                        best_precision = precision
                        best_recall = recall
                        best_fnr = fnr
                        best_fpr = fpr

                result = {
                    "round": i + 1,
                    "kpi_id": kpi_id,
                    "accuracy": best_accuracy,
                    "precision": best_precision,
                    "recall": best_recall,
                    "f1_score": best_f1,
                    "fnr": best_fnr,
                    "fpr": best_fpr,
                    "train_time": train_time,
                    "test_time": test_time,
                }
                results_summary.append(result)

        except Exception as e:
            print(f"ERROR processing {train_file}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    data = [r for r in results_summary if r['kpi_id'] == kpi_id]
    if data:
        df = pd.DataFrame(data)
        print(f" round           kpi_id  precision  recall  f1_score  train_time  test_time")
        for _, row in df.iterrows():
            print(f"     {row['round']} {row['kpi_id']}          {row['precision']:.0f}       {row['recall']:.0f}         {row['f1_score']:.0f}           {row['train_time']:.0f}          {row['test_time']:.0f}")

results_df = pd.DataFrame(results_summary)

if len(results_df) > 0:
    avg_results = results_df.groupby('kpi_id').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean', 
        'f1_score': 'mean',
        'fnr': 'mean',
        'fpr': 'mean',
        'train_time': 'mean',
        'test_time': 'mean'
    }).round(4)

    print(avg_results.to_string())
    
    os.makedirs("../results", exist_ok=True)
    avg_results.to_csv("../results/donut.csv")