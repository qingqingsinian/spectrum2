import glob
import os
import argparse
import torch
import time
import pandas as pd
import numpy as np
from evaluation_metric import range_lift_with_delay
from sklearn import metrics
from source import IntroVAE
from source import KPISeries


def arg_parse() -> argparse.Namespace:
    """Í
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description="SR-VAE Anomaly Detection")
    parser.add_argument(
        "--data",
        dest="data_path",
        type=str,
        default="../dataset/kpi",
        help="The dataset path",
    )
    parser.add_argument(
        "--max-epoch", dest="epochs", type=int, default=60, help="The random seed"
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=256,
        help="The number of the batch size for training",
    )
    parser.add_argument(
        "--window-size",
        dest="window_size",
        type=int,
        default=128,
        help="The size the sliding window",
    )
    parser.add_argument(
        "--latent-size",
        dest="latent_size",
        type=int,
        default=3,
        help="The dimension of the latent variables",
    )
    parser.add_argument(
        "--results",
        dest="filename",
        type=str,
        default="../results/sr-vae.csv",
        help="The random seed",
    )
    parser.add_argument(
        "--positive-margin",
        dest="margin",
        type=float,
        default=15,
        help="The positive margin",
    )
    parser.add_argument("--delay", dest="delay", type=float, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    print("-" * 150)
    for arg in vars(args):
        print("{:15s}:{}".format(arg, getattr(args, arg)))
    print("-" * 150)

    # set random seed
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # read args
    epochs = int(args.epochs)
    window_size = int(args.window_size)
    latent_size = int(args.latent_size)
    batch_size = int(args.batch_size)
    margin = float(args.margin)

    train_files = sorted(glob.glob("../dataset/kpi/sr-vae/*.csv"))
    test_files = sorted(glob.glob("../dataset/kpi/test/*.csv"))

    for train_file in train_files:
        filename = os.path.basename(train_file)
        test_file = os.path.join("../dataset/kpi/test", filename)

        if not os.path.exists(test_file):
            print(f"Warning: Test file not found for {filename}, skipping...")
            continue

        print(f"\n{'='*150}")
        print(f"Processing: {filename}")
        print(f"{'='*150}")

        train_df = pd.read_csv(train_file, header=0, index_col=None)
        train_kpi = KPISeries(
            value=train_df.value, timestamp=train_df.timestamp, label=train_df.pred
        )

        test_df = pd.read_csv(test_file, header=0, index_col=None)
        test_kpi = KPISeries(
            value=test_df.value,
            timestamp=test_df.timestamp,
            label=test_df.label,
            truth=test_df.label,
        )

        train_kpi, train_kpi_mean, train_kpi_std = train_kpi.normalize(
            return_statistic=True
        )

        test_kpi = test_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)

        kpi_id = filename.split(".")[0]

        # create model
        model = IntroVAE(
            cuda=torch.cuda.is_available(),
            max_epoch=epochs,
            latent_dims=latent_size,
            window_size=window_size,
            batch_size=batch_size,
            margin=margin,
        )

        print(f"\nTraining on {kpi_id}...")
        start = time.time()
        model.fit(train_kpi.label_sampling(1.0))
        end = time.time()
        train_time = end - start

        print(f"\nTesting on {kpi_id}...")
        start = time.time()
        y_prob_test = model.predict(test_kpi.label_sampling(0.0))
        end = time.time()
        test_time = end - start

        y_prob_test = range_lift_with_delay(y_prob_test, test_kpi.truth, delay=7)
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            test_kpi.truth, y_prob_test
        )
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)

        valid_f1 = f1_scores[np.isfinite(f1_scores)]
        if len(valid_f1) > 0:
            best_f1_idx = np.argmax(valid_f1)
            valid_indices = np.where(np.isfinite(f1_scores))[0]
            best_idx = valid_indices[best_f1_idx]

            best_threshold = (
                thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            )
            best_precision = precisions[best_idx]
            best_recall = recalls[best_idx]
            best_f1 = valid_f1[best_f1_idx]

            y_pred_binary = (y_prob_test >= best_threshold).astype(int)

            tn, fp, fn, tp = metrics.confusion_matrix(
                test_kpi.truth, y_pred_binary
            ).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            continue

        cont_list = [
            {
                "kpi-id": kpi_id,
                "Epoch": epochs,
                "Window-size": window_size,
                "Accuracy": accuracy,
                "Precision": best_precision,
                "Recall": best_recall,
                "F1-score": best_f1,
                "FNR": fnr,
                "FPR": fpr,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Train-Time(s)": train_time,
                "Test-Time(s)": test_time,
            }
        ]

        result_df = pd.DataFrame(cont_list)
        filepath = args.filename
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(filepath):
            result_df.to_csv(filepath, index=False)
        else:
            result_df.to_csv(filepath, mode="a", header=False, index=False)

    print(f"\n{'='*150}")
    print(f"All files processed! Results saved to: {args.filename}")
    print(f"{'='*150}")
