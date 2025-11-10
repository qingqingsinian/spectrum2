import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from utils import read_csv, average_filter, get_train_files


def _normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 3] range"""
    min_value = float(np.min(data))
    max_value = float(np.max(data))
    data_range = max_value - min_value
    # Avoid division by zero
    if data_range < 1e-10:
        return np.zeros_like(data)  # type: ignore
    return 3 * (data - min_value) / data_range


class DataGenerator:
    """Generates training data with synthetic anomalies for time series anomaly detection"""

    def __init__(self, window_size: int, step_size: int, max_anomalies: int):
        """
        Args:
            window_size: Size of the sliding window
            step_size: Step size for moving the window
            max_anomalies: Maximum number of anomalies to inject per window
        """
        self.control = 0.0
        self.window_size = window_size
        self.step_size = step_size
        self.max_anomalies = max(max_anomalies, 1)  # Ensure at least 1 anomaly

    def _inject_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inject synthetic anomalies into a window of data"""
        # Create anomaly labels
        labels = np.zeros(self.window_size, dtype=np.int64)

        # Determine number of anomalies to inject (1 to max_anomalies)
        num_anomalies = int(np.random.randint(1, self.max_anomalies + 1))

        # Select positions for anomalies
        anomaly_positions = np.random.choice(self.window_size, size=num_anomalies, replace=False)

        # Control mechanism to ensure periodic anomalies at specific position
        control_position = self.window_size - 6

        # Ensure control_position is valid
        if control_position < 0 or control_position >= self.window_size:
            control_position = max(0, min(self.window_size - 1, self.window_size // 2))

        # Control logic for periodic anomalies
        if control_position not in anomaly_positions:
            self.control += np.random.uniform(0.5, 1.5)
        else:
            self.control = 0

        if self.control > 100:
            # Ensure control_position is in anomaly positions
            if control_position not in anomaly_positions:
                if num_anomalies < self.window_size:
                    anomaly_positions = np.append(anomaly_positions, control_position)
                else:
                    anomaly_positions[0] = control_position # type: ignore
            self.control = 0

        # Calculate statistics for anomaly injection
        mean_val = np.mean(data)
        local_means = average_filter(data)
        variance = np.var(data)

        # Generate perturbation factors
        perturbations = ((local_means[anomaly_positions] + mean_val) * np.random.randn() * min(1 + float(variance), 10))

        # Apply perturbations
        data[anomaly_positions] += perturbations
        labels[anomaly_positions] = 1

        return data, labels

    def generate_train_data(self, time_series: np.ndarray, back_offset: int = 0) -> List[Tuple[List[float], List[int]]]:
        """
        Generate training data with injected anomalies
        
        Args:
            time_series: Input time series data
            back_offset: Offset to avoid end of series
            
        Returns:
            List of (data_window, label_window) tuples
        """
        # Validate input
        if len(time_series) < self.window_size:
            return []
        # Calculate safe offset
        offset = min(5, max(0, back_offset))

        # Pre-calculate window positions
        start_positions = range(self.window_size, len(time_series) - offset, self.step_size)

        results = []

        for start in start_positions:
            # Calculate window boundaries
            head = max(0, start - self.window_size)
            tail = min(len(time_series) - offset, start)

            # Extract window data
            window_data = time_series[head:tail].copy()

            # Ensure window is the correct size
            if len(window_data) != self.window_size:
                # Pad with zeros if window is too small
                if len(window_data) < self.window_size:
                    window_data = np.pad(window_data, (0, self.window_size - len(window_data)), 'constant')
                # Truncate if window is too large (shouldn't happen)
                else:
                    window_data = window_data[:self.window_size]

            # Normalize window data
            normalized_data = _normalize(window_data)

            # Inject anomalies and get labels
            data_with_anomalies, labels = self._inject_anomalies(normalized_data)

            results.append((data_with_anomalies.tolist(), labels.tolist()))

        return results


def main():
    """Main function for generating training data"""
    parser = argparse.ArgumentParser(description='sr-cnn Data Generator')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset directory')
    parser.add_argument('--window', type=int, default=32, help='Window size for sliding window')
    parser.add_argument('--step', type=int, default=64, help='Step size for sliding window movement')
    parser.add_argument('--seed', type=int, default=54321, help='Random seed for reproducibility')
    parser.add_argument('--num', type=int, default=10, help='Maximum number of anomalies per window')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: current directory/train/<dataset>)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path.cwd() / 'train' / args.dataset

    # Ensure output directory exists and is empty
    if output_dir.exists():
        # Remove existing files
        for file in output_dir.glob('*'):
            if file.is_file():
                file.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset files
    try:
        files = get_train_files(args.dataset)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Found {len(files)} files in dataset '{args.dataset}'")

    # Initialize data generator
    generator = DataGenerator(args.window, args.step, args.num)

    # Process files with progress bar
    processed_files = 0
    skipped_files = 0

    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Read CSV file
            _, values = read_csv(file_path)

            # Skip files with insufficient data
            if len(values) < args.window:
                skipped_files += 1
                continue
            # Generate training data
            train_data = generator.generate_train_data(values)
            # Skip if no data generated
            if not train_data:
                skipped_files += 1
                continue

            # Create output filename
            kpi_id = Path(file_path).stem
            output_file = output_dir / f"{kpi_id}_{args.window}_train.json"

            # Save data
            with output_file.open('w') as f:
                json.dump(train_data, f)

            processed_files += 1
        except Exception as e:
            print(f"\nError processing file {file_path}: {str(e)}")
            skipped_files += 1
            import traceback
            traceback.print_exc()

    print(f"\nProcessing complete:")
    print(f"  Files processed: {processed_files}")
    print(f"  Files skipped: {skipped_files}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
