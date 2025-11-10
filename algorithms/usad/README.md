support for kpi MSL SMAP SMD PSM swat datasets
数据集放在input文件夹下
运行方式 python main.py
依次运行所有数据集
python main.py
python realtime_detector.py --data SMAP;
python realtime_detector.py --data PSM;
python realtime_detector.py --data SWAT;
python realtime_detector.py --data SMD;
# USAD - UnSupervised Anomaly Detection on multivariate time series

Scripts and utility programs for implementing the USAD architecture.

Implementation by: Francesco Galati.

Additional contributions: Julien Audibert, Maria A. Zuluaga.

## How to cite

If you use this software, please cite the following paper as appropriate:

    Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
    USAD : UnSupervised Anomaly Detection on multivariate time series.
    Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 23-27, 2020

## Requirements
 * PyTorch 1.6.0
 * CUDA 10.1 (to allow use of GPU, not compulsory)

## Running the Software

All the python classes and functions strictly needed to implement the USAD architecture can be found in `usad.py`.
An example of an application deployed with the [SWaT dataset] is included in `USAD.ipynb`.

## Copyright and licensing

Copyright 2020 Eurecom.

This software is released under the BSD-3 license. Please see the license file_ for details.

## Publication

Audibert et al. [USAD : UnSupervised Anomaly Detection on multivariate time series]. 2020

[SWaT dataset]: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat
[USAD : UnSupervised Anomaly Detection on multivariate time series]: https://dl.acm.org/doi/pdf/10.1145/3394486.3403392
