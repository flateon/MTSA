# Modern Time Series Analysis

MTSA (Modern Time Series Analysis) is a library dedicated to the field of time series forecasting. Our primary objective
is to provide a comprehensive collection of both classical and deep learning-based algorithms for tackling time series
forecasting tasks.

We will gradually enhance and expand our library as the TSA (Time Series Analysis) course progresses.

[TSA home page](https://www.lamda.nju.edu.cn/yehj/TSA2023/)

## Getting Started

To get started with MTSA, follow these steps:

### Prerequisites

Git clone our repository, creating a conda environment and activate it via the following command

```bash
cd MTSA
conda env create -f environment.yml
conda activate MTSA-torch
```

### Download Datasets

Download the datasets from [nju box](https://box.nju.edu.cn/d/b33a9f73813048b8b00f/) and organize them as follows:

```
MTSA
└── dataset
    ├── electricity
    │   └── electricity.csv
    ├── ETT
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   ├── ETTm1.csv
    │   └── ETTm2.csv
    ├── exchange_rate
    │   └── exchange_rate.csv
    ├── illness
    │   └── national_illness.csv
    ├── m4
    │   ├── Daily-test.csv
    │   ├── Daily-train.csv
    │   ├── Hourly-test.csv
    │   ├── Hourly-train.csv
    │   ├── M4-info.csv
    │   ├── Monthly-test.csv
    │   ├── Monthly-train.csv
    │   ├── Quarterly-test.csv
    │   ├── Quarterly-train.csv
    │   ├── submission-Naive2.csv
    │   ├── test.npz
    │   ├── training.npz
    │   ├── Weekly-test.csv
    │   ├── Weekly-train.csv
    │   ├── Yearly-test.csv
    │   └── Yearly-train.csv
    ├── traffic
    │   └── traffic.csv
    └── weather
        └── weather.csv

```

## Usage examples

### Main Results

To reproduce the reported results, run the following command:

```bash
python benchmark_knn.py
python benchmark_decomposition.py
python benchmark_reimplement.py
```

The reported results will be store in *./results/test_\*.csv*

### Unittest and coverage report

To test our implementation, run the unittest with the following command:

```bash
coverage run -m unittest; coverage report -m
```
Here is testing result and the coverage report:
```
Ran 42 tests in 69.736s

OK
Name                             Stmts   Miss  Cover   Missing
--------------------------------------------------------------
src/dataset/data_visualizer.py      48      0   100%
src/dataset/dataset.py              89      0   100%
src/models/DLinear.py               98      0   100%
src/models/TsfKNN.py               130      0   100%
src/models/base.py                  16      0   100%
src/models/baselines.py             54      0   100%
src/utils/decomposition.py          28      0   100%
src/utils/distance.py               41      0   100%
src/utils/metrics.py                16      0   100%
src/utils/transforms.py             85      0   100%
tests/__init__.py                    0      0   100%
tests/test_dataset.py               90      0   100%
tests/test_decomposition.py         35      0   100%
tests/test_distance.py              38      0   100%
tests/test_metrics.py               35      0   100%
tests/test_models.py                84      0   100%
tests/test_transforms.py            94      0   100%
tests/test_visualizer.py            33      0   100%
--------------------------------------------------------------
TOTAL                             1014      0   100%
```

## Roadmap

### Datasets

All datasets can be found [here](https://box.nju.edu.cn/d/b33a9f73813048b8b00f/).

- [x] M4
- [x] ETT
- [x] Traffic
- [x] Electricity
- [x] Exchange-Rate
- [x] Weather
- [x] ILI(illness)

### Models

- [x] ZeroForecast
- [x] MeanForecast
- [x] TsfKNN
- [x] LinearRegression
- [x] ExponentialSmoothing
- [x] DLinear
- [x] DLinearClosedForm

### Transformations

- [x] IdentityTransform
- [x] Normalization
- [x] Standardization
- [x] Mean Normalization
- [x] Box-Cox
- [x] YeoJohnsonTransform

### Metrics

- [x] MSE
- [x] MAE
- [x] MASE
- [x] MAPE
- [x] SMAPE

### Distance

- [x] euclidean
- [x] manhattan
- [x] chebyshev
- [x] minkowski
- [x] cosine