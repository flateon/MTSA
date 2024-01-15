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
conda activate MTSA
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
python benchmark_global.py
python benchmark_spirit.py
```

The reported results will be store in *./results/test_\*.csv*

### Unittest and coverage report

To test our implementation, run the unittest with the following command:

```bash
coverage run -m unittest; coverage report -m
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
- [x] FLinear
- [x] SPIRIT

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

### Decomposition

- [x] MovingAverage
- [x] Differencing
- [x] Classical
- [x] Henderson
- [x] STL
- [x] X11
