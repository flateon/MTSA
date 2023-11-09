# Homework 2

Homework 2 considers multivariate time series (multi input multi output). The dataset used here is `ETTh1`.
## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model MeanForecast
```

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model TsfKNN --n_neighbors 1 --msas MIMO --distance euclidean
```

## Part 1 Update Existing Transformation Class (10 pts)
path: `src/utils/transforms.py`

**Objective:** Modify the `Standardization` transformation class to handle multivariate data.


## Part 2 TsfKNN (40 pts)

path: `src/models/TsfKNN.py`

**Objective:**
Refine the TsfKNN model to improve its forecasting ability for multivariate time series. This involves implementing a robust distance metric for multivariate data and designing an effective temporal embedding strategy.

**Instructions:**

**1. Multivariate Distance Metrics**

Implement additional distance metrics to handle multivariate sequences.
Make sure the distance function can compare two multivariate time series and return a scalar distance value.

**2. Temporal Embedding Concepts**

Learn about temporal embeddings and how they can encapsulate the temporal information within a time series.
Explore different embedding techniques such as Fourier transforms, autoencoder representations, or other methods. 
Note that lag-based embeddings are already implemented in the `TsfKNN` model. You can modify the time lag parameter tau and dimension m as described in PPT to improve the performance.

## Part 3 DLinear (30 pts)

path: `src/models/DLinear.py`

**Objective:** 
Implement the DLinear model as described in the provided [paper](https://arxiv.org/pdf/2205.13504.pdf).
You can define the dataloader yourself or modify `trainer.py` if necessary.

## Part 4 Decomposition (20 pts)

path: `src/utils/decomposition.py`

**Objective:**
Implement time series decomposition methods to separate the trend and seasonal components from the original time series data and integrate these methods into the TsfKNN and DLinear forecasting models.

**Instructions:**

**1. Moving Average Decomposition**

Implement the moving_average function that calculates the trend and seasonal components using a moving average with a specified seasonal period.

**2. Differential Decomposition**

Implement the differential_decomposition function that separates the trend and seasonal components by differencing the time series data.
Determine how to calculate the differences and reconstruct the trend and seasonal components from these differences.

**3. Other Decomposition Method (bonus 10 pts)**

Explore other decomposition methods as you like.

## Part 5 Evaluation

**Instructions:**

**1. Exploring Temporal Embedding and Distance Combinations in TsfKNN**

  In your report, write down the details of your method and fill the table below.

 | Temporal Embedding | Distance  | MSE  | MAE  |
 |--------------------|-----------| ----- | ----- |
| lag-based (lag=96) | euclidean      |      |      |
 |                    | ...       |      |      |

**2. Decomposition Method Evaluation for TsfKNN and DLinear**

Transform the data by Standardization and apply different decomposition methods.
In your report, write down the details of your method and fill the table below.
    
| Model  | Decomposition | MSE  | MAE  |
 |--------|---------------| ----- | ----- |
| TsfKNN | MA            |      |      |
 |        | ...           |      |      |



## Submission

**1. Modified Code:**

- Provide the modified code for all components of the task.
- Include a `README.md` file in Markdown format that covers the entire task. This file should contain:
  - how to install any necessary dependencies for the entire task.
  - how to run the code and scripts to reproduce the reported results.
  - datasets used for testing in all parts of the task.

**2. PDF Report:**

- Create a detailed PDF report that encompasses the entire task. The report should include sections for each component of the task.

**3. Submission Format:**

- Submit the entire task, including all code and scripts, along with the `README.md` file and the PDF report, in a compressed archive (.zip).

**4. Submission Deadline:**
  2023-11-21 23:55
