
# Network Anomaly Detection


## Overview

This repository contains scripts to detect anomalies in network traffic using machine learning techniques, specifically Isolation Forest and a deep learning model. It includes preprocessing, training, evaluation, and visualization steps to assess model performance.

### Features

- **Isolation Forest**: An unsupervised learning algorithm for anomaly detection.
- **Deep Learning Model**: A neural network trained for binary classification to refine anomaly detection.

## Installation

To run the scripts, ensure you have Python 3.x and the necessary libraries installed. Clone this repository and install dependencies:

```
git clone https://github.com/yourusername/network-anomaly-detection.git
cd network-anomaly-detection
pip install -r requirements.txt
```

## Dataset

The synthetic network traffic dataset used for this project can be found [here](https://link-to-your-dataset). Place the dataset file (`synthetic_network_traffic.csv`) in the `/data` directory.

## Usage

1. **Data Preprocessing**: 
   - Run `preprocess_data.py` to load the dataset, perform feature engineering, and oversample the 'Anomaly' class.

2. **Model Training**:
   - Execute `train_models.py` to train both Isolation Forest and the deep learning model on the preprocessed data.

3. **Evaluation**:
   - Use `evaluate_models.py` to evaluate model performance using metrics such as ROC curve, AUC, confusion matrix, and classification report.

4. **Visualization**:
   - Visualize results with plots generated from `plot_results.py`.

## Results

### Performance Metrics

- **Isolation Forest**:
  - AUC: 0.85
  - Confusion Matrix:
    ```
    [[1542  208]
     [  45  205]]
    ```

- **Deep Learning Model**:
  - AUC: 0.92
  - Confusion Matrix:
    ```
    [[1592  158]
     [  32  218]]
    ```

## Contributing

Contributions to improve this project are welcome. To contribute, please fork the repository and submit a pull request with your proposed changes.


