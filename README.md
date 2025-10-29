# Credit Card Fraud Detection Capstone Project

### Project Objective

The goal of this project is to develop and compare several machine learning models for the accurate detection of fraudulent credit card transactions. The primary challenge addressed is the highly imbalanced nature of the dataset, where fraudulent transactions constitute a very small minority at just 0.173%. Consequently, model success is measured not by raw accuracy, but by its ability to effectively identify fraudulent cases (a high recall) while simultaneously minimizing false alarms (a high precision).

### Dataset

This project utilizes the "Credit Card Fraud Detection" dataset from Kaggle, which was originally provided by researchers at Université Libre de Bruxelles (ULB).

The dataset can be accessed via the following link: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions, of which only 492 are fraudulent. For confidentiality purposes, the original transaction features were transformed using Principal Component Analysis (PCA) into 28 anonymized numerical features, labeled `V1` through `V28`. The only features that remain in their original form are `Time` and `Amount`.

### Exploratory Data Analysis (EDA) - Key Findings

A thorough exploratory data analysis was conducted to understand the dataset's structure and inherent challenges.

The initial check for data integrity confirmed that the dataset is exceptionally clean, containing no missing or null values. The most significant finding was the extreme class imbalance, with fraudulent transactions making up only 0.173% of the data. This confirmed that standard accuracy would be a misleading metric for evaluation.

An analysis of the feature distributions showed that the `Time` and `Amount` features were heavily skewed. To mitigate the effect of outliers and prepare the data for modeling, the `RobustScaler` was applied to these two features.

To assess whether the classes were separable, Kernel Density Plots were generated. These plots revealed that the distributions for several of the V-features (such as V4, V14, and V17) were distinctly different for fraudulent versus non-fraudulent transactions, providing a clear signal for a machine learning model to learn from. Furthermore, a t-SNE visualization was used to project the high-dimensional data into a two-dimensional space. This visualization visually confirmed that the fraudulent and non-fraudulent classes form largely distinct clusters, reinforcing the feasibility of building an effective classifier.

### Model Training Pipeline

The complete model training pipeline is documented within the Jupyter Notebook located at `notebooks/TaskFile.ipynb`.

#### Setup & Installation

To run this project, first clone the repository. Then, install the required Python packages using the command below, which references the `requirements.txt` file included in the repository.

```bash
pip install -r requirements.txt
