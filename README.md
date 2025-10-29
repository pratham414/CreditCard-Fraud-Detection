\# Credit Card Fraud Detection Capstone Project



\## 1. Project Objective



The goal of this project is to develop and compare several machine learning models to accurately detect fraudulent credit card transactions. The primary challenge is the highly imbalanced nature of the dataset, where fraudulent transactions represent a very small minority (0.173%). Success is measured not by raw accuracy, but by the model's ability to effectively identify fraudulent cases (high recall) while minimizing false alarms (high precision).



\## 2. Dataset



This project utilizes the \*\*Credit Card Fraud Detection\*\* dataset from Kaggle, provided by Université Libre de Bruxelles (ULB).



\- \*\*Link:\*\* \[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

\- \*\*Content:\*\* The dataset consists of 284,807 transactions, of which only 492 are fraudulent. For confidentiality, the original features have been transformed via Principal Component Analysis (PCA) into 28 numerical features (`V1` to `V28`). The only features not anonymized are `Time` and `Amount`.



\## 3. Exploratory Data Analysis (EDA) - Key Findings



A thorough EDA was conducted to understand the dataset's structure and challenges.

\- \*\*Data Integrity:\*\* The dataset is exceptionally clean with no missing or null values.

\- \*\*Extreme Class Imbalance:\*\* Fraudulent transactions make up only \*\*0.173%\*\* of the data. This makes standard accuracy a misleading metric and requires specialized techniques to handle.

\- \*\*Feature Scaling:\*\* The `Time` and `Amount` features were found to be heavily skewed. `RobustScaler` was applied to these features to make them less sensitive to outliers before model training.

\- \*\*Class Separability:\*\*

&nbsp;   - Kernel Density Plots showed that the distributions for several V-features (e.g., V4, V14, V17) are distinctly different for fraudulent vs. non-fraudulent transactions, providing a clear signal for the models.

&nbsp;   - A \*\*t-SNE visualization\*\* projected the high-dimensional data into 2D, visually confirming that the fraudulent and non-fraudulent classes form largely separable clusters.



\## 4. Model Training Pipeline



The training pipeline is contained within the `notebooks/TaskFile.ipynb`.



\#### \*\*Setup \& Installation\*\*

To run this project, clone the repository and install the required packages:

```bash

pip install -r requirements.txt

