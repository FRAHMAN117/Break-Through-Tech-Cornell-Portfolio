# Break-Through-Tech-Cornell-Portfolio
This project utilizes the World Happiness Report dataset to build a machine learning model that predicts a country's (happiness index). Through data preprocessing, exploratory data analysis, and regression modeling, the project identifies key factors influencing national happiness, providing insights for policy-making and resource allocation.
# World Happiness Report - Machine Learning Project

## Overview

This repository contains a machine learning project focused on predicting a country's "Life Ladder" score, which serves as an indicator of national happiness. The project leverages the World Happiness Report dataset, applying a standard machine learning lifecycle: data loading, problem definition, exploratory data analysis (EDA), data preprocessing, model training, and evaluation. The primary goal is to identify the most influential factors contributing to a country's happiness and to build a robust regression model for prediction.

## Project Goal

The main objective is to predict the `Life Ladder` (happiness index) of a country based on various socio-economic and emotional factors. This is a **supervised regression problem**.

## Dataset

The dataset used is the "World Happiness Report (WHR) data set" (`WHR2018Chapter2OnlineData.csv`), which contains various indicators related to well-being and happiness across different countries and years.

## Key Features & Target Variable

* **Target Variable (Label):** `Life Ladder` (renamed to `lifeL`) - A numerical score representing a country's happiness.
* **Key Features (after preprocessing):**
    * `log_GDP`: Log GDP per capita
    * `socialS`: Social support
    * `life_Exp`: Healthy life expectancy at birth
    * `freedom`: Freedom to make life choices
    * `generosity`: Generosity
    * `corruption`: Perceptions of corruption
    * `positive`: Positive affect
    * `negative`: Negative affect
    * `gov_confidence`: Confidence in national government
    * `dem_Quality`: Democratic Quality
    * `dev_Quality`: Delivery Quality
    * `gini_index`: GINI index (World Bank estimate)
    * `gini_Avg`: GINI index (World Bank estimate), average 2000-15
    * `gini_householdInc`: gini of household income reported in Gallup, by wp5-year
    * `year`: Year of data collection
    * `country_X`: One-hot encoded features for the top 20 most frequent countries.

## Machine Learning Lifecycle & Methodology

The project follows these steps:

1.  **Data Loading:** The `WHR2018Chapter2OnlineData.csv` dataset is loaded into a Pandas DataFrame.
2.  **Problem Definition:** Defined as a supervised regression problem to predict `Life Ladder`.
3.  **Exploratory Data Analysis (EDA):**
    * Initial inspection using `df.info()` and `df.describe()` to understand data types, non-null counts, and statistical summaries.
    * Correlation analysis using a heatmap (`seaborn.heatmap`) to identify relationships between features and the target variable.
    * Outlier detection using box plots (`seaborn.boxplot`) for numerical features.
4.  **Data Preprocessing:**
    * **Column Renaming:** Simplified long column names for easier use (e.g., `Life Ladder` to `lifeL`).
    * **Handling Missing Values:** Imputed missing numerical values with the mean of their respective columns.
    * **Outlier Treatment:** Applied Winsorization (limits=[0.01, 0.01]) to numerical features to cap outliers at the 1st and 99th percentiles, as the data's social nature suggests high variability rather than erroneous entries.
    * **Feature Engineering (One-Hot Encoding):** Converted the categorical `country` column into binary (one-hot encoded) features for the top 20 most frequent countries, dropping the original `country` column.
    * **Feature Selection:** Removed `std` and `stdMean` columns to prevent data leakage, as they are derived from the target variable (`lifeL`).
5.  **Data Splitting:** The dataset is split into training (70%) and testing (30%) sets using `train_test_split`.
6.  **Feature Scaling (Standardization):** Numerical features (excluding one-hot encoded country columns) are standardized using `StandardScaler` to ensure they contribute equally to the model.
7.  **Model Training:** A Linear Regression model (`sklearn.linear_model.LinearRegression`) is used to train on the preprocessed data.
8.  **Model Evaluation:** The model's performance will be evaluated using metrics appropriate for regression problems, such as Mean Squared Error (MSE) and R-squared (`r2_score`).
9.  **Model Improvement:** (To be implemented in subsequent steps, potentially involving hyperparameter tuning, trying different models, or advanced feature engineering).

## Setup and Usage

### Prerequisites

* Python 3.x
* Jupyter Notebook or JupyterLab (for running the `.ipynb` file)
* Required Python libraries:
    * `pandas`
    * `numpy`
    * `matplotlib`
    * `seaborn`
    * `scipy`
    * `scikit-learn`

### Installation

1.  Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scipy scikit-learn
    ```

### Running the Notebook

1.  Place the `WHR2018Chapter2OnlineData.csv` file in a `data` directory within your project's root folder.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Navigate to and open the project notebook (e.g., `Lab_8_ML_Project.ipynb`).
4.  Run all cells in the notebook to see the data loading, preprocessing, and model training steps.

## Future Work

* Implement model evaluation metrics (MSE, R-squared).
* Perform hyperparameter tuning for the Linear Regression model or explore other regression models (e.g., RandomForestRegressor, GradientBoostingRegressor).
* Conduct feature importance analysis to definitively identify the two most important features.
* Visualize model predictions vs. actual values.
* Explore more advanced feature engineering techniques.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

