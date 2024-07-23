# EstateForecaster-Precision-Pricing
## House Prices - Advanced Regression Techniques

## Project Overview

This project aims to predict the final sale price of homes in Ames, Iowa, using a dataset with 79 explanatory variables. The goal is to understand and model various aspects influencing home prices beyond basic features, utilizing advanced regression techniques and creative feature engineering to achieve accurate predictions.

**Dataset Link:** [Kaggle Competition Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)  
**Starter Code Link:** [Google Colab Notebook](https://colab.research.google.com/drive/1eMQwJMTU9W_UtzHBLWX-tcNw6bcg_bSe#scrollTo=6PA13VhO6O5q)

## Goal

Here primary goal is to predict the sales price for each house in the test set. For each `Id` in the test set, you must predict the value of the `SalePrice` variable.

## Approach

### 1. Data Uploading
- Load the dataset and prepare it for analysis.

### 2. Outlier Removal
- Identify and remove outliers to improve model performance.

### 3. Missing Value Handling
- Use median imputation to replace missing values instead of mean to reduce the impact of outliers.

### 4. Feature Engineering
- Create new features and modify existing ones to enhance model performance.

### 5. Exploratory Data Analysis (EDA)
- Perform EDA to understand the data distribution and relationships between variables.

### 6. Preprocessing
- Standardize features using `StandardScaler` for normalization.
- Handle missing values using `SimpleImputer`.

### 7. Model Selection
- Train and evaluate multiple regression models:
  - **Linear Regression:** A basic regression model that assumes a linear relationship between features and target.
  - **K-Nearest Neighbors (KNN):** Predicts target values based on the average target values of the k-nearest neighbors.
  - **Support Vector Machine (SVM):** Uses support vectors to find the optimal hyperplane for regression tasks.
  - **Decision Tree Regression:** Models predictions by partitioning the data into regions based on feature values.
  - **ElasticNet Regression:** Combines penalties of Lasso and Ridge regression for regularization.
  - **Bayesian Ridge Regression:** Uses Bayesian inference for regularization and prediction.
  - **AdaBoost Regression:** Boosts weak models to improve prediction accuracy through an ensemble approach.
  - **Random Forest Regression:** An ensemble of decision trees to improve prediction accuracy and control overfitting.
  - **XGBoost Regression:** Optimized gradient boosting algorithm for efficient and scalable regression.
  - **Ridge Regression:** Adds L2 regularization to linear regression to prevent overfitting.
  - **Gradient Boosting Regression:** Sequentially builds models to correct errors of previous models through boosting.
  - **LightGBM (LGBM):** Gradient boosting framework that uses tree-based learning algorithms.
  - **CatBoost:** Gradient boosting algorithm that handles categorical features directly.
  - **Voting Regressor:** Combines predictions from multiple regression models to improve overall performance.
  - **Stacking Regressor:** Uses predictions from multiple models as input for a final meta-model to enhance predictive performance.

### 8. Model Evaluation
- Evaluate models using:
  - **Root Mean Squared Error (RMSE):** Measures the square root of the average squared differences between predicted and actual values.
  - **Relative Absolute Error (RAE):** Measures the error relative to a simple model that predicts the mean value.
  - **Mean Squared Error (MSE):** Measures the average of the squared differences between predicted and actual values.
  - **Mean Absolute Error (MAE):** Measures the average of the absolute differences between predicted and actual values.
  - **R-squared (R2):** Indicates the proportion of variance in the dependent variable that is predictable from the independent variables.

## Data Preprocessing

### Scalers
- **StandardScaler:** Standardizes features by removing the mean and scaling to unit variance.

### Imputation
- **SimpleImputer:** Handles missing values by replacing them with the median of the column.

## Encoders

### Ordinal Encoder
- Transforms categorical features with ordinal relationships into numerical values.

### One-Hot Encoder
- Converts categorical features with no ordinal relationship into binary columns suitable for machine learning algorithms.

## Repository Structure

- `data/` - Folder containing the dataset.
- `notebooks/` - Jupyter notebooks for EDA and modeling.
- `scripts/` - Python scripts for preprocessing, feature engineering, and model training.
- `results/` - Folder for storing evaluation metrics and results.
- `README.md` - This file.

## Contributing

Feel free to contribute to this repository by suggesting improvements or submitting pull requests. Your contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
