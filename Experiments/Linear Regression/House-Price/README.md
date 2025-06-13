---

House Price Prediction using Linear Regression

This project focuses on predicting house prices using the Linear Regression model. The dataset was sourced from Kaggle's House Price Prediction challenge : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The pipeline includes data preprocessing, model training, evaluation, and testing.

Steps Followed in the Project

1. Load the Training Data

The dataset train.csv was loaded using pandas.

The target column SalePrice was transformed using np.log1p() to reduce skewness and improve model performance.

---

2. Data Splitting

Features (X) and target (y) were separated.

The dataset was split into training and validation sets using train_test_split().

---

3. Dropping Unnecessary Columns

Columns with a high percentage of missing values were identified and dropped to maintain data quality.

---

4. Handling Missing Values

Categorical Columns: Missing values were replaced with the mode (most frequent value).

Numerical Columns: Missing values were replaced with the mean.

---

5. One-Hot Encoding

Categorical columns were converted into numeric format using One-Hot Encoding to make them suitable for the regression model.

---

6. Feature Scaling

Standardization was applied to the feature data for both training and validation datasets to bring all features to the same scale.

---

7. Model Training

A Linear Regression model was chosen for training.

The model was fitted on the training data (X_train, y_train).

---

8. Validation

Predictions were made on the validation dataset.

Evaluation metrics such as R² score, Mean Absolute Error (MAE), and Mean Squared Error (MSE) were calculated to assess the model's performance.

---

9. Testing

The test.csv file was loaded and processed using the same steps applied to the training data (missing value handling, encoding, and scaling).

Predictions were made on the test data.

The trained model was saved as a .pkl file for future use.

---

Requirements

The project uses the following libraries:

Python 3.x

pandas

numpy

matplotlib

scikit-learn

joblib


To install the required libraries, run:

pip install pandas numpy scikit-learn matplotlib joblib


---


Results

Validation R² Score: 0.91

Validation MAE: 0.0884

Validation MSE: 0.0157

Training Accuracy: 0.9435


---

Future Work

Explore feature importance to better understand the key factors influencing house prices.

Experiment with additional models such as Decision Trees, Random Forests, and Gradient Boosting.

Implement cross-validation for better evaluation.



---
