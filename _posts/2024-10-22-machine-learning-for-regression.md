---
date: 2024-10-22 09:28:34
layout: post
title: "Machine Learning for Regression"
subtitle: "A Comprehensive Overview of Regression Techniques in Machine Learning"
description: "This post explores various regression techniques in machine learning, from linear regression to more advanced methods like decision trees and random forests. It provides insights into how these models predict continuous outcomes, along with use cases and evaluation metrics such as RMSE, MAE, and R-squared."
image: https://github.com/user-attachments/assets/54f58b9f-47c9-4455-9a22-9f676613e72d
optimized_image: https://github.com/user-attachments/assets/54f58b9f-47c9-4455-9a22-9f676613e72d
category: Medium
tags: Machine Learning
author: Dimas
paginate: True
---
# Machine Learning  for Regression

### Part 1

## Car price prediction project

This project focuses on predicting car prices using a dataset from Kaggle. The objective is to build a predictive model through structured phases, each covered in individual blog posts. The main steps include:

## Project Plan
1. **Prepare Data and Exploratory Data Analysis (EDA)**
2. **Use Linear Regression for Predicting Price**
3. **Understand the Internals of Linear Regression**
4. **Evaluate the Model with RMSE (Root Mean Squared Error)**
5. **Feature Engineering**
6. **Regularization**
7. **Using the Model**

### Part 2

## Data Preparation

### Key Considerations:
- **Data Cleaning:** Handle missing values using techniques like mean/median imputation, and address outliers by removing or transforming them. Ensure consistent data formats.
- **Data Integration:** If multiple datasets exist, they may need to be merged based on common identifiers or shared attributes.
- **Data Transformation:** Feature engineering can be applied, such as transforming categorical variables into numerical ones using one-hot or label encoding.
- **Feature Scaling:** Apply standardization or normalization to ensure features are on a similar scale.
- **Train-Validation Split:** Split the dataset into training and validation sets to better evaluate the model.

## Pandas attributes and methods:

pd.read_csv(<file_path_string>) -> read csv files
df.head() -> take a look of the dataframe
df.columns -> retrieve colum names of a dataframe
df.columns.str.lower() -> lowercase all the letters
df.columns.str.replace(' ', '_') -> replace the space separator
df.dtypes -> retrieve data types of all features
df.index -> retrieve indices of a dataframe

### Example Code:
```python
import pandas as pd
import numpy as np

# Loading the data
df = pd.read_csv('data.csv')

# First overview of the data
df.head()

# Standardizing column names: lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.head()
```
## Cleaning String Columns
String columns are standardized similarly to column names. First, we identify columns of type object:
```python
# Identify string columns
strings = list(df.dtypes[df.dtypes == 'object'].index)

# Apply cleaning to all string columns
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')
df.head()
```
## Modeling: Linear Regression for Car Price Prediction
The goal is to predict MSRP (Manufacturer's Suggested Retail Price) using a linear regression model. This includes:

## Understanding Linear Regression: Learn how linear regression operates internally.
Feature Engineering: Create new features to improve model performance.
Regularization: Apply regularization techniques to prevent overfitting.

## Model Evaluation
RMSE (Root Mean Squared Error): RMSE will be used as the evaluation metric to measure the accuracy of the model.
Using the Model
Once the model is trained and evaluated, it will be used for predicting car prices based on the input features.

### Part 3

## Exploratory Data Analysis (EDA)

## General Information
Exploratory data analysis (EDA) is an essential step in the data analysis process. It involves summarizing and visualizing the main characteristics of a dataset to gain insights and identify patterns or trends. By exploring the data, researchers can uncover hidden relationships between variables and make informed decisions.

Common techniques in EDA include calculating summary statistics such as mean, median, and standard deviation to understand data distribution. These statistics help identify potential outliers or unusual patterns.

Visualizations play a crucial role in EDA. Graphical representations like histograms, scatter plots, and box plots help visualize data distribution, identify clusters, and detect unusual patterns or trends. They are particularly useful for understanding relationships between variables.

Data cleaning is another important aspect of EDA. This involves handling missing values, outliers, and inconsistencies. By carefully examining the data, researchers can decide how to handle missing values and address outliers or errors.

EDA is an iterative process. As researchers delve deeper into the data, they may uncover additional questions or areas of interest that require further exploration. This iterative approach helps refine understanding and uncover valuable insights.

In conclusion, EDA is crucial in the data analysis process. By summarizing, visualizing, and cleaning data, researchers can uncover patterns, identify relationships, and make informed decisions, providing a foundation for more advanced data analysis techniques.

## EDA for Car Price Prediction Project

### Getting an Overview
To understand the data, we examine each column and print some values. We can also look at unique values in each column to gain further insights.

### Distribution of Price
Visualizing the price column is essential. We can use histograms to observe the distribution of prices. The initial histogram may reveal a long-tail distribution, with many cars at lower prices and few at higher prices. Zooming in on prices under a certain threshold can help clarify the distribution. 

Applying a logarithmic transformation can address issues with long-tail distributions, resulting in a more normal distribution that is ideal for machine learning models.

### Missing Values
Identifying missing values is critical. We can use functions to find and sum missing values across columns, providing insights into which columns may need attention during model training.

### Notes
- Pandas attributes and methods:
  - `df[col].unique()` returns a list of unique values in the series.
  - `df[col].nunique()` returns the number of unique values in the series.
  - `df.isnull().sum()` returns the number of null values in the dataframe.
  
- Matplotlib and seaborn methods:
  - `%matplotlib inline` ensures that plots are displayed in Jupyter notebook's cells.
  - `sns.histplot()` shows the histogram of a series.
  
- Numpy methods:
  - `np.log1p()` applies a log transformation to a variable, after adding one to each input value.
  
Long-tail distributions can confuse machine learning models, so it is recommended to transform the target variable distribution to a normal one whenever possible.

### Part 4

## Setting Up the Validation Framework

To validate a model, the dataset is split into three parts: training (60%), validation (20%), and test (20%). The model is trained on the training dataset, validated on the validation dataset, and the test dataset is used occasionally to evaluate overall performance. The feature matrix (X) and target variable (y) are created for each partition: Xtrain, ytrain, Xval, yval, Xtest, and ytest.

To calculate the sizes of the partitions:
1. Determine the total number of records in the dataset.
2. Calculate 20% of the total records for validation and test datasets.
3. The training dataset size is computed by subtracting the sizes of the validation and test datasets from the total.

The data is then split sequentially into three datasets. However, to avoid issues arising from any inherent order in the dataset, the indices are shuffled. Shuffling ensures that all partitions contain a mix of records, preventing bias.

After shuffling, the datasets are created using the shuffled indices, and the old index is dropped to reset the index for each partition. A log1p transformation is applied to the target variable (msrp) to improve model performance.

Finally, the msrp values are removed from the dataframes to avoid accidental usage during training.

## Important Methods

- **Pandas**:
  - `df.iloc[]`: Returns subsets of records selected by numerical indices.
  - `df.reset_index()`: Resets the original indices.
  - `del df[col]`: Eliminates a column variable.

- **Numpy**:
  - `np.arange()`: Returns an array of numbers.
  - `np.random.shuffle()`: Returns a shuffled array.
  - `np.random.seed()`: Sets a seed for reproducibility.

The entire code for this project is available in the provided Jupyter notebook.

### Part 5

## Linear Regression

## Overview
Linear regression is a statistical method used to model the relationship between one or more input features and a continuous outcome variable. The objective is to find the best-fitting line that represents this relationship.

## Linear Regression Formula
The linear regression model can be expressed as:

$g(x_i) = w_0 + x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$.

And that can be further simplified as:

$g(x_i) = w_0 + \displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$

## Implementation in Python
A simple implementation of linear regression can be done as follows:
```python
w0 = 7.1
def linear_regression(xi):
    n = len(xi)
    pred = w0
    w = [0.01, 0.04, 0.002]
    for j in range(n):
        pred = pred + w[j] * xi[j]
    return pred
```
## Objective
The main goal of linear regression is to estimate the coefficients 
https://github.com/user-attachments/assets/17d1bd4c-0a89-4140-93f4-bfe6884bb306
​such that the sum of squared differences between the predicted and actual values is minimized. This is achieved using the ordinary least squares method.

## Single Observation Analysis
For a single observation the function can be simplified as: 
https://github.com/user-attachments/assets/6f3dcaf8-ad54-4504-9883-748d3be709ca
is a vector of characteristics for one instance, and 
https://github.com/user-attachments/assets/5e0cab46-49d3-4c0e-9cfd-f8ca82049b53 
is the corresponding target value.

## Example of Feature Extraction
Given a dataset, one can extract features:
```python
xi = [138, 24, 1385]  # Example features
```
## Full Function Implementation
The implementation can be expressed as:
```python
def linear_regression(xi):
    n = len(xi)    
    pred = w0
    for j in range(n):
        pred = pred + w[j] * xi[j]
    return pred
  ```  
## Inverse Transformation
Since the target variable is log-transformed, predictions must be converted back to the original scale using:

```python
np.expm1(predicted_value)
```

This process provides a comprehensive understanding of how linear regression works, its implementation, and the considerations for transforming predictions back to their original scale.

### Part 6 

## Linear Regression vector form

The formula for linear regression can be represented using the dot product between a feature vector and a weight vector. The feature vector includes a bias term with an x value of one, denoted as $( w_0 x_{i0} \)$, where \( x_{i0} = 1 \) for \( w_0 \). 

When considering all records, linear regression predictions are derived from the dot product between a feature matrix \( X \) and a weight vector \( w \). This can be expressed as \( g(x_i) = w_0 + x_i^T w \).

To implement the dot product, a function can be defined:

```python
def dot(xi, w):
    n = len(xi)
    res = 0.0
    for j in range(n):
        res += xi[j] * w[j]
    return res
```

The linear regression function can then be defined as:

```python
def linear_regression(xi):
    return w0 + dot(xi, w)
```

To simplify, we can introduce an additional feature always set to 1, leading to:
$g(xi) = w0 + xiTw -> g(xi) = w0xi0 + xiTw$

This implies the weight vector 𝑤 expands to an n+1 dimensional vector:
$w = [w0, w1, w2, … wn]$
$xi = [xi0, xi1, xi2, …, xin] = [1, xi1, xi2, …, xin]$
$wTxi = xiTw = w0 + …$

The dot product can now be used for the entire regression.

Given sample values:

```python
xi = [138, 24, 1385]
w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w
```

The updated linear regression function becomes:

```python
def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w_new)
```

For a matrix 𝑋 with dimensions 𝑚 × (𝑛 + 1), predictions can be calculated as follows:

```python
X = [[1, 148, 24, 1385], [1, 132, 25, 2031], [1, 453, 11, 86]]
X = np.array(X)
```

Predictions for each car price can be obtained using:

```python
y = X.dot(w_new)
```

To adjust the output for the actual price:

```python
np.expm1(y)
```

Finally, an adapted linear regression function can be expressed as:

```python
def linear_regression(X):
    return X.dot(w_new)
```
### Part 7

## Training linear regression: Normal equation


Obtaining predictions as close as possible to \( y \) target values requires the calculation of weights from the general LR equation. The feature matrix does not have an inverse because it is not square, so it is required to obtain an approximate solution, which can be obtained using the Gram matrix (multiplication of feature matrix \( X \) and its transpose \( X^T \)). The vector of weights or coefficients \( w \) obtained with this formula is the closest possible solution to the LR system.

Normal Equation:

```python
w = (X^T X)^{-1} X^T y
```

Where \( X^T X \) is the Gram Matrix.

Training a linear regression model, we know that we need to multiply the feature matrix \( X \) with weights vector \( w \) to get \( y \) (the prediction for price).

```python
g(X) = Xw \approx y
```

To achieve this, we need to find a way to compute \( w \). The equation \( Xw = y \) can be transformed into \( Iw = X^{-1}y \) when multiplied by \( X^{-1} \). However, \( X^{-1} \) exists only for squared matrices, and \( X \) is of dimension \( m \times (n+1) \), which is not square in almost every case.

We need to approximate \( X^T X w = X^T y \). The matrix \( X^T X \) is squared \( (n+1) \times (n+1) \) and its inverse exists.

```python
(X^T X)^{-1} X^T X w = (X^T X)^{-1} X^T y
```

```python
Iw = (X^T X)^{-1} X^T y
```

Thus, the value obtained is the closest possible solution:

```python
w = (X^T X)^{-1} X^T y
```

We need to implement the function `train_linear_regression`, that takes the feature matrix \( X \) and the target variable \( y \) and returns the vector \( w \).

```python
def train_linear_regression(X, y):
    pass
```
To approach this implementation, we first use a simplified example:
```python
X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 83],
    [38, 54, 185],
    [142, 25, 431],
    [453, 31, 86]
]
```
From the last article, we know that we need to add a new column with ones to the feature matrix 𝑋
X for the multiplication with WO. We can use np.ones() to create a vector of ones.
```python
ones = np.ones(X.shape[0])
```
Now we need to stack this vector of ones with our feature matrix 𝑋 using np.column_stack().
```python
X = np.column_stack([ones, X])
y = [10000, 20000, 15000, 25000, 10000, 20000, 15000, 25000, 12000]
```
Next, we compute the Gram matrix and its inverse.
```python
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
```
To check if the multiplication of 𝑋𝑇𝑋 with 𝑋𝑇𝑋 inv produces the Identity matrix 𝐼 :
```python
XTX.dot(XTX_inv).round(1)
```
Now we can implement the formula to obtain the full weight vector.
```python
w_full = XTX_inv.dot(X.T).dot(y)
```
From that vector 𝑤full, we can extract 𝑤0  and the other weights.
```python
w0 = w_full[0]
w = w_full[1:]
```
Finally, we implement the function train_linear_regression.
```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
     
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
     
    return w_full[0], w_full[1:]
```

Testing the implemented function:
```python
X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 83],
    [38, 54, 185],
    [142, 25, 431],
    [453, 31, 86]
]
y = [10000, 20000, 15000, 25000, 10000, 20000, 15000, 25000, 12000]

train_linear_regression(X, y)
```

### Part 8

## Building a Baseline Model for Car Price Prediction

In this lesson, we build a baseline model using the `df_train` dataset to derive weights for the bias (w0) and features (w). We utilize the `train_linear_regression(X, y)` function, focusing only on numerical features due to the nature of linear regression. Missing values in `df_train` are set to 0 for simplicity, although using non-zero values like the mean would be more appropriate.

The model's prediction function is defined as \( g(X) = w_0 + X \cdot w \). We then plot both predicted and actual values on a histogram for visual comparison.

## Car Price Baseline Model

We begin by constructing a model for car price prediction, extracting only numerical columns from the dataset. The relevant columns selected for the model include `engine_hp`, `engine_cylinders`, `highway_mpg`, `city_mpg`, and `popularity`.

To prepare for training, we extract the values from these columns. It’s crucial to check for missing values, as they can adversely affect model performance. In `df_train`, we find missing values in `engine_hp` and `engine_cylinders`. While filling these with zeros is a simple solution, it may not be the most accurate representation of the data. Nonetheless, we proceed with this approach for the current example.

After addressing the missing values, we reassign the updated values to `X_train`. We also prepare our target variable, `y_train`. 

We then use the `train_linear_regression` function to obtain values for w0 and the weight vector w. These variables allow us to apply the model to the training dataset to assess its performance.

To evaluate the model's accuracy, we calculate predicted values using the derived weights. Finally, we visualize the comparison between actual and predicted values using histograms, illustrating that while the model is not perfect, it serves as a foundational step for further improvement.

The next lesson will focus on more objective methods to evaluate regression model performance.


### Part 9

## RMSE for Model Evaluation

In the previous lesson, we noted that our predictions were somewhat inaccurate compared to the actual target values. To quantify the model's performance, we introduce Root Mean Squared Error (RMSE), a metric used to evaluate regression models by measuring the error associated with the predictions. RMSE enables comparison between models to determine which offers better predictions.

The formula for RMSE is given by:

\[
RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (g(x_i) - y_i)^2}
\]

where \(g(x_i)\) is the prediction, \(y_i\) is the actual value, and \(m\) is the number of observations.

## Root Mean Squared Error (RMSE)

To calculate RMSE, we utilize the predictions and actual values from our model. The process involves calculating the difference between the predicted and actual values, squaring this difference to obtain the squared error, and then averaging these squared errors to compute the Mean Squared Error (MSE). Finally, we take the square root of the MSE to find the RMSE.

We can implement RMSE in code, which allows us to obtain a numerical value representing the model's performance. 

Using our training data, we calculate the RMSE and find a value of approximately 0.746.

## Validating the Model

Evaluating the model performance solely on the training data does not provide a reliable indication of its ability to generalize to unseen data. Therefore, we proceed to validate the model using a separate validation dataset. We apply the RMSE metric again to assess performance on this unseen data.

To prepare the dataset consistently, we implement a `prepare_X` function that handles both training and validation sets. After preparing the datasets, we train the model and compute predictions for the validation data.

Upon calculating the RMSE for the validation dataset, we obtain a value of approximately 0.733. When comparing this RMSE with the training RMSE (0.746), we observe similar performance on both seen and unseen data, which aligns with our expectations for a well-generalized model.

### Part 10

## Computing RMSE on validation data

# Summary of RMSE Calculation for Car Price Prediction Model

## RMSE as a Performance Metric
- RMSE (Root Mean Squared Error) is introduced as a metric to evaluate model performance.
- It is calculated using the predictions and actual values from the dataset.

## Calculation Steps
1. **Prediction vs Actual Values**: Calculate the difference between predicted values \( g(x_i) \) and actual values \( y_i \).
2. **Squared Errors**: Square the differences to obtain the squared errors.
3. **Mean Squared Error**: Compute the mean of the squared errors.
4. **Root Mean Squared Error**: Take the square root of the mean squared error to obtain RMSE.

### Example Calculation
- Given predicted values and actual values, differences are computed, squared, averaged, and then the square root is taken to get RMSE.

## Implementation
- RMSE can be implemented in code with a function that calculates the squared errors, averages them, and returns the square root.

## Model Validation
- Evaluating the model on training data does not provide an accurate indication of its performance on unseen data.
- The model is applied to a validation dataset after training to assess performance using RMSE.

### Data Preparation
- A function is implemented to prepare datasets consistently across training, validation, and test sets.

## Results
- The RMSE is calculated for both training and validation datasets, showing similar performance on seen (training) and unseen (validation) data, indicating the model's robustness.

### Part 11

## Feature Engineering

The feature "age" of the car was derived from the dataset by subtracting the year of each car from the maximum year (2017). This new feature enhanced model performance, evidenced by a decrease in RMSE and improved distributions of the target variable and predictions.

### Simple Feature Engineering

To create the "age" feature, the following calculation was performed:

This resulted in a series representing the age of each car in the dataset. The new feature "age" was added to the `prepare_X` function, ensuring a copy of the dataframe was used to avoid modifying the original data.

### Implementation of `prepare_X`

The function `prepare_X` was defined to:
1. Copy the dataframe.
2. Calculate the "age" feature.
3. Compile a list of features to extract numerical values for model training.

The essential base features included:
- `engine_hp`
- `engine_cylinders`
- `highway_mpg`
- `city_mpg`
- `popularity`

### Model Training and Evaluation

The prepared training data was used to train a linear regression model:
```python
X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
```

The RMSE was calculated
```python
rmse(y_val, y_pred)
```
The results indicated a decrease in RMSE from approximately 0.733 to 0.515, demonstrating significant improvement in model performance.

Visualization
Histograms comparing predicted values and actual values showed a clear enhancement in prediction accuracy, although further improvement was still possible.

```python
sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
sns.histplot(y_val, color='blue', alpha=0.5, bins=50)
```
### Part 12

## Categorical Variables and One-Hot Encoding in Machine Learning

## Introduction
Categorical variables are often represented as strings in pandas, typically identified as object types. Some variables that seem numerical, like the number of doors in a car, are actually categorical. For machine learning (ML) models to interpret these variables, they need to be converted into a numerical format. This transformation is known as **One-Hot Encoding**.

## Categorical Variables
In the dataset, categorical variables include: 
- `make`
- `model`
- `engine_fuel_type`
- `transmission_type`
- `driven_wheels`
- `market_category`
- `vehicle_size`
- `vehicle_style`

### Special Case: Number of Doors
The `number_of_doors` variable appears numerical but is categorical, as shown in the data type output where it is classified as `float64`. 

## One-Hot Encoding Process
One-hot encoding creates binary columns for each category in the variable. For example, for `number_of_doors`, we generate:
- `num_doors_2`
- `num_doors_3`
- `num_doors_4`

The implementation uses boolean conditions to create new binary features.

### Implementation Example
```python
for v in [2, 3, 4]:
    df_train['num_doors_%s' % v] = (df_train.number_of_doors == v).astype('int')
```
### Feature Preparation Function
The prepare_X function processes the dataframe to create numerical features, including:

### Calculating the car's age.
Applying one-hot encoding for the number_of_doors.
The function fills missing values and extracts the feature array.

### Output Example
The processed output from the function shows the newly created features for the number of doors.

### Model Training and Evaluation
After preparing the features, the model is trained, and performance is evaluated using RMSE. An initial run shows a minor improvement in performance with new features added.

### Expanding Categorical Features
Next, we consider other categorical variables like make, where there are 48 unique values. We focus on the top 5 most common makes to avoid dimensionality issues.

### Updated Prepare Function
The prepare_X function is modified to include one-hot encoding for the make variable and others.

### Evaluating Additional Features
Adding the new features again improves the model slightly.

### Comprehensive Categorical Encoding
To enhance performance, a comprehensive list of categorical variables is created, and a loop is used to generate one-hot encodings for each.

### Final Implementation
The final implementation of prepare_X incorporates two loops—one for the categorical variable names and one for their respective values.

### Results and Issues
Upon re-evaluating the model with all features, a significant increase in RMSE suggests a possible issue, indicating that the approach or the features may need to be reassessed.

## Conclusion
The transition from categorical to numerical variables is crucial for ML model performance, as seen in the various implementations. However, care must be taken to ensure that the added complexity genuinely benefits model accuracy.

### Part 13

## Regularization in Linear Regression

## Introduction
In linear regression, the feature matrix may contain duplicate columns or columns that can be expressed as linear combinations of others. This results in a singular matrix when calculating the inverse, leading to poor model performance. A common approach to address this issue is through **regularization**.

## Problem with Duplicate Columns
When duplicate columns exist in the feature matrix \(X\), the Gram matrix \(X^TX\) becomes singular, making its inverse non-existent. For example, if two columns are identical, attempting to compute \( \text{np.linalg.inv}(X^TX) \) results in a "Singular matrix" error. 

### Example
Given a matrix \(X\):
```python
X = [
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5]
]
```
The Gram matrix calculated is:
XTX = X.T.dot(X)
This results in duplicate columns, leading to a singular matrix.

### Regularization Technique
To mitigate the effects of duplicate columns, a small value (alpha) can be added to the diagonal of the Gram matrix:
XTX = XTX + \alpha \cdot I Where 𝐼 is the identity matrix. This addition improves the likelihood of obtaining a non-singular matrix and stabilizes the computation of the weights.

### Impact of Noise

Introducing slight noise to the duplicate columns can also make the columns no longer identical, thus allowing the computation of the inverse:
```python
X = [
    [4, 4, 4],
    [3, 5, 5],
    [5, 1, 1],
    [5, 4, 4],
    [7, 5, 5],
    [4, 5, 5.0000001],
]
``` 
This adjustment results in a computable Gram matrix 𝑋𝑇𝑋 with non-singular properties.

### Practical Implementation
To incorporate regularization in the linear regression training function:

```python
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]
```

### Results
After applying the regularization technique with a regularization parameter 𝑟=0.01, the root mean square error (RMSE) improved significantly:
```python
rmse(y_val, y_pred)  # Output: 0.45685446091134857
```
This demonstrates the effectiveness of regularization in controlling the weights and improving model performance.

### Conclusion
Regularization is an essential technique in linear regression to address issues arising from duplicate features. By adding a small value to the diagonal of the Gram matrix, we can stabilize the inverse calculation, resulting in better model performance. Future work will involve optimizing the regularization parameter  


### Part 14

## Model Tuning 

Model Tuning
The process of tuning the linear regression model involved identifying the optimal regularization parameter r using a validation set. The goal was to determine how this parameter impacts model performance.

### Hyperparameter Search
A range of values for r was tested:

```python
for r in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    
    score = rmse(y_val, y_pred)
    print("reg parameter: ", r, "bias term: ", w0, "rmse: ", score)
```    
### Results Summary

r = 0.0: Huge bias term and high RMSE.
Optimal r = 0.001: RMSE was found to be 0.4568807317131709, indicating good model performance.

### Final Model Training
After identifying the optimal r, the model was retrained on the combined training and validation datasets.

### Combining Datasets
The datasets were concatenated using:

```python
df_full_train = pd.concat([df_train, df_val])
y_full_train = np.concatenate([y_train, y_val])
df_full_train = df_full_train.reset_index(drop=True)
```

### Preparing Features
The feature matrix was prepared using:

```python
X_full_train = prepare_X(df_full_train)
```

### Final Training
The model was trained on the full dataset:

```python
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
```

### Testing the Model
The final model was evaluated on a test dataset to check its performance:

```python
X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
print("rmse: ", score)
```

### Results

Test RMSE: 0.5094518818513973, indicating good generalization as it was close to the validation RMSE.
Using the Model for Predictions
The final model can be utilized to predict the price of an unseen car by extracting features and applying the model.

### Feature Extraction

For instance, extracting features from a car in the test dataset:

```python
car = df_test.iloc[20].to_dict()
df_small = pd.DataFrame([car])
X_small = prepare_X(df_small)
```
### Price Prediction

The model was applied to the feature vector:

```python
y_pred = w0 + X_small.dot(w)
y_pred = np.expm1(y_pred[0])  # Undoing the logarithm
```
### Final Predicted Price

The predicted price was approximately $21,044.36.

### Actual Price Comparison

Comparing with the actual price:

```python
actual_price = np.expm1(y_test[20])  # Output: 34975.0
```

This comprehensive approach illustrates the steps involved in tuning a linear regression model, training it on a combined dataset, and making predictions on unseen data. The model performed well, showing generalization capabilities through consistent RMSE values.

### Part 15

## Using the model

Using the model involves two main steps:

1. **Feature Extraction**: Extracting the feature vector from a car's attributes.
2. **Price Prediction**: Applying the trained model to the feature vector to predict the car's price.

### Feature Extraction
To demonstrate the model's functionality, we take a specific car from the test dataset as if it were a new car. For example, the selected car has the following features:

- Make: Saab
- Model: 9-3 Griffin
- Year: 2012
- Engine Fuel Type: Premium Unleaded (Recommended)
- Engine HP: 220.0
- Engine Cylinders: 4.0
- Transmission Type: Manual
- Driven Wheels: All Wheel Drive
- Number of Doors: 4.0
- Market Category: Luxury
- Vehicle Size: Compact
- Vehicle Style: Wagon
- Highway MPG: 30
- City MPG: 20
- Popularity: 376

This information can be represented as a Python dictionary, simulating data input from a user on a website or app.

### Creating a DataFrame for the Model
To prepare the data for the model, we convert the dictionary into a single-row DataFrame:

```python
df_small = pd.DataFrame([car])
```

This DataFrame is then passed to the prepare_X() function to generate the feature matrix (feature vector).

### Price Prediction
Once we have the feature vector, we apply the final model to predict the price:

```python
y_pred = w0 + X_small.dot(w)
```
To obtain the actual price in dollars, we must undo the logarithm transformation applied during training:
```python
predicted_price = np.expm1(y_pred)
```
For our example, this results in a predicted price of approximately $21,044.36.

### Model Performance Evaluation

Finally, we can evaluate the model's performance by comparing the predicted price to the actual price of the car:
```python
actual_price = np.expm1(y_test[20])
```
The actual price of the selected car was $34,975.00, highlighting the discrepancy between the predicted and actual values.

## Summary of Linear Regression Process

### Data Preparation
- Import necessary libraries, including NumPy and Pandas.
- Load the dataset containing information about cars or relevant features.
- Identify the feature columns to be used in the regression model.

### Pre-Processing
- Fill missing values in the dataset with zeros.
- Calculate new features, such as the age of the vehicle based on the manufacturing year.
- Create dummy variables for categorical features.

### Building the Linear Regression Model
- Develop the `train_linear_regression` function to calculate model weights (coefficients) using the Least Squares method.
- Implement regularization with the `train_linear_regression_reg` function to address multicollinearity by adding a regularization parameter (r).

### Model Training
- Prepare the feature matrix (X) and target (y) from the training and validation data.
- Train the model using the training and validation data.
- Calculate predictions using the trained model and compute errors using the Root Mean Square Error (RMSE) function.

### Model Evaluation
- Use the full training data (combined training and validation data) to train the final model.
- Compute predictions on the test dataset and calculate the RMSE score to evaluate model performance.

### Individual Prediction
- Extract a single entry from the test data for individual prediction.
- Calculate the predicted value and convert it back to the original scale using the `np.expm1` function.
