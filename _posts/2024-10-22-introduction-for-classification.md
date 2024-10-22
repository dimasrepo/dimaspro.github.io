---
date: 2024-10-22 09:29:21
layout: post
title: "Introduction for Classification"
subtitle: "A Beginner's Guide to Classification in Machine Learning"
description: "This post introduces the fundamental concepts of classification in machine learning, exploring how models can categorize data into predefined labels. It covers basic algorithms such as logistic regression, decision trees, and support vector machines, providing a foundation for understanding supervised learning tasks."
image: https://github.com/user-attachments/assets/54f58b9f-47c9-4455-9a22-9f676613e72d
optimized_image: https://github.com/user-attachments/assets/54f58b9f-47c9-4455-9a22-9f676613e72d
category: Medium
tags: Machine Learning
author: Dimas
paginate: True
---

# ML Zoomcamp 2024: Introduction for Classification

### Part 1

# Churn Prediction Project

## Project Overview
In this project, we focus on predicting customer churn for a telecommunications company that has a varied customer base. While some customers are satisfied with the services they receive, others are considering terminating their contracts and switching to a different service provider.

## Objective
Our goal is to identify customers who are likely to churn (i.e., stop using the service). We will assign each customer a score between 0 and 1 that indicates their likelihood of leaving. This score will help us target at-risk customers with promotional emails and discounts to encourage retention.

## Approach
We will approach this problem using Binary Classification through Machine Learning. The mathematical representation of a single observation is defined as:

\[ g(x_i) \approx y_i \]

In this scenario, our target variable \( y_i \) indicates whether customer \( x_i \) has left the company. The feature vector \( x_i \) consists of various attributes describing the customer. The output of our model, \( y_i \), is a value between {0, 1}, representing the likelihood that a specific customer will churn. A score of 1 signifies that the customer has left the company (positive example), while a score of 0 indicates that the customer has not left (negative example). Thus, a score of 1 means that the effect we want to predict is present, while a score of 0 means it is absent.

## Data Collection
To achieve our goal, we will analyze customer data from the previous month. For each customer, we will label those who actually left with a target label of 1 and those who remained with a label of 0. All target labels will collectively form the variable 'y.' The features related to the customers—including demographics, geographical location, payment amounts, services used, and contract types—will form the variable 'X.'

Our main objective is to gather and analyze information to understand the factors that lead to customer churn. To build our model, we will use historical data from the dataset available on Kaggle titled **"Telco Customer Churn – Focused Customer Retention Programs."** The specific column we aim to predict is the **‘Churn’** column.

### Part 2

# Data Preparation

This section covers the following topics:

- Downloading the data
- Reading the data
- Standardizing column names and values
- Verifying column data integrity
- Preparing the churn variable

## Downloading the Data

We begin by importing the necessary packages and downloading our CSV file using the `wget` command. In Jupyter Notebook, the `!` symbol allows us to execute shell commands, while `$` is used to reference data within those commands.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_url = "https://..."
!wget $data_url -O data-week-3.csv
```
Reading the Data
Upon reading the data, we find that it consists of 21 columns. The presence of ellipses (...) in the header indicates that not all columns are visible, making it challenging to obtain a complete overview.

```python
df = pd.read_csv('data-week-3.csv')
df.head()
```
To display all columns, we can transpose the DataFrame, converting rows into columns and vice versa.
```python
df.head().T
```
Standardizing Column Names and Values
We observe inconsistencies in the data. To address this, we will standardize the column names and values similarly to our previous projects.

```python
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.head().T
```
Verifying Column Data Integrity
Now that all column names are uniform, we check the data types of the columns.

```python
df.dtypes
```
Notably, the seniorcitizen column is represented as integers (0 or 1), while totalcharges is classified as an object but should be numeric.

```python
df.totalcharges
```

To convert the totalcharges column to numeric, we must handle non-numeric entries.
```python
tc = pd.to_numeric(df.totalcharges, errors='coerce')
```
This conversion will replace unparseable strings with NaN. We can then check for missing values.
```python
tc.isnull().sum()
df[tc.isnull()][['customerid', 'totalcharges']]
```

We can fill these missing values with 0, although this approach may not always be ideal.
```python
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.totalcharges.isnull().sum()
```
Preparing the Churn Variable
Finally, we examine the churn variable, which consists of values "yes" and "no." For machine learning classification tasks, we need to convert these values into numerical format, using 1 for "yes" and 0 for "no."
```python
df.churn.head()
```

We can achieve this transformation as follows:

```python
df.churn = (df.churn == 'yes').astype(int)
df.churn
```
After applying this transformation, the churn column is now numerically represented.



### Part 3

# Setting Up the Validation Framework

In this guide, we will set up a validation framework for machine learning using Scikit-Learn. The first step involves splitting your dataset into training, validation, and test sets using the `train_test_split` function from the `sklearn.model_selection` package.

## Importing the Function

Start by importing the necessary function:

```python
from sklearn.model_selection import train_test_split

# To view the documentation
train_test_split?
```

Data Splitting
The train_test_split function divides the dataframe into two parts: 80% for the training set and 20% for the test set. By setting random_state=1, we ensure that our results are reproducible.

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
len(df_full_train), len(df_test)

# Output: (5634, 1409)
```

Next, to create three sets—training, validation, and test—we perform a second split on the training data. This time, we will allocate 60% for training and 20% for validation. Since we are now dealing with 80% of the original data, we calculate the validation set size as 20% of 80%, which equals 25%. Thus, we will use test_size=0.25.

```python
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val)

# Output: (4225, 1409)
```

Now, both the validation set and the test set are of the same size.
```python
len(df_train), len(df_val), len(df_test)

# Output: (4225, 1409, 1409)
```

Resetting Indices
After splitting, the indices of the records will be shuffled. To reset the indices and ensure they are continuous, use the reset_index function with the drop=True parameter:
```python
df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
```

Extracting the Target Variable
The next step is to extract our target variable, ‘y’ (churn), from each dataset:
```python
y_full_train = df_full_train.churn.values
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values
```

To prevent the accidental use of the “churn” variable during model building, we should remove it from all four datasets. This can be done as follows:
```python
del df_full_train['churn']
del df_train['churn']
del df_val['churn']
del df_test['churn']
```

After completing these steps, the "churn" variable will be removed, allowing you to proceed with model building without the risk of unintended usage.

### Part 4

# Exploratory Data Analysis (EDA)

In this section, we will cover several key topics related to Exploratory Data Analysis (EDA):

## 1. Checking Missing Values

To determine if there are any missing values in the dataset `df_full_train`, we can use the following code snippet:

```python
df_full_train.isnull().sum()
```
Output
```python
customerid          0
gender              0
seniorcitizen       0
partner             0
dependents          0
tenure              0
phoneservice        0
multiplelines       0
internetservice     0
onlinesecurity      0
onlinebackup        0
deviceprotection    0
techsupport         0
streamingtv         0
streamingmovies     0
contract            0
paperlessbilling    0
paymentmethod       0
monthlycharges      0
totalcharges        0
churn               0
dtype: int64
```

The output indicates that there are no missing values in the dataset.

2. Analyzing the Target Variable (Churn)
To analyze the target variable churn, we can view its distribution using:

```python
df_full_train.churn.value_counts()
```

Output:

0    4113
1    1521
Name: churn, dtype: int64


This shows that out of 5634 customers, 1521 are classified as churning (dissatisfied), while 4113 are not churning (satisfied).

To calculate the churn rate, we can normalize the value counts:

```python
df_full_train.churn.value_counts(normalize=True)
```
Output:
```python
0    0.730032
1    0.269968
Name: churn, dtype: float64
```

The churn rate is approximately 27%. We can also compute the global churn rate using the mean function:
```python
global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate, 2)
```
Output:
```python
0.27
```

This value matches the churn rate calculated earlier, as the mean of binary values directly corresponds to the proportion of ones in the dataset.

3. Examining Numerical and Categorical Variables
To identify numerical and categorical variables in the dataset, we can use the dtypes function:
```python
numerical_vars = df.select_dtypes(include=['int64', 'float64'])
categorical_vars = df.select_dtypes(include=['object'])
```
Output:

```python
print("Numerical Variables:")
print(numerical_vars.columns)

print("\nCategorical Variables:")
print(categorical_vars.columns)
```
DataFrame Structure:

```python
df_full_train.dtypes
```
Output:
```python
customerid           object
gender               object
seniorcitizen         int64
partner              object
dependents           object
tenure                int64
phoneservice         object
multiplelines        object
internetservice      object
onlinesecurity       object
onlinebackup         object
deviceprotection     object
techsupport          object
streamingtv         object
streamingmovies      object
contract             object
paperlessbilling     object
paymentmethod        object
monthlycharges      float64
totalcharges        float64
churn                 int64
dtype: object
```

From the analysis, we identify three numerical variables: tenure, monthlycharges, and totalcharges. We will define our numerical and categorical columns accordingly.

Numerical Variables:
```python
numerical = ['tenure', 'monthlycharges', 'totalcharges']
```
Categorical Variables:
```python
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
```

To find the number of unique values for each categorical variable, we use the nunique() function:
```python
df_full_train[categorical].nunique()
```
Output:
```python
gender              2
seniorcitizen       2
partner             2
dependents          2
phoneservice        2
multiplelines       3
internetservice     3
onlinesecurity      3
onlinebackup        3
deviceprotection    3
techsupport         3
streamingtv         3
streamingmovies     3
contract            3
paperlessbilling     2
paymentmethod       4
dtype: int64
```
This output provides insight into the diversity of the categorical variables in the dataset.

### Part 5

# Feature Importance Analysis: Churn Rate and Risk Ratio

Feature importance analysis is a crucial part of exploratory data analysis (EDA), focusing on identifying the features that influence our target variable—customer churn.

## Churn Rate

We previously assessed the global churn rate. Now, we will delve into churn rates among specific demographics, such as gender.

### Female Customer Subset

To analyze female customers, we filter the dataset:

```python
female_customers = df_full_train[df_full_train.gender == 'female']
```

The churn rates for females and males can be compared to the global churn rate. The global churn rate is approximately 27%, with female customers exhibiting a churn rate of 27.7%, while male customers have a rate of 26.3%. This indicates a slightly higher likelihood of churn among females.

Churn Rate Calculation

```python
global_churn = df_full_train.churn.mean()  # Output: 0.269968
churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()  # Output: 0.276824
churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()  # Output: 0.263213
```
Churn Rate by Partnership Status
Next, we analyze churn rates based on whether customers have partners:
```python
df_full_train.partner.value_counts()
# Output:
# no     2932
# yes    2702
```
Customers with partners exhibit a significantly lower churn rate of 20.5% compared to the global rate, whereas those without partners show a churn rate of 33%.

Churn Rates Calculation by Partnership
```python
churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()  # Output: 0.205033
churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()  # Output: 0.329809
```
This suggests that the partnership variable is likely more influential for predicting churn than gender.

Risk Ratio
In machine learning and classification contexts, the risk ratio quantifies the likelihood of an event occurring in one group compared to another. It is particularly useful in customer churn analysis.

Definition and Interpretation
Risk Ratio: The ratio of the probability of an event in one group to that in another. A risk ratio greater than 1 indicates higher likelihood, while a ratio less than 1 indicates lower likelihood.
Application: Risk ratios can be employed to assess the impact of different features on churn.
Risk Ratio Calculation for Partnership Status
Calculating risk ratios for customers with and without partners yields:
```python
risk_no_partner = churn_no_partner / global_churn  # Output: 1.221
risk_partner = churn_partner / global_churn  # Output: 0.759
```

This reveals that customers without partners have a 22% higher churn risk, while those with partners have a 24% lower risk compared to the global churn rate.

Average Churn Rate Analysis by Group
To further investigate, we can calculate the average churn rates grouped by gender and other categorical variables:
```sql
SELECT
gender,
AVG(churn),
AVG(churn) - global_churn AS diff,
AVG(churn) / global_churn AS risk
FROM
df_full_train
GROUP BY
gender;
```
Results Overview
The average churn rates grouped by gender reveal:

```python
df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])
df_group['diff'] = df_group['mean'] - global_churn
df_group['risk'] = df_group['mean'] / global_churn
```
Gender Group Results
Gender	Mean	Count	Diff	Risk
Female	0.276824	2796	0.006856	1.025396
Male	0.263214	2838	-0.006755	0.974980
This table demonstrates how churn rates vary across gender groups. To broaden this analysis, we can repeat the calculations for all categorical columns, providing a more comprehensive understanding of churn dynamics across different customer segments.
```python
for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
```

Summary of Key Findings
Gender Impact: Female customers show a slightly higher churn rate than males.
Partnership Status: Customers with partners have significantly lower churn rates.
Risk Ratios: Analysis indicates that partnership status plays a more critical role in predicting churn compared to gender.
This structured approach to analyzing churn rates and risk ratios provides valuable insights for developing strategies aimed at reducing customer churn effectively.

### Part 6

# Feature Importance: Mutual Information

The risk ratio offers valuable insights into the significance of categorical variables, especially in relation to customer churn. For instance, when analyzing the "contract" variable, which includes options like "month-to-month," "one_year," and "two_years," it becomes evident that customers with a "month-to-month" contract have a higher likelihood of churning compared to those with a "two_years" contract. This indicates that the "contract" variable may be a critical factor in predicting churn. However, to fully grasp its relative importance compared to other variables, we need an effective method of comparison.

**Mutual Information** is a concept from information theory that helps in this regard. It measures how much information we can gain about one variable by knowing the value of another. A higher mutual information score signifies a greater understanding of churn based on the observation of another variable. This allows us to evaluate the significance of categorical variables and their respective values in predicting churn, facilitating comparisons among them.

To compute the mutual information between the churn variable and other categorical variables, we can use the `mutual_info_score` function from the `sklearn.metrics` library. Here are some examples:

```python
from sklearn.metrics import mutual_info_score

# Mutual information between churn and contract
mutual_info_score(df_full_train.churn, df_full_train.contract)
# Output: 0.0983203874041556

# Mutual information between churn and gender
mutual_info_score(df_full_train.churn, df_full_train.gender)
# Output: 0.0001174846211139946

# Mutual information between churn and partner
mutual_info_score(df_full_train.churn, df_full_train.partner)
# Output: 0.009967689095399745
```

The intuition behind these scores is to understand how much we learn about churn from each variable. For example, the low mutual information score for gender suggests that it is not particularly informative regarding churn.

To evaluate the relative importance of all features, we can apply the mutual information metric across all categorical variables. Since mutual_info_score requires two arguments, we define a helper function, mutual_info_churn_score, to compute the scores column-wise:

```python
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
```
The output is as follows:
```python
gender              0.000117
seniorcitizen       0.009410
partner             0.009968
dependents          0.012346
phoneservice        0.000229
multiplelines       0.000857
internetservice     0.055868
onlinesecurity      0.063085
onlinebackup        0.046923
deviceprotection    0.043453
techsupport         0.061032
streamingtv         0.031853
streamingmovies     0.031581
contract            0.098320
paperlessbilling    0.017589
paymentmethod       0.043210
dtype: float64
```

To prioritize the variables based on their mutual information scores, we can sort them in descending order:

```python
mi.sort_values(ascending=False)
```

The sorted output will be:
```python
contract            0.098320
onlinesecurity      0.063085
techsupport         0.061032
internetservice     0.055868
onlinebackup        0.046923
deviceprotection    0.043453
paymentmethod       0.043210
streamingtv         0.031853
streamingmovies     0.031581
paperlessbilling    0.017589
dependents          0.012346
partner             0.009968
seniorcitizen       0.009410
multiplelines       0.000857
phoneservice        0.000229
gender              0.000117
dtype: float64
```

### Part 7


# Feature Importance Analysis

## 1. Feature Importance: Mutual Information

Mutual information quantifies the amount of information gained about one variable when observing another. This is particularly useful for analyzing categorical variables, such as "contract" types—where customers with a "month-to-month" contract are more likely to churn than those with a "two_years" contract. While this indicates that the "contract" variable is important for predicting churn, mutual information allows for comparison across multiple variables.

### Implementation

Using `mutual_info_score` from `sklearn`, we can compute the mutual information between the target variable (churn) and each categorical variable.

```python
from sklearn.metrics import mutual_info_score

# Calculate mutual information for categorical variables
mi_contract = mutual_info_score(df_full_train.churn, df_full_train.contract)
mi_gender = mutual_info_score(df_full_train.churn, df_full_train.gender)
mi_partner = mutual_info_score(df_full_train.churn, df_full_train.partner)
```

Example Outputs
Contract: 0.09832
Gender: 0.000117
Partner: 0.009968
Summary of Mutual Information Scores
```python
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi_sorted = mi.sort_values(ascending=False)
```
Sorted Output
Contract: 0.098320
Online Security: 0.063085
Tech Support: 0.061032
Internet Service: 0.055868
Online Backup: 0.046923
2. Feature Importance: Correlation
For numerical variables, Pearson's correlation coefficient measures the linear relationship between two variables, with values ranging from -1 to 1. The absolute value of the correlation coefficient indicates the strength of the relationship:

Low correlation: 
0.0<∣𝑟∣<0.2
0.0<∣r∣<0.2
Moderate correlation: 
0.2<∣𝑟∣<0.5
0.2<∣r∣<0.5
Strong correlation: 
0.6<∣𝑟∣<1.0
0.6<∣r∣<1.0
Implementation
To calculate the correlation between numerical variables and churn, we can use the corrwith function:
```python
correlation = df_full_train[numerical].corrwith(df_full_train.churn)
```
Example Outputs

tenure           -0.351885
monthlycharges    0.196805
totalcharges     -0.196353


Absolute Values of Correlation Coefficients
To focus on overall impact, we can sort the absolute values:
```python
abs_correlation = df_full_train[numerical].corrwith(df_full_train.churn).abs()
```

Output
Tenure: 0.351885
Monthly Charges: 0.196805
Total Charges: 0.196353
Churn Analysis Based on Tenure
Examining churn rates based on tenure provides further insights:
```python
df_full_train[df_full_train.tenure <= 2].churn.mean()  # Output: ~0.595
df_full_train[df_full_train.tenure > 2].churn.mean()   # Output: ~0.224
df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()  # Output: ~0.399
df_full_train[df_full_train.tenure > 12].churn.mean()  # Output: ~0.176
```

Churn Analysis Based on Monthly Charges
Similarly, we analyze churn rates concerning monthly charges:

```python
df_full_train[df_full_train.monthlycharges <= 20].churn.mean()  # Output: ~0.088
df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean()  # Output: ~0.183
df_full_train[df_full_train.monthlycharges > 50].churn.mean()  # Output: ~0.325
```
### Part 8

# One-Hot Encoding in Machine Learning

One-hot encoding is a crucial technique in machine learning used to convert categorical data (non-numeric) into a numeric format suitable for various algorithms, especially classification and regression models. This method is essential for algorithms that require numerical input, and Scikit-Learn, a popular Python library, offers effective tools for this task.

## Problem Statement
Categorical data, such as colors ("red," "green," "blue"), cannot be directly utilized by most machine learning algorithms due to their requirement for numerical input. One-hot encoding addresses this limitation by transforming categorical data into binary vectors.

## How It Works
For each categorical feature, one-hot encoding generates a new binary (0 or 1) feature for each category within that feature. For instance, if we have a "color" feature, it would create three binary features: "IsRed," "IsGreen," and "IsBlue." An observation belonging to the "red" category would result in "IsRed" being set to 1, while the others would be 0.

## Implementing One-Hot Encoding with Scikit-Learn
To demonstrate how to encode categorical features using Scikit-Learn, we can use the `DictVectorizer` class:

1. **Transforming Data to Dictionary Format**: 
   We convert the DataFrame into a list of dictionaries:
   
```python
   dicts = df_train[['gender', 'contract']].iloc[:100].to_dict(orient='records')
```

Creating a DictVectorizer Instance:
```python
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
```
Fitting the DictVectorizer:
```python
dv.fit(dicts)
```

Transforming the Dictionaries: By default, DictVectorizer creates a sparse matrix. To obtain a dense array, set sparse=False:
```python
dv = DictVectorizer(sparse=False)
dv.fit(dicts)
X_train = dv.transform(dicts)
```
Feature Names: To retrieve the feature names:
```python
dv.get_feature_names_out()
```

Transforming Validation Data: For validation data, reuse the same DictVectorizer instance to maintain consistency:
```python
val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
```
### Part 9

# Logistic Regression

Logistic regression is primarily applied to classification problems, which can be divided into binary and multi-class categories. In binary classification, the target variable \( y \) belongs to one of two classes: 0 or 1. These classes are often referred to as “negative” and “positive” and represent mutually exclusive outcomes. For instance, in churn prediction, the classes could be “no churn” and “churn.” Similarly, in email classification, the classes could be “no spam” and “spam.”

In this context, the function \( g(x_i) \) outputs a value between 0 and 1, which can be interpreted as the probability of \( x_i \) belonging to the positive class.

### Mathematical Formulation

The formulas for linear regression and logistic regression are as follows:

- **Linear Regression**: 
  \[
  g(x_i) = w_0 + w^T x_i \quad \text{(outputs a number in the range } -\infty \text{ to } \infty \text{)}
  \]
  
- **Logistic Regression**: 
  \[
  g(x_i) = \text{SIGMOID}(w_0 + w^T x_i) \quad \text{(outputs a number in the range } 0 \text{ to } 1\text{)}
  \]

The sigmoid function is defined as:
\[
\text{sigmoid}(z) = \frac{1}{1 + \exp(-z)}
\]
This function maps any real number \( z \) to a value between 0 and 1, making it suitable for modeling probabilities in logistic regression.

### Implementation of Sigmoid Function

To demonstrate the sigmoid function, we can create an array of 51 values between -7 and 7 using `np.linspace(-7, 7, 51)`, which serves as our \( z \):

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-7, 7, 51)
```

When we apply the sigmoid function to our array z, we get:

```python
sigmoid(z)
```
This results in a new array that transforms the values of 𝑧 into probabilities.

Visualizing the Sigmoid Function
To visualize the sigmoid function, we can plot the values of 𝑧 against their corresponding sigmoid outputs:

```python
import matplotlib.pyplot as plt

plt.plot(z, sigmoid(z))
```
Comparing Linear and Logistic Regression
At the end of this article, both implementations will be presented for comparison. The main distinction between linear and logistic regression lies in the application of the sigmoid function to the output of linear regression, converting it into a probability value between 0 and 1.

Linear Regression Implementation
```python
def linear_regression(x_i):
    result = w_0
    for j in range(len(w)):
        result += x_i[j] * w[j]
    return result

```
Logistic Regression Implementation

```python
def logistic_regression(x_i):
    score = w_0
    for j in range(len(w)):
        score += x_i[j] * w[j]
    result = sigmoid(score)
    return result
```
Both linear regression and logistic regression are considered linear models, as the dot product in linear algebra acts as a linear operator. These models are efficient to use and quick to train.

### Part 10

# Training Logistic Regression with Scikit-Learn

## Introduction
Training a logistic regression model using Scikit-Learn involves a process similar to that of training a linear regression model. This document outlines the steps involved in training the model, applying it to validation data, and calculating its accuracy.

## Model Training
To begin, you need to import the `LogisticRegression` class and initialize your model:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

Model Coefficients
After training the model, you can access the weights (coefficients) using the coef_ attribute:
```python
model.coef_
```
This returns a 2D array, but you can extract the weight vector by indexing the first row:
```python
model.coef_[0].round(3)
```
Model Intercept
You can also access the bias term (intercept) using the intercept_ attribute:

```python
model.intercept_[0]
```

Model Predictions
Now that the model is trained, you can test it on the training data:
```python
model.predict(X_train)
```

The model outputs binary predictions (0 or 1), indicating whether an instance represents "not churn" or "churn."

Probability Predictions
For soft predictions, use the predict_proba method, which provides probabilities for both classes:
```python
model.predict_proba(X_train)
```

This output contains two columns: the first for the probability of the negative class (0) and the second for the positive class (1). For churn prediction, focus on the second column, which indicates the probability of churn. You can select a threshold to classify individuals as churned or not:
```python
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = y_pred > 0.5
```
Applying the Model to Validation Data
Using the model, you can apply it to the validation dataset and identify potential churn customers:
```python
df_val[churn_decision]
```
This selection highlights individuals predicted to churn based on the chosen threshold.

Calculate Accuracy
Finally, to evaluate the model's performance, calculate its accuracy:

Convert predicted probabilities to binary predictions based on the threshold:
```python
churn_decision.astype(int)
```
Compare the predicted values with actual values (y_val) to determine accuracy:
```python
(y_val == churn_decision).mean()
```

### Part 11

# Model Interpretation and Coefficients in Logistic Regression

## Understanding Model Coefficients

To interpret a logistic regression model, we need to examine the coefficients assigned to each feature. These coefficients indicate the weight of each feature's contribution to the model's prediction.

1. **Feature Extraction**  
   We extract feature names and their corresponding coefficients from the trained model. 

```python
   feature_names = dv.get_feature_names_out()
   coefficients = model.coef_[0].round(3)
```

Example Output:
```python
Features: ['contract=month-to-month', 'contract=one_year', ..., 'totalcharges']
Coefficients: [0.475, -0.175, ..., 0.0]
```
Combining Features with Coefficients
We can pair each feature with its coefficient using the zip function.
```python
feature_coeff_pairs = list(zip(feature_names, coefficients))
```
Example Output:
```python
[('contract=month-to-month', 0.475), ('contract=one_year', -0.175), ...]
```
Training a Smaller Model
To simplify the model, we can select a subset of features. For instance, we might choose contract, tenure, and monthlycharges.

```python
small_features = ['contract', 'tenure', 'monthlycharges']
```
Example of Data Subset:
```python
| contract         | tenure | monthlycharges |
|------------------|--------|----------------|
| two_year         | 72     | 115.50         |
| month-to-month   | 10     | 95.25          |
| ...              | ...    | ...            |
```

Transforming Data
We transform our selected features into a format suitable for the model.

```python
dicts_train_small = df_train[small_features].to_dict(orient='records')
dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)
```
Output Features:

```python
['contract=month-to-month', 'contract=one_year', 'contract=two_year', 'monthlycharges', 'tenure']
```
Training the Smaller Model
We train the logistic regression model using the smaller feature set.

```python
X_train_small = dv_small.transform(dicts_train_small)
model_small = LogisticRegression()
model_small.fit(X_train_small, y_train)
```
Example Coefficients:
```python
{'contract=month-to-month': 0.97, 'contract=one_year': -0.025, ..., 'tenure': -0.036}
```

Scoring a Customer
To assess a customer's likelihood of churning, we calculate their score using the model's coefficients.

Example Calculation:
For a customer with a monthly contract, monthly charges of $50, and a tenure of 5 months, the score is calculated as follows:
```python
score = -2.47 + (1 * 0.97) + (50 * 0.027) + (5 * -0.036)
```
Final Score Output:
```python
Score: -0.33
Probability of Churning: sigmoid(-0.33) = 41.8%
```
Additional Examples:

For a customer with a score of approximately 0, indicating a 50% churn probability:
```python
score = -2.47 + 0.97 + (60 * 0.027) + (1 * -0.036)
Probability: sigmoid(score) = 52.1%
```

For another customer:
```python
score = -2.47 + (-0.949) + (30 * 0.027) + (24 * -0.036)
Probability: sigmoid(score) = 3%
```

### Part 12

# Customer Churn Prediction Using Logistic Regression

In this analysis, we aim to train a logistic regression model using the `df_full_train` dataset, which contains various customer attributes related to their subscription and service usage.

## Dataset Overview

The dataset includes the following columns:
- `customerid`: Unique identifier for each customer.
- `gender`: Gender of the customer.
- `seniorcitizen`: Indicates if the customer is a senior citizen (1 for yes, 0 for no).
- `partner`: Indicates if the customer has a partner (yes/no).
- `dependents`: Indicates if the customer has dependents (yes/no).
- `tenure`: Number of months the customer has been with the service.
- `phoneservice`: Indicates if the customer has a phone service (yes/no).
- `multiplelines`: Indicates if the customer has multiple lines (yes/no).
- `internetservice`: Type of internet service (DSL, Fiber optic, or no service).
- Other service-related features...
- `monthlycharges`: Monthly charges for the customer.
- `totalcharges`: Total charges incurred by the customer.
- `churn`: Target variable indicating if the customer has churned (1) or not (0).

## Preparing Data for Model Training

We start by converting the relevant columns into a dictionary format, which will be used to create the feature matrix for the model.

```python
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
```

Feature Matrix Creation
Next, we utilize the DictVectorizer to transform our dictionary into a feature matrix X_full_train, while y_full_train contains the target variable values.
```python
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
y_full_train = df_full_train.churn.values
```

Model Training
We then initialize the logistic regression model and fit it to our training data.
```python
model = LogisticRegression()
model.fit(X_full_train, y_full_train)
```
Model Evaluation on Test Data
For the test data, we repeat the same preprocessing steps and make predictions.
```python
dicts_test = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)
y_pred = model.predict_proba(X_test)[:, 1]
```

Accuracy Calculation
To assess the model's performance, we compute the accuracy based on our predictions.
```python
churn_decision = (y_pred >= 0.5)
accuracy = (churn_decision == y_test).mean()
```
The model achieved an accuracy of approximately 81.5% on the test data, indicating a slight improvement over the validation dataset. Consistency in model performance across different datasets is crucial for assessing generalization.

Real-Time Prediction Deployment
Imagine deploying the logistic regression model on a website. When a customer provides their information, it is sent as a dictionary to the server. The model computes the probability of churn, allowing real-time predictions to facilitate proactive customer engagement.

Sample Customer Analysis
We can examine individual customer predictions. For example, consider the following customer:
```python
customer = dicts_test[10]
```

Transforming this customer's data to the feature matrix yields:
```python
X_small = dv.transform([customer])
```
The model predicts a 40.6% probability of churn for this customer, leading us to assume they are not likely to churn.

Actual Value Comparison
To verify:
```python
actual_value = y_test[10]

```

The actual value confirms our prediction as correct (not churning).

Predicting a Likely Churner
Testing another customer, we observe:
```python
customer = dicts_test[-1]
X_small = dv.transform([customer])
```

The model indicates a 59.7% chance of churn, suggesting this customer is likely to leave.

Final Verification
Checking the actual churn status:
```python
actual_value = y_test[-1]
```
