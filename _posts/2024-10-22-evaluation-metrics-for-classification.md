---
date: 2024-10-22 09:29:50
layout: post
title: "Evaluation Metrics for Classification"
subtitle: "Understanding the Key Metrics for Evaluating Classification Models"
description: "This post provides an in-depth overview of the most important evaluation metrics for classification models, including accuracy, precision, recall, F1-score, and AUC-ROC, and explains how each metric can be used to assess model performance in various machine learning tasks."
image: https://github.com/user-attachments/assets/54f58b9f-47c9-4455-9a22-9f676613e72d
optimized_image: https://github.com/user-attachments/assets/54f58b9f-47c9-4455-9a22-9f676613e72d
category: Medium
tags: Machine Learning
author: Dimas
paginate: True
---

# ML Zoomcamp 2024: Evaluation Metrics for Classification

## Part 1

# Churn Prediction Model Using Logistic Regression

## Overview
This script covers the essential steps for building a churn prediction model using Logistic Regression. We will:
1. Import necessary libraries.
2. Prepare the data by cleaning and transforming it.
3. Split the dataset into training, validation, and test sets.
4. Train a Logistic Regression model on the training data.
5. Validate the model on the validation data and evaluate its performance.

---

### 1. Necessary Imports

We start by importing the required libraries: Pandas for data manipulation, NumPy for numerical operations, Matplotlib for visualization, and several modules from Scikit-Learn for machine learning tasks.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
```

2. Data Preparation
Load the dataset and standardize column names by converting them to lowercase and replacing spaces with underscores.
Identify categorical columns and ensure that the 'totalcharges' column is correctly converted to a numerical format.
Convert the 'churn' column into a binary format, where 'yes' becomes 1 and 'no' becomes 0.


```python
df = pd.read_csv('data-week-3.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.churn = (df.churn == 'yes').astype(int)
```
3. Data Splitting
We split the dataset into:

60% for training,
20% for validation, and
20% for testing.
The indices are reset to ensure continuous indexing, and the 'churn' column is separated as the target variable.

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']
```
4. Feature Preparation
We define two lists: one for numerical features and one for categorical features.

```python
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']
```

5. Vectorization and Model Training
We use the DictVectorizer to transform the categorical and numerical columns into vectors, and then train the Logistic Regression model.
```python
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)
```

6. Model Validation
We transform the validation dataset similarly, predict churn probabilities, and evaluate the accuracy of the model.
```python
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
accuracy = (y_val == churn_decision).mean()

# Output the accuracy
accuracy
```

# Model Accuracy and Evaluation

## Part 2

## Accuracy and Dummy Model

In the previous analysis, our model achieved **80% accuracy** on the validation data, but we need to evaluate whether this is a good result.

**Accuracy** measures the proportion of correct predictions made by the model. In this case, a prediction was considered correct if a customer's predicted value was above the 0.5 threshold, meaning they were classified as "churn." Otherwise, they were classified as "non-churn."

Out of **1409 customers** in the validation dataset, the model correctly predicted the churn status for **1132 customers**, resulting in an accuracy of **80%**:
```python
len(y_val)  # Output: 1409
(y_val == churn_decision).sum()  # Output: 1132
1132 / 1409  # Output: 0.8034
(y_val == churn_decision).mean()  # Output: 0.8034
```

Evaluating Model on Different Thresholds
We can test if 0.5 is the best threshold for our model by experimenting with various threshold values. We can generate a range of values using NumPy's linspace function and evaluate the model at each threshold to find the one that maximizes accuracy.

```python
import numpy as np

thresholds = np.linspace(0, 1, 21)  # Generate thresholds from 0 to 1
scores = []

for t in thresholds:
    churn_decision = (y_pred >= t)
    score = (y_val == churn_decision).mean()
    print('%.2f %.3f' % (t, score))
    scores.append(score)
```
```python
0.00 0.274
0.05 0.509
0.10 0.591
0.15 0.666
0.20 0.710
0.25 0.739
0.30 0.760
0.35 0.772
0.40 0.785
0.45 0.793
0.50 0.803
0.55 0.801
0.60 0.795
0.65 0.786
0.70 0.766
0.75 0.744
0.80 0.735
0.85 0.726
0.90 0.726
0.95 0.726
1.00 0.726
```
The model performs best at a 0.5 threshold, confirming it is the optimal choice for this context. We can visualize how accuracy changes with different thresholds:
```python
import matplotlib.pyplot as plt

plt.plot(thresholds, scores)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy at Different Thresholds')
plt.show()
```
Scikit-learn Accuracy
We can simplify this evaluation by using Scikit-Learn's accuracy_score function:
```python
from sklearn.metrics import accuracy_score

thresholds = np.linspace(0, 1, 21)
scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)
```
```python
0.00 0.274
0.05 0.509
0.10 0.591
0.15 0.666
0.20 0.710
0.25 0.739
0.30 0.760
0.35 0.772
0.40 0.785
0.45 0.793
0.50 0.803
0.55 0.801
0.60 0.795
0.65 0.786
0.70 0.766
0.75 0.744
0.80 0.735
0.85 0.726
0.90 0.726
0.95 0.726
1.00 0.726
```
Dummy Model Accuracy
The dummy model (which predicts all customers as non-churners) achieves an accuracy of 73%, even though it doesn’t distinguish between churning and non-churning customers. This reveals the limitations of accuracy as a metric, especially with imbalanced datasets.
```python
from collections import Counter

# Distribution of predictions
Counter(y_pred >= 1.0)  # Output: Counter({False: 1409})

# Distribution of actual values
Counter(y_val)  # Output: Counter({0: 1023, 1: 386})

1023 / 1409  # Output: 0.7260468417317246
y_val.mean()  # Output: 0.2739531582682754
1 - y_val.mean()  # Output: 0.7260468417317246
```
With only 27% churners, accuracy can be deceptive, as predicting everyone as non-churners already gives a high score.

Alternative Metrics for Imbalanced Datasets
In cases like this, it’s important to consider other metrics:

Precision: Measures the proportion of true positives among all positive predictions.
Recall: Measures the proportion of true positives among all actual positives.
F1-Score: The harmonic mean of precision and recall.
AUC-ROC: Measures the ability to distinguish between classes at various thresholds.
Choosing the best metric depends on the problem's goals and whether minimizing false positives or false negatives is more important.

## Part 3

# Confusion Matrix and Types of Errors

## Overview
In this section, we will explore the **confusion matrix**, a critical tool for evaluating the performance of binary classification models. The confusion matrix provides a breakdown of how a model's predictions align with actual outcomes, revealing the types of correct and incorrect decisions made.

The confusion matrix is especially useful when dealing with **class imbalance**, as it gives us a more detailed view of model performance than accuracy alone.

## Components of the Confusion Matrix
The confusion matrix is structured around four key metrics:

- **True Positives (TP):** Correctly predicted positive class (e.g., churn customers).
- **True Negatives (TN):** Correctly predicted negative class (e.g., non-churn customers).
- **False Positives (FP):** Incorrectly predicted positive class when the actual class is negative (**Type I error**).
- **False Negatives (FN):** Incorrectly predicted negative class when the actual class is positive (**Type II error**).

### Example Table Layout
| Prediction vs Actual | No Churn (Negative) | Churn (Positive) |
|----------------------|---------------------|------------------|
| **Predicted No Churn** | True Negative (TN) | False Negative (FN) |
| **Predicted Churn**   | False Positive (FP) | True Positive (TP) |

## Calculating the Confusion Matrix
Let’s implement a confusion matrix calculation using Python.

### Data Setup
We start by defining thresholds for predictions and grouping the data into actual positives and negatives:

```python
# True churners (actual positives)
actual_positive = (y_val == 1)

# True non-churners (actual negatives)
actual_negative = (y_val == 0)

# Prediction thresholds
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)
```

Logical Operations for Each Category
To find each category in the confusion matrix:
```python
# True Positives
tp = (predict_positive & actual_positive).sum()

# True Negatives
tn = (predict_negative & actual_negative).sum()

# False Positives
fp = (predict_positive & actual_negative).sum()

# False Negatives
fn = (predict_negative & actual_positive).sum()
```

Confusion Matrix Example
Arranging these values into a confusion matrix:
```python
import numpy as np

confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

confusion_matrix
```

Output:
```python
array([[922, 101],
       [176, 210]])
```

Accuracy Calculation
Accuracy is calculated by summing the correct predictions (True Positives + True Negatives) divided by the total predictions:
```python
accuracy = (tn + tp) / (tn + tp + fn + fp)
accuracy * 100  # Output: 80%
```

In our case, the accuracy is 80%, but the confusion matrix provides more context about the errors.

Relative Values in Confusion Matrix
To better understand the model’s performance, we can express these values as relative proportions:
```python
(confusion_matrix / confusion_matrix.sum()).round(2)
```

Output:
```python
array([[0.65, 0.07],
       [0.12, 0.15]])
```

Key Insights
False Positives (FP) result in unnecessary costs by targeting non-churning customers.
False Negatives (FN) cause financial loss by missing potential churners who leave without receiving an offer.
Both scenarios have a negative impact, and understanding the confusion matrix helps us strategize around these errors.

## Part 4

# Precision & Recall in Binary Classification

## Overview
Precision and recall are essential metrics used to evaluate binary classification models. While accuracy gives a broad view of performance, precision and recall provide deeper insights into a model's ability to handle positive predictions, especially in cases with class imbalance, such as predicting customer churn.

## Key Definitions

- **Precision** measures how many of the positive predictions made by the model were actually correct. It answers the question: "Of all the customers predicted to churn, how many truly churned?"
  
  **Formula:**
  \[
  \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
  \]

- **Recall** (also known as sensitivity) measures how many of the actual positive cases were correctly identified by the model. It answers the question: "Of all the customers who churned, how many were correctly predicted?"

  **Formula:**
  \[
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
  \]

In short:
- **Precision** focuses on how accurate positive predictions are.
- **Recall** emphasizes how well the model identifies all true positive cases.

## Confusion Matrix Overview

| Actual vs Prediction | Predicted Negative | Predicted Positive |
|----------------------|--------------------|--------------------|
| **Actual Negative**   | True Negative (TN) | False Positive (FP) |
| **Actual Positive**   | False Negative (FN)| True Positive (TP)  |

Both **precision** and **recall** are calculated based on the True Positives (TP), False Positives (FP), and False Negatives (FN) from the confusion matrix.

## Example Calculation

Let’s calculate **accuracy**, **precision**, and **recall** using a binary classification example in Python.

### Data Setup
Using the confusion matrix, we define the True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN):

```python
# Confusion matrix components
tp = 210
fp = 101
fn = 176
tn = 922
```

Accuracy Calculation
Accuracy measures the overall correctness of the model by calculating the proportion of correct predictions:

```python
precision = tp / (tp + fp)
precision
# Output: 0.6752 (67.52%)
```

Out of 311 predicted positive cases (TP + FP), 210 were actually going to churn, meaning 33% were incorrect predictions.


Recall Calculation
Recall quantifies the model's ability to find all actual positive cases:
```python
recall = tp / (tp + fn)
recall
# Output: 0.5440 (54.40%)
```

Out of 386 actual churn cases (TP + FN), the model failed to identify 46% of them.

Conclusion: Precision vs. Recall
While accuracy gives an overall picture, metrics like precision and recall are far more informative when dealing with class imbalance. In this example:

Precision (67.52%) shows that 33% of positive predictions were incorrect.
Recall (54.40%) indicates that 46% of churners were not detected.
These metrics provide insights into the trade-offs between correctly identifying customers at risk of churn and minimizing false positives (non-churners wrongly predicted as churners). In scenarios like churn prediction, where specific identification is critical, precision and recall offer a much more nuanced view of model performance than accuracy alone.

## Part 5

# ROC Curve (Receiver Operating Characteristic)

## Overview
The ROC Curve (Receiver Operating Characteristic) is an important tool used to evaluate binary classification models. It allows us to visualize the performance of a classifier by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) across various threshold settings. The curve illustrates the trade-off between sensitivity and specificity at different thresholds.

The **Area Under the ROC Curve (AUC-ROC)** provides a single score to assess the model's performance. A higher AUC indicates better model performance, with values closer to 1 representing stronger discriminative capabilities between positive and negative outcomes.

## ROC Curve Definitions
The ROC curve is derived from the confusion matrix, where:

- **TPR (True Positive Rate)**: Also called sensitivity or recall, it is calculated as:
```python
  tpr = tp / (tp + fn)
```

FPR (False Positive Rate): The proportion of false positives out of the total actual negatives
```python
fpr = fp / (fp + tn)
```

Example: Computing TPR and FPR
We can compute TPR and FPR for different threshold values and assess the model performance:

```python
thresholds = np.linspace(0, 1, 101)
scores = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, tn, fp, fn))

# Convert scores into a DataFrame
df_scores = pd.DataFrame(scores, columns=['threshold', 'tp', 'tn', 'fp', 'fn'])

# Compute TPR and FPR
df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
```

Visualizing ROC Curve
We can plot the ROC curve by plotting TPR against FPR across thresholds:

```python
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.legend()
plt.show()
```
Random Model Comparison
To assess the performance of a random classifier, we can simulate random predictions and compare:

```python
y_rand = np.random.uniform(0, 1, size=len(y_val))
df_rand = tpr_fpr_dataframe(y_val, y_rand)

# Plot ROC curve for the random model
plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR (Random)')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR (Random)')
plt.legend()
plt.show()
```
Using Threshold of 0.6
For a specific threshold, such as 0.6, we obtain the following TPR and FPR values:

TPR = 0.4
FPR = 0.05
These values help us decide whether the model meets the criteria for a given problem.

## Part 6

# ROC AUC – Area under the ROC Curve

## Overview

The Area Under the ROC Curve (AUC) is a valuable metric used to evaluate binary classification models. It measures how well the model can distinguish between positive and negative classes. AUC values range from 0.5 (random model) to 1.0 (perfect model). Generally, an AUC of 0.8 is considered good, 0.9 is excellent, while an AUC of 0.6 is poor.

## Calculation Using `sklearn`

To calculate the AUC, we use the `auc` function from `scikit-learn`:

```python
from sklearn.metrics import auc

# Example of calculating AUC using FPR and TPR
auc(fpr, tpr)
# Output: 0.843850505725819
```

Alternatively, you can use roc_auc_score to simplify the process:

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_val, y_pred)
# Output: 0.843850505725819
```

Example of AUC Calculation
Below is a step-by-step example that demonstrates how to calculate AUC:

```python
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(fpr, tpr)
# Output: 0.843850505725819
```

Interpretation of AUC
AUC represents the probability that a randomly selected positive example has a higher score than a randomly selected negative example. To illustrate this, let’s pick random positive and negative examples and compare their scores:
```python
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

import random
pos_ind = random.randint(0, len(pos) -1)
neg_ind = random.randint(0, len(neg) -1)

pos[pos_ind] > neg[neg_ind]
# Output: True
```
By repeating this process many times, we can compute the performance:

```python
n = 100000
success = 0

for i in range(n):
    pos_ind = random.randint(0, len(pos) -1)
    neg_ind = random.randint(0, len(neg) -1)
    if pos[pos_ind] > neg[neg_ind]:
        success += 1

success / n
# Output: 0.84389
```

This result is very close to the AUC score: roc_auc_score(y_val, y_pred) = 0.843850505725819.

AUC Calculation with NumPy
Instead of manually comparing random indices, we can simplify the process with NumPy:


```python
import numpy as np

n = 50000
np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()
# Output: 0.84646
```
## Part 7

# Cross-Validation and Parameter Tuning in Logistic Regression

### Cross-Validation Overview
Cross-validation is a method to evaluate a model's performance by splitting the dataset into different subsets. Typically, the dataset is divided into three parts: **training**, **validation**, and **testing**. For model tuning, the validation set helps find the optimal parameters, while the test set is reserved for final evaluation.

To implement cross-validation, we split the training and validation dataset into 'k' parts (e.g., `k = 3`). The model is trained on two parts and validated on the third, with the process repeated for different combinations. After each fold, the AUC (Area Under Curve) score is calculated.

### Example of K-Fold Cross-Validation
```python
from sklearn.model_selection import KFold

# Splitting the full dataset into training and validation sets
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
train_idx, val_idx = next(kfold.split(df_full_train))
df_train = df_full_train.iloc[train_idx]
df_val = df_full_train.iloc[val_idx]
```

Training and Prediction Functions
Below are the functions for training a logistic regression model and predicting the output.
```python
def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.fit_transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred
```
Implementing 10-Fold Cross-Validation
We calculate the AUC score for each fold:
```python
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

kfold = KFold(n_splits=10, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
```

Results from K-Fold Cross-Validation
AUC scores from 10 folds:
```python
# Output:
# [0.8479, 0.8410, 0.8557, 0.8333, 0.8262, 0.8342, 0.8412, 0.8186, 0.8452, 0.8621]
```

Mean and Standard Deviation
We compute the average AUC score and its spread:
```python
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))
# Output: 0.841 +- 0.012
```

Parameter Tuning: Logistic Regression's Regularization (C)
Parameter tuning involves adjusting the regularization strength (C) in the logistic regression model. Smaller C values indicate stronger regularization.
```python
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, model
```

Tuning C Parameter Across Folds
Iterating over different values of C:
```python
from sklearn.model_selection import KFold

for C in [0.001, 0.01, 0.1, 0.5, 1, 5, 10]:
    scores = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        y_train = df_train.churn.values
        y_val = df_val.churn.values
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
    
    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

```
Results for Various C Values
```python
# Output:
# C=0.001 0.826 +- 0.012
# C=0.01 0.840 +- 0.012
# C=0.1 0.841 +- 0.011
# C=0.5 0.841 +- 0.011
# C=1 0.840 +- 0.012
# C=5 0.841 +- 0.012
# C=10 0.841 +- 0.012
```

Using tqdm for Visual Progress
For a more interactive experience, use the tqdm package to visualize the progress.
```python
from tqdm.auto import tqdm

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    scores = []
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        y_train = df_train.churn.values
        y_val = df_val.churn.values
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
    
    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
```

### Conclusion

In this project, we successfully built a churn prediction model using logistic regression techniques. This process provided valuable insights into the essential steps in developing machine learning models and highlighted the importance of understanding the data used. Below are the key learning points from each stage:

1. **Data Preprocessing**: The data cleaning and transformation process is crucial. Addressing missing values, encoding categorical variables, and performing normalization or standardization are important initial steps to ensure data quality before building the model.

2. **Model Development**: Choosing the right model (in this case, logistic regression) allows us to predict the probability of churn with clear interpretation. This model provides insights into the factors contributing to customers' decisions to leave the service.

3. **Model Evaluation**: Using evaluation metrics such as accuracy, precision, recall, and F1-score helps us understand the model's performance. This is essential for assessing how well the model can predict churn and identifying areas for improvement.

4. **Cross-Validation**: Implementing cross-validation techniques provides a deeper understanding of the model's stability and reliability. By splitting the data into several subsets, we can ensure that the model does not overfit and has good generalization on unseen data.

5. **Result Interpretation**: It is important to interpret the results of the model correctly. Understanding the regression coefficients and how each variable influences the likelihood of churn can help the business team make better decisions in customer retention strategies.

6. **Strategy Implementation**: Finally, the results from the predictive model can be integrated into marketing and customer service strategies. By identifying customers at high risk of churn, companies can implement appropriate interventions to improve customer retention.
