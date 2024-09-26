---
date: 2024-09-26 15:10:43
layout: post
title: "Introduction to Machine Learning"
subtitle: Delving into features, targets, and the steps involved in Machine Learning models, as well as the importance of model selection and development environment for effective implementation.
description: This article explores the fundamentals of Machine Learning, explaining concepts, terminology, and the processes involved in training and using models to predict unknown variables using available data.
image: https://github.com/user-attachments/assets/54f58b9f-47c9-4455-9a22-9f676613e72d
optimized_image: https://github.com/user-attachments/assets/54f58b9f-47c9-4455-9a22-9f676613e72d
category: Medium
tags: Machine Learning
author: Dimas
paginate: True
---


The concept of Machine Learning (ML) is illustrated through an example of predicting car prices. Data, including features such as year and mileage, is used by the ML model to learn and identify patterns. The target variable, in this case, is the car’s price.

New data, which lacks the target variable, is then provided to the model to predict the price.

In summary, ML involves extracting patterns from data, which is categorized into two types:

- **Features**: Information about the object.
- **Target**: The property to be predicted for unseen objects.

New feature values are inputted into the model, which generates predictions based on the patterns it has learned. This is an overview of what has been learned from the ML course by Alexey Grigorev (ML Zoomcamp). All images in this post are sourced from the course material. Images in other posts may also be derived from this material.

## What is Machine Learning?

Machine Learning (ML) is explained as the process of training a model using features and target information to predict unknown object targets. In other words, ML is about extracting patterns from data, which includes features and targets.

### Key Terms in ML

- **Features**: What is known about an object. In this example, it refers to the characteristics of a car. A feature represents an object’s attribute in various forms, such as numbers, strings, or more complex formats (e.g., location information).
- **Target**: The aspect to be predicted. The term “label” is also used in some sources. During training, a labeled dataset is used since the target is known. For example, datasets of cars with known prices are used to predict prices for other cars with unknown values.
- **Model**: The result of the training process, which encompasses the patterns learned from the training data. This model is utilized later to make predictions about the target variable based on the features of an unknown object.

### Training and Using a Model

- **Train a Model**: The training process involves the extraction of patterns from the provided training data. In simpler terms, features are combined with the target, resulting in the creation of the model.
- **Use a Model**: Training alone does not make the model useful. The benefit is realized through its application. By applying the trained model to new data (without targets), predictions for the missing information (e.g., price) are obtained. Therefore, features are used during prediction, while the trained model is applied to generate predictions for the target variable.

## What Did I Learn?

### Part 1: What is Machine Learning

#### Definition

Machine Learning (ML) is a process where models are trained using data to predict outcomes. The main components involved in ML are:

- **Features**: Attributes or characteristics of the objects (e.g., year, mileage of cars).
- **Target**: The variable to be predicted (e.g., car price).

#### How ML Works

- **Model Training**: The model learns patterns from the data using features and targets.
- **Model Usage**: New data is input into the trained model to predict outcomes.

#### Key Components

- **Model**: Result of the training process containing learned patterns.
- **Prediction**: The process where the trained model generates output for unseen data.

### Part 2: Machine Learning vs Rule-Based Systems

#### Rule-Based Systems

- Depend on predefined characteristics (like keywords).
- Require continuous updates, becoming complex over time.

#### Machine Learning Approach

- **Data Collection**: Gather examples of spam and non-spam emails.
- **Feature Definition**: Define features and label emails based on their source.
- **Model Training**: Use algorithms to build a predictive model based on encoded emails.
- **Model Application**: Apply the model to classify new emails based on probability thresholds.

#### Comparison

- **Maintenance**: Rule-based systems require constant adjustments, while ML models adapt to new data through training.

### Part 3: Supervised Machine Learning Overview

#### Definition

In Supervised Machine Learning (SML), models learn from labeled data, with:

- **Feature Matrix (X)**: Two-dimensional array of features.
- **Target Variable (y)**: One-dimensional array of outcomes.

#### Types of SML Problems

- **Regression**: Predicting continuous values (e.g., car prices).
- **Ranking**: Scores associated with items (e.g., recommender systems).
- **Classification**: Predicting categories (e.g., spam or not).
  - **Binary Classification**: Two categories.
  - **Multiclass Classification**: More than two categories.

### Part 4: CRISP-DM — Cross-Industry Standard Process for Data Mining

#### Overview

CRISP-DM is an iterative process model for data mining, consisting of six phases:

1. **Business Understanding**: Identify the problem and requirements.
2. **Data Understanding**: Analyze available data.
3. **Data Preparation**: Clean and format data for modeling.
4. **Modeling**: Train various models and select the best.
5. **Evaluation**: Assess model performance against business goals.
6. **Deployment**: Implement the model in a production environment.

The process may require revisiting previous steps based on feedback and evaluation results.

### Part 5: Model Selection Process

#### Overview

Steps:

1. **Split the Dataset**: Divide into training (60%), validation (20%), and test (20%) sets.
2. **Train the Models**: Use the training dataset for training.
3. **Evaluate the Models**: Assess model performance on the validation dataset.
4. **Select the Best Model**: Choose the model with the best validation performance.
5. **Apply the Best Model**: Test on the unseen test dataset.
6. **Compare Performance Metrics**: Ensure the model generalizes well by comparing validation and test performance.

#### Multiple Comparison Problem (MCP)

To mitigate MCP, the test set verifies that the selected model truly performs well, rather than relying solely on validation results.

### Part 6: Setting Up the Environment

#### Requirements

To prepare your environment, you’ll need the following:

- Python 3.10 (Note: Videos utilize Python 3.8)
- NumPy, Pandas, and Scikit-Learn (ensure you have the latest versions)
- Matplotlib and Seaborn for data visualization
- Jupyter Notebooks for interactive computing
- Ubuntu 22.04 on AWS

For a comprehensive guide on configuring your environment on an AWS EC2 instance running Ubuntu 22.04, refer to this video.

Make sure to adjust the instructions to clone the relevant repository instead of the MLOps one. These instructions can also be adapted for setting up a local Ubuntu environment.

#### Note for WSL

Most instructions from the video are applicable to Windows Subsystem for Linux (WSL) as well. For Docker, simply install Docker Desktop on Windows; it will automatically be used in WSL, so there’s no need to install docker.io.

### Anaconda and Conda

It is recommended to use Anaconda or Miniconda:

- **Anaconda**: This distribution includes everything you need for data science, including a variety of libraries and tools.
- **Miniconda**: A lighter version that contains only the essential components to manage Python environments and packages.

Make sure to follow the installation instructions provided on their respective websites to set up your environment correctly.

### Part 7: NumPy: A Comprehensive Overview

NumPy is a highly regarded library in Python that serves as a cornerstone for numerical computing. Its primary strength lies in its ability to facilitate the creation and manipulation of multi-dimensional arrays, along with providing a rich set of mathematical functions. This makes it an indispensable tool for a wide range of applications, including data analysis, scientific computing, and machine learning.

#### Creating Arrays

One of the key features of NumPy is its flexibility in creating arrays. Users can generate NumPy arrays in various ways:

- **From Python Lists**: Creating an array from a standard Python list is straightforward with the `np.array()` function. For instance:

    ```python
    import numpy as np
    arr = np.array([1, 2, 3])
    ```

- **Using Built-in Functions**: NumPy also offers a variety of built-in functions such as `np.zeros()`, `np.ones()`, and `np.arange()` to initialize arrays. For example:

    ```python
    zeros_array = np.zeros((2, 3))  # Creates a 2x3 array filled with zeros.
    ones_array = np.ones((3, 2))     # Creates a 3x2 array filled with ones.
    range_array = np.arange(10)       # Generates an array with values from 0 to 9.
    ```

- **Using Random Generation**: The `numpy.random` module allows for the generation of arrays filled with random values, which is especially useful for testing and simulations. For example:

    ```python
    random_integers = np.random.randint(10, size=5)        # Generates a 1D array of random integers.
    random_floats = np.random.random((3, 4))               # Creates a 2D array of random floats.
    random_normal = np.random.normal(size=(2, 3, 2))       # Produces a 3D array from the standard normal distribution.
    ```

#### Element-wise Operations

One of the most powerful features of NumPy is its support for element-wise operations. This capability allows users to perform mathematical operations on arrays without the need for explicit loops, greatly enhancing efficiency. This includes:

- **Addition and Subtraction**: Simple operations like addition and subtraction can be performed directly between arrays:

    ```python
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    sum_array = arr1 + arr2  # Element-wise addition.
    ```

- **Multiplication and Division**: Similarly, multiplication and division can be applied in the same manner:

    ```python
    product_array = arr1 * arr2  # Element-wise multiplication.
    division_array = arr2 / arr1   # Element-wise division.
    ```

- **Statistical Functions**: NumPy offers a range of built-in statistical functions, such as mean, median, and standard deviation, that can be applied to arrays:

    ```python
    mean_value = np.mean(arr1)         # Computes the mean.
    std_dev_value = np.std(arr2)       # Calculates the standard deviation.
    ```

### Conclusion

Machine Learning is a powerful tool for extracting patterns from data, enabling predictions for unseen data. Understanding the fundamentals of ML, including features, targets, and the model training process, is essential for successfully applying ML techniques. By leveraging the capabilities of libraries like NumPy, practitioners can enhance their data analysis and machine learning workflows.

## Documentation & Links

* To access the complete article, please visit our link [Medium](https://medium.com/@dimasadit)

* To view the repository, please visit our link [Dimas Project](https://dimasrepo.github.io/)

