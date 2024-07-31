---
date: 2024-07-31 11:19:38
layout: post
title: "Bank Portuguese Campaign"
subtitle:
description: The dataset is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls, often requiring multiple contacts with the same client to assess if the product (bank term deposit) would be subscribed ('yes') or not ('no'). The dataset consists of 45,211 instances with 16 features.
image: ![banco-portugal](https://github.com/user-attachments/assets/057ff5bc-ccb3-4a67-8980-72eb0b9f4b3e)
optimized_image: ![banco-portugal](https://github.com/user-attachments/assets/1b0b0356-ac03-4b1a-a073-81206b3e3fb2)
category:
tags:
author:
paginate: true
---

# Bank Marketing Data

![banco-portugal](https://github.com/user-attachments/assets/263edb2b-e978-4a08-8605-ffc3a7f161f6)

## Overview
The dataset is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls, often requiring multiple contacts with the same client to assess if the product (bank term deposit) would be subscribed ('yes') or not ('no'). The dataset consists of 45,211 instances with 16 features.

## Directory Structure
This directory contains the following files:
- `bank-additional-full.csv`: Contains all examples (41,188) and 20 inputs, ordered by date (from May 2008 to November 2010).
- `bank-additional.csv`: Contains 10% of the examples (4,119), randomly selected from `bank-additional-full.csv`, with 20 inputs.
- `bank-full.csv`: Contains all examples and 17 inputs, ordered by date (older version of this dataset with fewer inputs).
- `bank.csv`: Contains 10% of the examples from `bank-full.csv`, with 17 inputs, randomly selected from `bank-full.csv` (older version with fewer inputs).
- `README.md`: This README file.
- `scripts/`: Directory for scripts used to analyze the data.

## Column Descriptions
- `age`: Age of the client (numeric).
- `job`: Type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown").
- `marital`: Marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed).
- `education`: Education level (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown").
- `default`: Has credit in default? (binary: "yes","no").
- `balance`: Average yearly balance (numeric, in euros).
- `housing`: Has housing loan? (binary: "yes","no").
- `loan`: Has personal loan? (binary: "yes","no").
- `contact`: Contact communication type (categorical: "cellular","telephone").
- `day_of_week`: Last contact day of the week (categorical: "mon","tue","wed","thu","fri").
- `duration`: Last contact duration (numeric, in seconds).
- `campaign`: Number of contacts performed during this campaign for this client (numeric, includes last contact).
- `pdays`: Number of days that passed since the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted).
- `previous`: Number of contacts performed before this campaign for this client (numeric).
- `poutcome`: Outcome of the previous marketing campaign (categorical: "unknown","other","failure","success").
- `y`: Has the client subscribed a term deposit? (binary: "yes","no").

## How to Use This Data

### Exploring the Data
1. Load the dataset into a data analysis environment like R or Python.
2. Clean the data by handling missing values and standardizing column names for consistency.

### Analyzing Trends
1. Analyze subscription trends over time to identify any patterns or significant changes.
2. Compare the subscription rates across different client demographics and attributes.
3. Examine the effectiveness of previous marketing campaigns using the `poutcome` variable.

### Visualizations
1. Create visualizations such as line charts, bar charts, and maps to illustrate trends and distributions.
2. Use tools like ggplot2 in R or Matplotlib in Python for visualization.

### Reporting
1. Summarize findings in reports or presentations.
2. Highlight key insights and recommendations based on the analysis.

## Dataset Characteristics
- **Multivariate**
- **Subject Area**: Business
- **Associated Tasks**: Classification
- **Feature Type**: Categorical, Integer

## Additional Information
- The dataset has no missing values.
- The classification goal is to predict if the client will subscribe a term deposit (`y`).
- The data was collected from May 2008 to November 2010.

## Instances and Features
- **# Instances**: 45,211
- **# Features**: 16

## Dataset Source :
https://rpubs.com/senddimas/1201699
