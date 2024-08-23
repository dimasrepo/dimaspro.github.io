---
date: 2024-08-23 10:01:59
layout: post
title: "Employee Attrition Analysis"
subtitle: Understanding Employee Attrition: Analyzing Key Factors Influencing Turnover in the Workplace
description: This dataset contains information on 1,470 employees, exploring various demographic, job-related, and environmental factors to analyze patterns and drivers of employee attrition.
image: https://github.com/user-attachments/assets/8556a750-fd3e-4521-8332-b100424b52ea
optimized_image: https://github.com/user-attachments/assets/9a97cce4-3b47-4576-b36b-e905842b57d3
category: Rpubs
tags: Rpubs
author: Dimas
paginate: true
---


## Overview

The dataset examines employee attrition within a company and includes 1,470 employee records with 35 attributes. These attributes provide insights into various factors influencing employee turnover, such as demographic information, job-related metrics, compensation details, work environment, and career progression. This comprehensive dataset is valuable for understanding the dynamics of employee attrition and identifying key factors that contribute to turnover, enabling targeted interventions to enhance employee retention and organizational stability.

## Business Questions

1. **What is the breakdown of distance from home by job role and attrition?**
2. **How does the average monthly income compare across different education levels and attrition?**

## Project Structure

- **Introduction:** Overview of the dataset and the purpose of the analysis.
- **Exploring the Data:** Initial data exploration and understanding the distribution of variables.
- **Analyzing Trends:** Identifying patterns and correlations related to employee attrition.
- **Model Building:** Developing predictive models to understand factors influencing employee attrition.
- **Visualizations:** Creating visual representations of the data to highlight key insights.
- **Reporting/Conclusion:** Summarizing findings and providing recommendations based on the analysis.

## Column Descriptions

| Variable                    | Data Type | Role       | Description                                                        | Example                     |
|-----------------------------|-----------|------------|--------------------------------------------------------------------|-----------------------------|
| Age                         | Integer   | Predictor  | Age of the employee                                                | 41                          |
| Attrition                   | Character | Target     | Whether the employee left the company (Yes, No)                    | Yes                         |
| BusinessTravel              | Character | Predictor  | Frequency of business travel (Non-Travel, Travel_Rarely, Travel_Frequently) | Travel_Rarely               |
| DailyRate                   | Integer   | Predictor  | Amount paid per day of work                                        | 1102                        |
| Department                  | Character | Predictor  | Work department (Research & Development, Sales, Human Resources)   | Sales                       |
| DistanceFromHome            | Integer   | Predictor  | Distance between company and home                                  | 1                           |
| Education                   | Integer   | Predictor  | Level of education (1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor) | 2                   |
| EducationField              | Character | Predictor  | Field of education (Life Sciences, Medical, Human Resources, Technical Degree, Marketing, Other) | Life Sciences |
| EmployeeCount               | Integer   | Identifier | Count of employees (always 1)                                      | 1                           |
| EmployeeNumber              | Integer   | Identifier | ID of the employee                                                 | 1                           |
| EnvironmentSatisfaction     | Integer   | Predictor  | Satisfaction with environment (1: Low, 2: Medium, 3: High, 4: Very High) | 2                     |
| Gender                      | Character | Predictor  | Gender of the employee (Male, Female)                              | Female                      |
| HourlyRate                  | Integer   | Predictor  | Amount paid per hour of work                                       | 94                          |
| JobInvolvement              | Integer   | Predictor  | Level of job involvement (1: Low, 2: Medium, 3: High, 4: Very High) | 3                          |
| JobLevel                    | Integer   | Predictor  | Level of the job (1 - 5)                                           | 2                           |
| JobRole                     | Character | Predictor  | Role of the job (Sales Executive, Research Scientist, Laboratory Technician, etc.) | Sales Executive   |
| JobSatisfaction             | Integer   | Predictor  | Satisfaction with the job (1: Low, 2: Medium, 3: High, 4: Very High) | 4                          |
| MaritalStatus               | Character | Predictor  | Marital status (Married, Single, Divorced)                         | Single                      |
| MonthlyIncome               | Integer   | Predictor  | Monthly income                                                     | 5993                        |
| MonthlyRate                 | Integer   | Predictor  | Monthly rate of pay                                                | 19479                       |
| NumCompaniesWorked          | Integer   | Predictor  | Number of companies the employee has worked with                   | 8                           |
| Over18                      | Character | Identifier | Whether the employee is over 18 years old (Yes, No)                | Yes                         |
| OverTime                    | Character | Predictor  | Whether the employee works overtime frequently (Yes, No)           | Yes                         |
| PercentSalaryHike           | Integer   | Predictor  | Percentage increase in salary                                      | 11                          |
| PerformanceRating           | Integer   | Predictor  | Level of performance assessment (1: Low, 2: Good, 3: Excellent, 4: Outstanding) | 3                     |
| RelationshipSatisfaction    | Integer   | Predictor  | Satisfaction with relationships (1: Low, 2: Medium, 3: High, 4: Very High) | 1                    |
| StandardHours               | Integer   | Predictor  | Standard work hours (always 80)                                    | 80                          |
| StockOptionLevel            | Integer   | Predictor  | Stock option level (0 - 3)                                         | 0                           |
| TotalWorkingYears           | Integer   | Predictor  | Total number of years the employee has worked                      | 8                           |
| TrainingTimesLastYear       | Integer   | Predictor  | Number of training times in the last year                          | 0                           |
| WorkLifeBalance             | Integer   | Predictor  | Level of work-life balance (1: Bad, 2: Good, 3: Better, 4: Best)   | 1                           |
| YearsAtCompany              | Integer   | Predictor  | Number of years at the current company                             | 6                           |
| YearsInCurrentRole          | Integer   | Predictor  | Number of years in the current role                                | 4                           |
| YearsSinceLastPromotion     | Integer   | Predictor  | Number of years since last promotion                               | 0                           |
| YearsWithCurrManager        | Integer   | Predictor  | Number of years with the current manager                           | 5                           |

## Workflow

1. **Data Collection:** Gather the dataset from the source.
2. **Data Cleaning:** Handle missing values, remove duplicates, and ensure data quality.
3. **Exploratory Data Analysis (EDA):** Analyze the dataset to understand the distribution and relationships between variables.
4. **Feature Engineering:** Create new features or modify existing ones to improve model performance.
5. **Model Building:** Develop machine learning models to predict employee attrition.
6. **Model Evaluation:** Assess the performance of the models using accuracy, recall, precision, and other metrics.
7. **Visualization:** Create visualizations to present findings and insights.
8. **Reporting:** Summarize the analysis, model results, and provide actionable insights.

## Exploring the Data

We will start by exploring the dataset to understand the distribution of key variables such as age, attrition rates, job roles, and income levels. We'll also examine correlations between different predictors and the target variable, attrition.

## Analyzing Trends

Identifying trends in employee attrition based on different attributes such as job role, department, and job satisfaction. This analysis will help in understanding which factors contribute most significantly to employee turnover.

## Visualizations

Visualizations will be created to represent the data and findings, such as:

- Attrition rate by job role and department.
- Distribution of employee age and its correlation with attrition.
- Monthly income comparisons across different education levels and job roles.

## Reporting

Based on the analysis, we conclude that while certain factors such as distance from home, job satisfaction, and work-life balance significantly influence employee attrition, the model's performance needs further improvement. High accuracy with low recall suggests that the model may be biased towards the majority class. Future work will focus on enhancing model recall and precision through advanced modeling techniques and more balanced data.

## Dataset Source

The dataset used in this project can be found at the following link: [Employee Attrition Analysis](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset).
Report for this project can be found at the following link: [Employee Attrition Analysis](https://rpubs.com/senddimas/1212700)
---
