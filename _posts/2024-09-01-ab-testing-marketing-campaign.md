---
date: 2024-09-01 15:40:21
layout: post
title: "Marketing A/B Testing Analysis"
subtitle: The goal of this project is to evaluate how effective a marketing campaign is by examining user conversions and measuring the impact of ad exposure. Our analysis will focus on quantifying the role of advertisements in the overall success of the campaign
description: This project aims to analyze the effectiveness of a marketing campaign by assessing user conversions and determining the impact of ad exposure. By evaluating the data, we seek to quantify the contribution of advertisements to the campaign's success
image: https://github.com/user-attachments/assets/4eb340b4-e10d-4e05-beea-08a9a6ef5a58
optimized_image: https://github.com/user-attachments/assets/4eb340b4-e10d-4e05-beea-08a9a6ef5a58
category: Rpubs
tags: A/B Testing
author: Dimas
paginate: true
---

## Overview
This project focuses on analyzing the effectiveness of marketing campaigns through A/B testing. The aim is to evaluate whether exposure to advertisements influences user conversions and to quantify the impact of these ads on campaign success.

Click this link to directly access the report: [AB Testing Marketing Campaign](https://rpubs.com/senddimas/1215371)


## Introduction
Marketing companies often use A/B testing to compare different variations of marketing strategies. This project utilizes A/B testing data to determine the impact of ad exposure on user behavior. By analyzing this data, we aim to assess campaign success and attribute success to advertisement exposure.

## Business Question
The primary goals of this analysis are:
1. **Assessing Campaign Success**: Determine if the campaign was effective by analyzing the relationship between ad exposure and user conversions.
2. **Quantifying Success Attribution**: Measure the extent to which ads contribute to the overall success of the campaign.

## Project Structure
1. **Data Preparation**
   - Prerequisites
   - Importing Libraries
   - Importing Data
   - Data Inspection
2. **Exploratory Data Analysis**
   - Missing Values
3. **Data Wrangling**
   - Exploring Categorical Variables
   - Bootstrap Analysis
     - Conversion Rate by Group
     - Conversion Rate All Distribution
   - Univariate Analysis
   - Bivariate Analysis
   - Statistical Testing
4. **Conclusion**
5. **Dataset**

## Column Descriptions
- **no**: Integer, Identifier, Row index of the dataset
- **user.id**: Integer, Identifier, Unique identifier for the user
- **test.group**: Character, Predictor, Group assignment indicating exposure to advertisement
- **converted**: Logical, Target, Whether the user bought the product (True/False)
- **total.ads**: Integer, Predictor, Total number of ads seen by the user
- **most.ads.day**: Character, Predictor, Day of the week when the user saw the most ads
- **most.ads.hour**: Integer, Predictor, Hour of the day when the user saw the most ads

## Workflow

1. **Data Preparation**
   - **Prerequisites**: Ensure necessary libraries are installed.
   - **Importing Libraries**: Libraries like `tidyverse`, `lubridate`, `plotly`, and others are imported for data manipulation and visualization.
   - **Importing Data**: Load the dataset using `read.csv`.
   - **Data Inspection**: Inspect the structure and summary of the dataset.

2. **Exploratory Data Analysis**
   - **Missing Values**: Check for and handle any missing values in the dataset.

3. **Data Wrangling**
   - **Exploring Categorical Variables**: Analyze categorical variables for insights.
   - **Bootstrap Analysis**: Perform bootstrap analysis to estimate conversion rates.
     - **Conversion Rate by Group**: Analyze conversion rates by test group.
     - **Conversion Rate All Distribution**: Examine the overall distribution of conversion rates.
   - **Univariate Analysis**: Analyze individual variables.
   - **Bivariate Analysis**: Explore relationships between pairs of variables.
   - **Statistical Testing**: Conduct statistical tests to validate findings.

4. **Conclusion**
   - Summarize the findings from the analysis and provide insights into the effectiveness of the marketing campaign.

## Exploring the Data
The dataset consists of 588,101 observations across 7 variables. Initial inspection shows a balanced mix of converted (0) and not converted (1) users. There are no missing values in the dataset.

## Analyzing Trends
The analysis focuses on:
- The impact of the number of ads seen on conversion rates.
- Variations in conversion rates based on the day and hour ads were seen.

## Visualizations
Visualizations will include:
- Histograms of ads seen and conversion rates.
- Time series plots to analyze trends over days and hours.
- Bootstrap analysis plots to visualize the distribution of conversion rates.

## Reporting/Conclusion

- The analysis of the marketing campaign data aligns closely with the project's primary objectives of assessing campaign success and quantifying success attribution. By examining the optimal timing for campaigns—such as targeting Mondays and specific hours like 16:00—and evaluating how ad exposure impacts conversion rates, the study provides essential insights into the effectiveness of advertising strategies. The analysis reveals that Mondays, particularly in the late afternoon, consistently yield higher conversion rates, which is crucial for predicting campaign success and optimizing ad scheduling to enhance impact. Additionally, identifying the optimal ad exposure range (250-749 exposures) highlights a sweet spot where ads are most effective, addressing the goal of understanding how exposure influences conversions without leading to ad fatigue.

- Quantifying success attribution involves determining how much of the conversion success can be attributed to the ads themselves. The observed correlation between increased ad exposure and higher conversion rates suggests that well-calibrated exposure levels significantly contribute to campaign success. This underscores the importance of managing exposure levels carefully to avoid diminishing returns, ensuring that ads effectively engage users and drive conversions. Overall, the findings offer a clear path to achieving the project's goals by demonstrating the factors that most significantly drive campaign success and accurately attributing this success to the advertisements. Further analysis with additional data could provide even more precise insights, leading to improved campaign planning and resource allocation.

## Dataset Source
The dataset is sourced from Kaggle and can be accessed at [Marketing A/B Testing Dataset](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing/data)
The report for this project is available here: [AB Testing Marketing Campaign](https://rpubs.com/senddimas/1215371)
