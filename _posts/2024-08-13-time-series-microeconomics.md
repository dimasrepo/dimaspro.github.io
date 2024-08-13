---
date: 2024-08-13 14:51:48
layout: post
title: "Time Series Microeconomics"
subtitle: In this project, we will use the Microeconomic time series data from the Time Series Data Library to develop and evaluate four forecasting models—ARIMA, Holt’s Winter, STLF, and TBATS—to determine the most accurate method for predicting future profits and enhancing strategic financial planning.
description:
image: ![hour](https://github.com/user-attachments/assets/ff4f8c7a-5920-4c68-929a-a993e9570c7f)
optimized_image:
category:
tags:
author:
paginate: false
---
# Time Series Forecasting with Microeconomic Data

## Overview

In this project, we focus on time series data from the Time Series Data Library (FinYang/tsdl). The dataset encompasses a wide range of 648 time series across various domains. Specifically, we examine the Microeconomic data subset, which includes 36 time series records. The data spans different frequencies, from very high (0.1) to annual (365), and covers a variety of subjects relevant to time series analysis. The goal is to develop and assess forecasting models tailored to microeconomic variables, providing insights and predictions for economic analysis and decision-making.

## Directory Structure

The directory structure for this project is as follows:



## Features of the Dataset
- **Comprehensive Data Access**: Explore data on the production, consumption, and trade of oil, gas, coal, power, and renewables, along with CO2 emissions from fuel combustion.
- **Global Coverage**: Data encompasses 60 countries and regions worldwide, from 1990 to 2023.
- **Exclusive Foresight**: Gain insights into essential energy data and evaluate the COP28 pledge to determine if current trends support the tripling of renewable capacity and the doubling of energy efficiency by 2030.

## Benefits of the Interactive Data Tool
- **Animated Data Evolution**: Visualize trends over time from 1990 to 2023.
- **Interactive Map**: Easily select areas with zoom in and out controls.
- **Country Benchmarking**: Compare data across different countries.
- **Flexible Period Selection**: Choose any time range to view data.
- **Data Export**: Export data globally or by specific energy sources.


## Column Descriptions

The dataset contains the following columns:

- **Date**: The date of the observation.
- **Value**: The observed value for the given date.

## How to Use This Data

To utilize this dataset, follow these steps:

1. **Load the Libraries**: Ensure that the required libraries are installed and loaded in your R environment.

   library(tidyverse)
   library(lubridate)
   library(forecast)
   library(TTR)
   library(fpp)
   library(tseries)
   library(TSstudio)
   library(padr)
   library(recipes)
   library(tidyquant)
   library(ggplot2)
   library(tsdl)
  

2. Import the Dataset: Load the dataset from the Time Series Data Library.
tsdl_microeconomic <- subset(tsdl, 12, "microeconomic")
3. Prepare and Clean the Data: Handle missing values and format the dataset as needed.


### Exploring the Data
The exploratory data analysis involves:
- Visualizing Time Series: Plot the time series to understand its patterns.
micro_ts %>% autoplot()
- Decomposition: Decompose the time series into trend, seasonal, and residual components.
micro_ts %>% decompose() %>% autoplot()
- Inspecting Components: Analyze individual components like trend and seasonality.
micro_decom <- decompose(x = micro_ts)
micro_decom$trend %>% autoplot()
micro_decom$seasonal %>% autoplot()

### Analyzing Trends
Time series analysis includes:

Cross-Validation: Split the data into training and testing sets.
test_micro <- tail(micro_ts, 24)
train_micro <- head(micro_ts, -length(test_micro))
Modeling: Apply different forecasting models:

- ARIMA:
model_arima_ts <- stlm(train_micro, s.window = 12, method = "arima")
forecast_arima_ts <- forecast(model_arima_ts, h = 24)
- Holt’s Winter
- STLF
- TBATS

### Visualizations
Visualizations include:
Time Series Plots: Display the time series data and its components.
Forecast Plots: Show the forecasts and confidence intervals.

###Reporting
Generate reports summarizing:

Model Performance: Compare the forecasting accuracy of different models.
Insights: Provide actionable insights based on the analysis.

## Source
The dataset used in this project can be found at the following link:
[TSDL](https://pkg.yangzhuoranyang.com/tsdl/)

Report for this project can be found at the following link:
[Time Series Microeconomics](https://rpubs.com/senddimas/1210655)
