---
date: 2024-08-21 11:33:35
layout: post
title: "Time Series Scooty"
subtitle: Mastering the art of accurate forecasting with advanced time series analysis.
description: Dive into a comprehensive exploration of time series data, leveraging sophisticated preprocessing and model selection techniques to achieve precise and reliable predictions.
image: https://github.com/user-attachments/assets/2f5a751e-0ea7-4e34-aa0d-70355e47cb64
optimized_image: https://github.com/user-attachments/assets/2f5a751e-0ea7-4e34-aa0d-70355e47cb64
category: Rpubs
tags: Rpubs
author: Dimas
paginate: true
---


<!--
## Overview

Scotty, a ride-sharing service operating in major Turkish cities, specializes in motorcycle transportation, offering citizens a quick and efficient way to navigate traffic. Their app even playfully nods to Star Trek with a “beam me up” order button. Using real-time transaction data provided by Scotty, we aim to assist in optimizing their business processes and enhancing overall efficiency.

This project involves forecasting time series data by preprocessing, cross-validating, and selecting the best models to predict future values. The primary goal is to build a robust forecasting model that can handle multiple seasonality effects and provide accurate predictions based on historical data.

## Business Question

The main objective is to determine how to effectively forecast time series data by applying various preprocessing techniques, model specifications, and cross-validation strategies. The focus is on accurately predicting future values and understanding the impact of different seasonalities and preprocessing methods on model performance.

## Project Structure

The project is organized into the following sections:

- **Data Preprocessing**: Includes rounding datetime values, aggregating data, and padding time series to ensure completeness.
- **Cross-Validation Scheme**: Prepares data for automated model selection, including grouping, nested dataframes, and rolling origin method.
- **Automated Model Selection**: Compares different preprocessing approaches, seasonality specifications, and forecasting models.
- **Prediction Performance**: Evaluates and submits predictions based on the selected model.
- **Reporting/Conclusion**: Summarizes findings and the effectiveness of the chosen methods.

## Column Descriptions

The dataset includes the following columns:

- **id**: Unique identifier for each record
- **trip_id**: Identifier for individual trips
- **driver_id**: Identifier for drivers
- **rider_id**: Identifier for riders
- **start_time**: Timestamp when the trip started
- **src_lat**: Latitude of the source location
- **src_lon**: Longitude of the source location
- **src_area**: Source area description
- **src_sub_area**: Sub-area within the source area
- **dest_lat**: Latitude of the destination location
- **dest_lon**: Longitude of the destination location
- **dest_area**: Destination area description
- **dest_sub_area**: Sub-area within the destination area
- **distance**: Distance of the trip
- **status**: Status of the trip
- **confirmed_time_sec**: Time in seconds when the trip was confirmed

## Workflow

- **Data Preprocessing**: Round datetime values to the hour, aggregate data by `sub_area` and `datetime`, and apply time series padding from `2017-10-01 00:00:00 UTC` to `2017-12-02 23:00:00 UTC`.
- **Cross-Validation Scheme**: Prepare training and testing datasets with a 1-week test period. Use the rolling origin method to split training data into training and validation sets.
- **Automated Model Selection**: Test various preprocessing methods (square root transformation, mean subtraction), seasonality specifications (single and multiple seasonality), and models (ETS, Auto ARIMA, STLM, Holt-Winters, TBATS).
- **Prediction Performance**: Measure model performance using Mean Absolute Error (MAE) and select the best-performing model.

## Exploring the Data

Data exploration involves analyzing the completeness and consistency of the dataset, understanding the distribution of key variables, and preparing the data for model training and testing.

## Analyzing Trends

Trends are analyzed by fitting different models to the time series data and evaluating their performance. The analysis focuses on understanding how well each model captures seasonal patterns and other temporal characteristics.

## Visualizations

Visualizations are used to present data trends, model performance, and forecasting results. Key visualizations include time series plots, error metrics, and comparison charts for different models.

## Reporting/Conclusion

The preprocessing and model selection processes were rigorously executed to ensure accurate forecasting results. The data was preprocessed by rounding datetime values, aggregating based on `sub_area` and `datetime`, and performing time series padding to handle incomplete data. Cross-validation was prepared with a clear split between training and testing datasets, and the rolling origin method was used to evaluate model performance effectively. Multiple preprocessing approaches and seasonality specifications were compared, with the TBATS model emerging as the best-performing model based on the lowest Mean Absolute Error (MAE). This comprehensive approach led to reliable forecasting results.

## Dataset Source

 The dataset used for this project is sourced from -->

