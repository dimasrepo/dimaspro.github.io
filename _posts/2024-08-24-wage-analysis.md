---
date: 2024-08-24 04:26:22
layout: post
title: "Wage Analysis"
subtitle: Evaluating Employee Compensation in Relation to Living Costs Across States
description: This study examines employee compensation data, including gross pay and state pay, against the living wage standards across various states, providing a comprehensive view of wage adequacy and economic well-being of employees.
image: https://github.com/user-attachments/assets/3408e9ff-544a-4408-9586-6a58932fdd0e
optimized_image: https://github.com/user-attachments/assets/3408e9ff-544a-4408-9586-6a58932fdd0e
category: Power BI Dashboard
tags: Power BI
author: Dimas
paginate: false
---


## Overview
This project analyzes a dataset containing employee compensation details, including gross pay, living wage, average state pay, and other demographic information. The objective is to understand the distribution of pay across different states, compare compensation against living costs, and identify trends in employee earnings over several years.

## Business Question
The key business questions addressed in this analysis are:
1. How does employee gross pay vary across different states?
2. How do average state pay and gross pay compare with the living wage in different regions?
3. What are the trends in gross pay over the years from 2013 to 2015?

## Project Structure
The project is structured as follows:
- **data/**: Contains the raw dataset (`employee_compensation.csv`).
- **scripts/**: Includes R scripts for data cleaning, analysis, and visualization (`compensation_analysis.R`).
- **reports/**: Contains the generated reports and visualizations (`compensation_analysis_report.pdf`).
- **README.md**: Documentation file outlining the project details, objectives, and results.

## Column Descriptions
- **EmpID**: Unique identifier for each employee.
- **GeographyKey**: Numeric key corresponding to geographic regions.
- **FirstName**: First name of the employee.
- **LastName**: Last name of the employee.
- **Company**: The company where the employee works.
- **Address**: Street address of the employee.
- **City**: City of residence for the employee.
- **County**: County of residence for the employee.
- **State**: State of residence for the employee.
- **Zip.Code**: Postal code of the employee's address.
- **Phone**: Contact phone number of the employee.
- **Email**: Contact email address of the employee.
- **LivingWage**: Minimum wage required for basic living expenses in the employee's region.
- **AveStatePay**: Average state pay for the employee's state of residence.
- **GrossPay2013**: Gross pay earned by the employee in the year 2013.
- **GrossPay2014**: Gross pay earned by the employee in the year 2014.
- **GrossPay2015**: Gross pay earned by the employee in the year 2015.

## Workflow
1. **Data Import and Cleaning**: Import the dataset and clean any missing or irrelevant data entries, including formatting numeric columns for analysis.
2. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand the distribution of employee compensation data across different states and regions.
3. **Data Visualization**: Create various plots to visualize state pay distributions, comparisons between pay, state pay, and living cost, and other key metrics.
4. **Trend Analysis**: Identify and analyze trends in gross pay over the years from 2013 to 2015.
5. **Reporting**: Compile the findings and visualizations into a comprehensive report.

## Exploring the Data
The dataset contains 5008 observations across 17 variables. Initial exploration involves understanding the distribution of employee compensation data by state and region. We focus on comparing the living wage with average state pay and gross pay across different states.

## Analyzing Trends
The analysis focuses on:
- Differences in gross pay, average state pay, and living wage across various states.
- Identifying states with significant deviations in compensation compared to the living wage.
- Analyzing trends in gross pay from 2013 to 2015 to understand changes in employee compensation.

## Visualizations
Several visualizations have been created to highlight key findings:
1. **Bar Plot - State Pay Over State**: Shows the distribution of average state pay across different states.
2. **Bar Plot - Pay vs State Pay vs Living Cost**: Compares employee gross pay, average state pay, and living wage across different states.
3. **Card Plot - Avg Living Cost**: Displays the average living cost across different regions.
4. **Card Plot - Avg State Pay**: Displays the average state pay across different regions.
5. **Card Plot - Avg Gross Pay**: Displays the average gross pay across different regions.
6. **Geo Spatial Plot**: Visualizes the geographic distribution of employee data.
7. **Gauge Plot - Sum Living Wage and Sum of Gross Pay**: Compares the sum of living wages to the sum of gross pay to highlight discrepancies.

## Reporting/Conclusion
The analysis reveals significant variations in employee compensation across different states. Some states offer a gross pay that is significantly above the living wage, while others are closer to or below the living wage threshold. The trends over the years indicate slight increases in gross pay, but disparities between living wages and actual compensation remain a concern in certain regions.

## Dataset Source
The dataset used in this project is a hypothetical dataset provided for educational purposes, simulating real-world employee compensation data for analytical exercises.

---
