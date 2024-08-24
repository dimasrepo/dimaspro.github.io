---
date: 2024-08-24 04:26:48
layout: post
title: "Revenue vs Projection"
subtitle: Analyzing Regional Revenue Performance Across Various Geographies
description: This analysis provides insights into the financial performance of different regions by comparing projected and actual revenues, highlighting key areas of growth and identifying regions that require strategic attention.
image: https://github.com/user-attachments/assets/e86624f5-db26-442c-85e6-c7691574331d
optimized_image: https://github.com/user-attachments/assets/e86624f5-db26-442c-85e6-c7691574331d
category: Power BI Dashboard 
tags: Power BI
author: Dimas
paginate: false
---

## Overview
This project involves analyzing a geographical dataset to understand the differences between projected and actual revenues across various regions and countries. The dataset includes key information such as geography types, continent names, city names, state names, employee counts, location identifiers, and revenue data.

## Business Question
The primary business questions addressed in this analysis are:
1. How does the actual revenue compare to the projected revenue across different regions and countries?
2. Which regions show significant overperformance or underperformance in terms of revenue?
3. What trends can be observed in revenue differences across continents or specific states?

## Project Structure
The project is structured as follows:
- **data/**: Contains the raw Geo dataset (`geo_dataset.csv`).
- **scripts/**: Includes R scripts for data cleaning, analysis, and visualization (`data_analysis.R`).
- **reports/**: Contains the generated reports and visualizations (`revenue_analysis_report.pdf`).
- **README.md**: Documentation file outlining the project details, objectives, and results.

## Column Descriptions
- **GeographyKey**: Unique numerical identifier for each geographical entry.
- **GeographyType**: Type of geographical entity (e.g., "Country/Region").
- **Continent**: Numerical code representing the continent.
- **ContinentName**: Name of the continent (e.g., "Asia").
- **CityName**: Name of the city (if applicable).
- **Region**: Name of the region (if applicable).
- **State**: Name of the state or country.
- **Locations**: Number of locations associated with each entry.
- **Employees**: Number of employees associated with each location.
- **Location ID**: Unique identifier for each location.
- **LoadDate**: Date when the data was loaded.
- **UpdateDate**: Date when the data was last updated.
- **Projected Revenue**: Estimated revenue for each geographical entity.
- **Actual Revenue**: Actual revenue achieved for each geographical entity.
- **Difference**: Numerical difference between projected and actual revenue.
- **% Difference**: Percentage difference between projected and actual revenue.

## Workflow
1. **Data Import and Cleaning**: Import the dataset and clean any missing or irrelevant data entries.
2. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand the distribution of revenue data across different regions and geographical types.
3. **Data Visualization**: Create bar plots to visualize the differences between projected and actual revenues.
4. **Trend Analysis**: Identify and analyze trends in revenue data across different continents and states.
5. **Reporting**: Compile the findings and visualizations into a comprehensive report.

## Exploring the Data
The dataset contains 671 records across 16 columns. Initial exploration involves understanding the distribution of data across different continents, countries, and states. The analysis focuses on comparing the projected and actual revenue to identify patterns and anomalies.

## Analyzing Trends
To analyze trends, we focus on:
- Differences between projected and actual revenue by continent and state.
- Identifying geographical regions that consistently overperform or underperform.
- Analyzing the percentage difference to determine the scale of deviation from projections.

## Visualizations
Three primary visualizations have been created to highlight key findings:
1. **Bar Plot - Sales over Projection**: Displays the total sales compared to projections across different geographical entities.
2. **Bar Plot - Project vs Actual**: Compares projected versus actual revenue for each region.
3. **Bar Plot - Actual Revenue over Projection**: Shows the actual revenue achieved as a percentage over or under the projected revenue.

## Reporting/Conclusion
The analysis highlights significant variations between projected and actual revenues across different regions. Certain regions consistently outperform projections, while others fall short. The findings suggest a need for a more granular approach to revenue forecasting, considering regional characteristics and market dynamics.

## Dataset Source
The dataset used in this project is a hypothetical Geo dataset provided for educational purposes, simulating real-world geographical and revenue data for analytical exercises.

---



