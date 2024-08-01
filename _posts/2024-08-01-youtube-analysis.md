---
date: 2024-08-01 19:12:05
layout: post
title: "Youtube Analysis"
subtitle: 
description: This project aims to analyze trending YouTube videos for 2023, focusing on various aspects such as views, likes, and the impact of different publishing times and categories on video performance.
image: ![Yt6](https://github.com/user-attachments/assets/24acd77e-6c53-4edf-8f66-e99c98e2b410)

optimized_image:
category: Dashboard
tags: Shiny
author: Dimas
paginate: true
---
# YouTube Analysis Project

## Overview
This project aims to analyze trending YouTube videos for 2023, focusing on various aspects such as views, likes, and the impact of different publishing times and categories on video performance. The dataset includes information about 72,397 videos with 16 variables ranging from trending dates to viewer engagement metrics.

## Directory Structure
- `data/`: Contains the dataset used for analysis.
- `scripts/`: Includes the R scripts used for data cleaning, analysis, and visualization.
- `outputs/`: Stores the generated plots and reports.
- `README.md`: This document, explaining the project structure and findings.

## Column Descriptions
- **trending_date**: The date the video trended on YouTube.
- **title**: The title of the video.
- **channel_title**: The name of the channel that published the video.
- **category_id**: The category under which the video is listed.
- **publish_time**: The exact time the video was published.
- **views**: The number of views the video has received.
- **likes**: The number of likes on the video.
- **dislikes**: The number of dislikes on the video.
- **comment_count**: The number of comments on the video.
- **comments_disabled**: Whether comments are disabled for the video.
- **ratings_disabled**: Whether ratings are disabled for the video.
- **video_error_or_removed**: Whether the video has errors or has been removed.
- **publish_hour**: The hour at which the video was published.
- **publish_when**: Time range indicating when the video was published.
- **publish_wday**: The day of the week the video was published.
- **timetotrend**: Time taken for the video to trend after being published.

## How to Use This Data
To use this dataset, load it into any data analysis tool such as R, Python, or Excel. The dataset is particularly useful for analyzing patterns and trends in YouTube videos, such as the best times to publish for maximum engagement, the most popular categories, and identifying top-performing channels.

## Exploring the Data
Key explorations include:
- Identifying the top 10 channels within the "Gaming" category based on average views.
- Counting the number of videos in each category to see which content types are most popular.
- Analyzing the average views by the hour of publication to determine the optimal time to release content.

## Business Questions
1. **What is the top YouTube category by count?**
   This question explores which category has the highest number of trending videos, providing insights into the most active or popular genres on YouTube.

2. **Who are the top 10 channels in the Gaming category based on average views?**
   This analysis identifies the leading channels in the Gaming category, highlighting which content creators are most successful in terms of average viewership.

3. **How does the average view count for videos in the Gaming category vary by the hour of publication?**
   This question examines the optimal time for publishing Gaming videos to maximize viewership, based on historical data.

## Project Structure
- **Data**: The dataset used in this analysis includes 72,397 observations with 16 variables, capturing various aspects of YouTube videos.
- **Scripts**: All analysis scripts are organized to facilitate data exploration, trend analysis, and visualization.

## Workflow
1. **Data Cleaning**: The dataset was cleaned to remove irrelevant columns, handle missing data, and ensure consistency in categorical variables.
2. **Exploration**: Initial data exploration was conducted to understand the distribution of videos across categories and identify key trends.
3. **Analysis**: The analysis was divided into three main parts:
   - Counting the number of videos in each category.
   - Identifying the top 10 channels in the Gaming category.
   - Analyzing the impact of publishing hour on viewership for Gaming videos.
4. **Visualization**: Key findings were visualized using plots to aid in understanding trends and patterns in the data.

## Conclusion
- **Top Category**: The Gaming category leads in terms of the number of trending videos.
- **Top Channels in Gaming**: Channels like Rockstar Games, MrBeast Gaming, and Technoblade are among the top performers in the Gaming category based on average views.
- **Optimal Publishing Time**: The analysis revealed that certain hours, such as 3 AM and 4 AM, tend to have higher average views for Gaming videos.

## Analyzing Trends
The analysis reveals that channels like **Rockstar Games** and **MrBeast Gaming** dominate the "Gaming" category. The data also shows that videos published early in the morning (around 4 AM) tend to receive higher average views. Categories like **Gaming**, **Entertainment**, and **Music** have the highest counts of trending videos.

## Visualizations
The project includes several visualizations, such as:
- A bar chart showing the top 10 gaming channels by average views.
- A pie chart of video counts by category.
- A line graph depicting average views by the hour of publication for the gaming category.

## Reporting
Reports generated from this analysis provide insights into video performance trends, helping content creators and marketers optimize their strategies. These reports can be used to determine the best times to publish videos and the types of content that are likely to trend.

## Source
The dataset used in this project can be found at the following link:
[YouTube Analysis Project Data](https://github.com/dimasrepo/Youtube-Analysis/tree/main/Youtube_Analysis/data_input)

The report for this project can be found at the following link:
[YouTube Analysis Project Report](https://dimasaditya.shinyapps.io/Youtube_Analysis/)
