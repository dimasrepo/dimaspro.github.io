---
date: 2024-08-13 14:52:13
layout: post
title: "Spotify Tracks Model"
subtitle: This series of articles will explore how the Spotify Web API was used to automatically retrieve data, with a focus on this topic.
description:
image: ![spot](https://github.com/user-attachments/assets/73da0e6d-9916-4bdb-ac1a-182d7a92e4b0)

optimized_image:![spot](https://github.com/user-attachments/assets/b8adbf61-ee0b-4b2f-9dfa-e8ddb8ea00fd)

category:
tags:
author:
paginate: false
---


## Overview

This project explores a dataset from Spotify, focusing on clustering analysis using the K-means method and examining the possibility of dimensionality reduction through Principal Component Analysis (PCA). The dataset, sourced from Kaggle, includes various audio features of tracks that will be analyzed to identify patterns and insights.

## Directory Structure

- `README.md`: This documentation file.
- `SpotifyFeatures.csv`: The dataset file containing Spotify track features.
- `analysis.R`: R script for data processing, analysis, and visualization.
- `results/`: Directory containing output files from the analysis.

## Column Descriptions

- `genre`: Genre of the track.
- `artist_name`: Name of the artist.
- `track_name`: Name of the track.
- `track_id`: Unique identifier for the track.
- `popularity`: Popularity score of the track.
- `acousticness`: Measure of acoustic quality.
- `danceability`: Measure of danceability.
- `duration_ms`: Duration of the track in milliseconds.
- `energy`: Measure of energy.
- `instrumentalness`: Measure of instrumental content.
- `key`: Key of the track.
- `liveness`: Measure of liveness.
- `loudness`: Loudness of the track in decibels.
- `mode`: Musical mode (Major/Minor).
- `speechiness`: Measure of speechiness.
- `tempo`: Tempo of the track in beats per minute.
- `time_signature`: Time signature of the track.
- `valence`: Measure of valence (happiness or mood).

## How to Use This Data

1. **Load the Data**: Use the provided `SpotifyFeatures.csv` file in your analysis.
   ```r
   data <- read.csv("SpotifyFeatures.csv")
2. Data Inspection: Examine the structure and summary of the data to understand its content and check for missing values.
   str(data)
   summary(data)
   anyNA(data)
3. Data Preparation: Convert columns to appropriate data types and handle any preprocessing
   data1 <- data %>%
  mutate(
    genre = as.character(genre),
    artist_name = as.character(artist_name),
    track_name = as.character(track_name),
    track_id = as.character(track_id),
    popularity = as.numeric(popularity),
    acousticness = as.numeric(acousticness),
    danceability = as.numeric(danceability),
    duration_ms = as.numeric(duration_ms),
    energy = as.numeric(energy),
    instrumentalness = as.numeric(instrumentalness),
    key = as.factor(key),
    liveness = as.numeric(liveness),
    loudness = as.numeric(loudness),
    mode = as.factor(mode),
    speechiness = as.numeric(speechiness),
    tempo = as.numeric(tempo),
    time_signature = as.factor(time_signature),
    valence = as.numeric(valence))
   
# Exploring the Data
Subsetting Data: Remove non-numeric columns if focusing on numeric features for clustering.
data2 <- data1 %>%
  select(-c(genre, artist_name, track_name, track_id, key, mode, time_signature))
  
## Summary Statistics: Analyze basic statistics to understand the distribution of features.
summary(data2)

# Analyzing Trends

## Clustering Potential: Assess the potential of features for clustering by examining their distributions.
Example: Distribution plot
ggplot(data2, aes(x = acousticness, fill = mode)) + geom_histogram()

## Principal Component Analysis (PCA): Use PCA to identify principal components and reduce dimensionality.
data_scale <- scale(data2)
pca_result <- prcomp(data_scale)
summary(pca_result)

# Visualizations

## Correlation Matrix: Visualize correlations between features to understand relationships.
ggcorr(data2, label = TRUE)
## PCA Biplot: Plot PCA results to visualize data in reduced dimensions.
biplot(pca_result)

# Reporting

## Clustering Results: Save and review clustering results.
write.csv(clustering_results, "results/clustering_results.csv")
PCA Results: Save PCA results for further analysis.
write.csv(pca_result$x, "results/pca_results.csv")


# Dataset Source
The dataset used in this project can be found at the following link: 
[Ultimate Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db/data)


Report for this project can be found at the following link:
[Spotify Tracks Model](https://rpubs.com/senddimas/1210395)











