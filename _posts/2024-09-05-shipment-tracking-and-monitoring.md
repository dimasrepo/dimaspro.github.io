---
date: 2024-09-05 03:49:29
layout: post
title: "Shipment Tracking and Monitoring"
subtitle: Real-time tracking and monitoring of shipments ensure transparency and timely updates for both the company and customers
description: Shipment tracking and monitoring involve the systematic oversight of delivery progress, enabling stakeholders to receive real-time updates and enhance operational efficiency.
image: https://github.com/user-attachments/assets/754bbdd4-6478-4a04-bc16-591dec3e5736
optimized_image: https://github.com/user-attachments/assets/754bbdd4-6478-4a04-bc16-591dec3e5736
category: Looker Studio Dashboard
tags: Looker Studio
author: Dimas
paginate: true
---

## Overview
The dashboard provides a comprehensive view of delivery operations for a logistics or e-commerce company in Indonesia, focusing on efficiency, geographic distribution, and courier performance. It highlights trends and areas for improvement, ultimately aiming to enhance service quality and operational effectiveness.

## Business Question
**How can the company optimize its delivery operations and improve customer satisfaction based on current delivery performance, geographic trends, and courier efficiencies?**

## Project Structure
1. **Data Collection**: Gathering delivery data from various sources.
2. **Data Cleaning**: Ensuring accuracy and consistency in the dataset.
3. **Data Analysis**: Examining trends and performance metrics.
4. **Visualization**: Creating graphical representations of the data.
5. **Reporting**: Summarizing findings and making recommendations.

## Column Descriptions
- **Periode**: The month of the delivery record.
- **Batch_Id**: Unique identifier for the batch of records.
- **Order_Id**: Unique identifier for each order.
- **No_Resi_OEX / No_Resi_3PL**: Tracking numbers for different couriers.
- **Reference_Code**: Reference identifier for tracking.
- **Kurir**: Courier service used.
- **Submit_date**: Date the order was submitted.
- **schedule_pickup**: Scheduled pickup date.
- **Tgl_pickup**: Actual pickup date.
- **SLA Pickup**: Service Level Agreement for pickup time.
- **Lead time delivery**: Time taken from pickup to delivery.
- **Pengirim**: Sender's name.
- **Dest_address**: Delivery address.
- **shipping_service**: Type of shipping service chosen.
- **Last_status**: Final status of the delivery.
- **Tgl_last_status**: Date of last status update.
- **penerima**: Recipient's name.
- **Telp_penerima**: Recipient's phone number.
- **shipping_price**: Cost of shipping.
- **metode_pembayaran**: Payment method.
- **COD_amount**: Cash on delivery amount.
- **Insurance_fee**: Fee for insurance on the shipment.
- **Firstmile Oexpress / Lastmile Oexpress**: First and last-mile delivery metrics.
- **Oexpress coverage**: Coverage area for Oexpress.

## Workflow
1. **Data Import**: Load the delivery dataset into the analysis tool.
2. **Data Preparation**: Clean and format the data for analysis.
3. **Exploratory Data Analysis (EDA)**: Identify trends and patterns.
4. **Visualization Creation**: Generate graphs and charts to illustrate findings.
5. **Report Drafting**: Compile insights into a coherent report with recommendations.

## Exploring the Data
- **Delivery Performance**: Examine metrics such as average delays, success rates, and types of delivery statuses.
- **Geographic Analysis**: Map delivery density across Indonesia, identifying regions with high and low activity.
- **Courier Analysis**: Compare performance metrics of different couriers, focusing on delivery times and success rates.

## Analyzing Trends
- **Delivery Timing**: Analyze variations in delivery times to identify trends over periods.
- **Courier Performance**: Evaluate which couriers consistently meet delivery expectations and which fall short.
- **Customer Behavior**: Investigate the reasons behind "failed-requests" and their impact on overall delivery performance.

## Visualizations
1. **Map of Delivery Activity**: Illustrates geographic distribution of deliveries.
2. **Pie Chart of Couriers Used**: Shows proportions of deliveries handled by each courier.
3. **Bar Charts for Delivery Status**: Displays counts of different delivery outcomes.
4. **Vertical Bar Chart for Delay Analysis**: Compares average delays across different couriers.

## Reporting/Conclusion
The analysis indicates a generally efficient delivery operation with slight over-performance in delivery timing. However, attention is required for high "failed-request" counts, which could reflect issues needing resolution. Recommendations include:
- Investigating the causes of "failed-requests" to enhance order fulfillment.
- Analyzing courier performance for better selection based on delivery metrics.
- Leveraging geographic insights to optimize delivery routes and expand into high-demand areas.
