---
date: 2018-11-22 12:26:40
layout: post
title: Ecommerce Apache Superset
subtitle: This repository contains a collection of visualizations created with Apache Superset.
description: 
image: ![Brazilan Ecommerce](https://github.com/user-attachments/assets/fe816fd1-f5aa-4891-9fe6-2ddc97e7aa3b)

optimized_image: ![Brazilan Ecommerce](https://github.com/user-attachments/assets/37733743-755d-48c1-9031-c194ecca2b02)

category: Dashboard
tags:
  - Postgree
  - Apache
author: Dimas
paginate: true
---

The dashboards provide insights into various aspects of the Olist dataset, including product performance, sales trends, customer profiles, and more.

## Overview

The visualizations in this repository explore different facets of the Olist dataset to offer valuable business insights. These dashboards include analysis on product performance by category, the influence of product weight on pricing, seller performance, payment methods, and more. Each visualization aims to identify trends and patterns that can inform business strategies and decision-making.

## Directory Structure


## Column Descriptions

Each dataset used for the visualizations has specific columns that were leveraged for analysis:

- **`olist_products_dataset`**: Product ID, Product Category, Product Weight, Product Dimensions
- **`olist_order_items_dataset`**: Order ID, Product ID, Price, Freight Value
- **`olist_orders_dataset`**: Order ID, Purchase Time, Delivery Time, Order Status
- **`olist_sellers_dataset`**: Seller ID, Seller City
- **`olist_customers_dataset`**: Customer ID, Customer City, Purchase Frequency
- **`olist_order_payments_dataset`**: Order ID, Payment Method, Payment Value

## How to Use This Data

1. **Clone the Repository**: Use `git clone https://github.com/yourusername/your-repository.git` to get a local copy of the repository.
2. **Install Dependencies**: Ensure you have Apache Superset installed to visualize the dashboards.
3. **Open Dashboards**: Use Apache Superset to open and interact with the visualizations provided in the `dashboards/` directory.

## Exploring the Data

The dashboards offer various views and insights into the Olist dataset. Key visualizations include:

- **Presentation Order by Category**: Identifies best and worst performing products by category.
- **Influence of Product Weight and Dimensions**: Assesses how product weight and dimensions affect price and freight value.
- **Purchase Time vs. Order Status**: Analyzes the impact of purchase time on order status and delivery.
- **Seller Performance**: Evaluates seller performance based on their location.
- **Number of Photos vs. Sales**: Measures the effect of the number of product photos on sales levels.
- **Payment Methods vs. Payment Value**: Compares payment methods to transaction values.
- **Purchase Patterns by Time**: Examines purchase patterns based on time of day or month.
- **Customer Profile by Location**: Identifies customer profiles based on location and purchase frequency.
- **Sales Growth**: Analyzes months with significant sales growth.

## Analyzing Trends

The visualizations reveal several trends:

- **Product Categories**: Household and beauty products are popular, influencing inventory and promotions.
- **Price and Shipping Costs**: Larger and heavier products generally have higher prices and shipping costs.
- **Order Fulfillment**: Faster order fulfillment correlates with higher success rates.
- **Seller Performance**: Sao Paulo leads in sales, with growth potential in other cities.
- **Number of Photos**: More product photos can lead to higher sales.
- **Payment Methods**: Credit cards are the most popular and generate the largest transaction values.
- **Purchase Patterns**: Significant increases in orders are observed during the holiday season.
- **Customer Profiles**: Large cities show higher purchase frequencies, indicating key markets.
- **Sales Growth**: Overall sales growth is positive, with notable spikes during specific periods.

## Visualizations

 1. Presentation Order by Category
 2. The Influence of Product Weight and Dimensions on Price and Freight Value
 3. The Influence of Purchase Time on Order Status
 4. Seller Performance
 5. The Influence of the Number of Product Photos on Sales
 6. Comparison of Payment Methods to Payment Value
 7. Purchase Patterns Based on Time
 8. Customer Profile Based on Location and Purchase Frequency
 9. Sales Growth

## Reporting

For any questions or further information regarding these visualizations, please open an issue or contact via [LinkedIn](https://www.linkedin.com/in/your-profile) or [GitHub](https://github.com/yourusername).

## Dataset Source

The visualizations are based on the Olist dataset, available at [Olist Data Repository](https://www.kaggle.com/olistbr/brazilian-ecommerce).










