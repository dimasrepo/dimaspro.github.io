---
date: 2024-07-31 10:54:06
layout: post
title: "Ecommerce Superset"
subtitle: "This repository contains a collection of visualizations created with Apache Superset."
description: "The dashboards provide insights into various aspects of the Olist dataset, including product performance, sales trends, customer profiles, and more."
image: ![Brazilan Ecommerce](https://github.com/user-attachments/assets/02b3a354-0372-4873-bd3d-880fa35b1455)
optimized_image: ![Brazilan Ecommerce](https://github.com/user-attachments/assets/02b3a354-0372-4873-bd3d-880fa35b1455)
category: Dashboard
tags:
  - PostgreSQL
  - Apache
author: Dimas
paginate: true
---

# Ecommerce Superset

![Brazilan Ecommerce](https://github.com/user-attachments/assets/02b3a354-0372-4873-bd3d-880fa35b1455)


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

This repository contains a collection of visualizations created with Apache Superset. The dashboards are designed to provide insights into various aspects of the Olist dataset, including product performance, sales trends, customer profiles, and more.

From this graph, we can observe several things:

Price and Shipping Cost Variation: There is significant variation in price and shipping costs across product categories. Some product categories have significantly higher prices and shipping costs than others.
Effect of Weight and Dimensions: In general, products with larger weights and dimensions tend to have higher prices and shipping costs. This makes sense since production and shipping costs are usually proportional to the size and weight of the product.
Role of Product Categories: Different product categories have different characteristics. For example, the category "agro_industry_and_commerce" tends to have products with larger weights and higher shipping costs than the category "art".
3. The Influence of Purchase Time on Order Status
Objective: Analyze whether there are specific time patterns when orders tend to experience delays or status changes. Data from olist_orders_dataset can be used to understand these trends. Description: This graph illustrates the relationship between purchase time and order status. There are three main metrics measured: total time to customer delivery, average delivery time, and average other time.

Overall, the graph shows that orders with a status of “delivered” have a shorter total time to customer delivery time than orders with a status of “canceled.” This suggests that faster order fulfillment tends to result in higher success rates. Additionally, the average other time for canceled orders is also higher, which may indicate additional bottlenecks or issues that are causing cancellations.

4. Seller Performance
Objective: Assess seller performance based on their location. By linking data from olist_sellers_dataset and olist_order_items_dataset, you can determine whether seller location affects their sales volume. Description: This graph provides a clear picture of the sales performance across cities in Brazil. Sao Paulo significantly dominates in terms of total orders and total sales, far above other cities. This shows that Sao Paulo is a very important and potential market for this business. Other cities such as Ibitinga, Curitiba, and Rio de Janeiro also contribute significantly to total sales, but are still far below Sao Paulo.

From this graph, we can draw several important conclusions:

Market Concentration: Most of the sales activity is concentrated in a few large cities such as Sao Paulo and Rio de Janeiro. This indicates that there is untapped market potential in other cities.
Growth Potential: Cities such as Ibitinga, Curitiba, and Rio de Janeiro have significant growth potential. With the right marketing strategy, the company can increase its market share in these cities.
Performance Differences: There are significant differences in sales performance across cities. This shows that local factors such as purchasing power, competition, and consumer preferences have a major impact on sales results.
5. The Influence of the Number of Product Photos on Sales
Objective: Measure whether the number of product photos affects sales levels. Data from olist_products_dataset and olist_order_items_dataset can be used for this analysis. Description: This pie chart provides a clear picture of the distribution of products based on the number of photos used. We can see that the majority of products (shown by the dark blue segment) have only 1 photo. This indicates that many sellers are not fully utilizing the potential of visuals to promote their products. In contrast, products with a higher number of photos (5 or more) have a much smaller proportion.

Importance of Visuals: The number of product photos used has a significant impact on the appeal of the product to consumers. Products with more photos tend to attract more attention and provide more complete information to potential buyers.
Growth Potential: There is still great potential to increase sales by increasing the number of product photos. Sellers can utilize additional features such as video, zoom, and 360 degrees to provide a more interactive visual experience to customers.
Market Segmentation: There are differences in visual strategies between sellers. Some sellers may focus more on products with high profit margins, so they are willing to invest more time and resources in creating quality product photos.
6. Comparison of Payment Methods to Payment Value
Objective: Analyze how payment methods (e.g., credit card vs. bank transfer) affect the average payment value. Data from olist_order_payments_dataset can be used for this insight. Description: From this graph, it can be concluded that credit cards are the most popular payment method and generate the largest transaction value. This indicates that most customers feel comfortable using credit cards to make transactions. Additionally, the low transaction value in the "not_defined" category shows the importance of ensuring that payment data is recorded correctly so that data analysis becomes more accurate. This information is very valuable for businesses to optimize payment strategies, such as offering special promotions for credit card users or providing more diverse payment options.

7. Purchase Patterns Based on Time
Objective: Analyze purchase patterns based on time of day or month to understand peak purchase periods. Data from olist_orders_dataset can be used to determine these patterns. Description: This graph shows a significant upward trend in the number of orders from October 2017 to a peak in early 2018. After that, the number of orders tends to be stable with little fluctuation. A fairly clear seasonal pattern is seen, with a significant increase occurring towards the end of the year (most likely due to the holiday season) and a relative decline at the beginning of the year.

Some insights that we can draw from this graph are:

Business Growth: Overall, the graph shows positive business growth during the observed period. This indicates that the business strategy implemented is quite effective.
Seasonal Pattern: The presence of a clear seasonal pattern indicates that seasonal factors have a significant influence on sales volume. The company needs to prepare for these seasonal fluctuations by managing inventory and marketing campaigns accordingly.
Growth Potential: Although there has been a significant increase, there is still potential for further increase in the number of orders, especially during off-peak periods.
8. Customer Profile Based on Location and Purchase Frequency
Objective: Identify customer characteristics such as location (city, state) and purchase frequency to determine the most active customer segments. Data from olist_customers_dataset and olist_order_items_dataset can provide these insights. Description: This visualization provides an interesting overview of customer profiles based on location and purchase frequency. Large cities such as Sao Paulo and Rio de Janeiro have a very significant number of customers, indicated by their larger land area. This indicates that these two cities are key markets for the business in question. Additionally, there is variation in purchase frequency across cities, indicated by the different colors or color intensity of each small box. Cities with lighter colors tend to have higher purchase frequencies.

Overall, this visualization can be used to identify cities with high growth potential, as well as to design more effective marketing strategies based on customer characteristics in each location. For example, businesses can allocate more marketing resources to cities with high purchase frequencies, or develop products and services that are more in line with customer preferences in each region. Additionally, this visualization can also help identify interesting purchasing patterns, such as certain seasons or special events that trigger increased sales in certain regions.

9. Sales Growth
Objective: Identify months with significant sales growth to plan better marketing and promotional strategies in the future. This analysis can also provide insights into strong or weak sales periods, helping in inventory planning and other business strategies. Description: Insights from the Sales Growth Chart:

Overall Trend: The chart depicts a general upward trend in sales over the period from October 2017 to July 2018. This indicates overall business growth and positive sales performance.
Specific Observations: Significant Growth Spikes: There are several periods of rapid sales growth, particularly in late 2017 and early 2018. These spikes might be attributed to factors such as successful marketing campaigns, product launches, or seasonal trends.

## Reporting

For any questions or further information regarding these visualizations, please open an issue or contact via [LinkedIn](https://www.linkedin.com/in/your-profile) or [GitHub](https://github.com/yourusername).

## Dataset Source

The visualizations are based on the Olist dataset, available at Postgreesql.
---
