# High-traffic-prediction-recipes
This script processes a dataset of recipes, conducts data validation,  provides visualizations for key metrics, and applies Decision Tree and  Logistic Regression models to predict website traffic based on recipe attributes.

# Data Validation and Preprocessing

## Overview

This documentation details the validation and preprocessing applied to a recipes dataset, initially containing 892 rows across 8 columns.

## Data Columns and Preprocessing

- **recipe**
  - **Datatype**: Integer
  - **Validations**: 
    - No missing values 
    - Unique identifiers

- **calories**
  - **Datatype**: Float
  - **Validations**: 
    - Some missing values (approx. 5%); these were dropped

- **carbohydrates, sugar, protein**
  - **Datatype**: Float
  - **Validations**: 
    - No missing values post dropping rows with missing calory values

- **category**
  - **Datatype**: Object (Converted to 'category')
  - **Validations**: 
    - Merged 'Chicken Breast' into 'Chicken'
    - Total categories reduced from 11 to 10

- **servings**
  - **Datatype**: Object (Considered converting to 'int')
  - **Validations**: 
    - Rows with '4 as snack' and '6 as snack' were dropped

- **high_traffic**
  - **Datatype**: Categorical
  - **Validations**: 
    - Null values converted to 'No', indicating 'No High traffic'

## Initial Data Analysis

- **Target Variable**: Noted slight class imbalance; to be addressed during train-test split
- **Numerical Variables**: Displayed log-normal distribution; unchanged for model utilization
- **Categorical Variables**: 
  - Noted frequency of recipes with 4 servings 
  - Predominance of 'Chicken' category
- **Traffic Insights**: Beverages rarely yielded high traffic. Vegetables, potato, and pork typically did.

## Model Development

- **Approach**: Utilized Decision Tree Classifier and Logistic Regressor
- **Metrics**: Prioritized specificity and precision. Also monitored F1 score and accuracy.
- **Features**: calories, carbohydrate, sugar, protein, category, servings (Excluded: recipe, as it's only an identifier)
- **Preprocessing Steps**: 
  - Encoded categorical features
  - Scaled numerical features
  - Addressed class imbalance during train-test split
- **Performance Evaluation**: Decision Tree showcased superior precision, achieving the 80% KPI target

## Recommendations

1. Contemplate raising the model's threshold for higher prediction certainty
2. Explore deploying the model via an API or app interface
3. Augment dataset with more features, like number of steps or ingredients used
4. Distinguish between high traffic and general traffic for richer insights; consider regression modeling for traffic prediction


