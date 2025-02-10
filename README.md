# Tourism-Experience-Analytics
Tourism Experience Analytics: Classification, Prediction, and Recommendation System
# Tourism Experience Analytics: Classification, Prediction, and Recommendation System

## Overview
This project focuses on analyzing tourism experiences using data science techniques. The goal is to enhance user experience by providing personalized recommendations, predicting user satisfaction, and classifying potential user behavior.

## Skills Utilized
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Data Visualization
- SQL
- Streamlit
- Machine Learning (Regression, Classification & Recommendation)

## Domain
**Tourism**

## Problem Statement
Tourism agencies and travel platforms aim to enhance user experiences by leveraging data to provide personalized recommendations, predict user satisfaction, and classify potential user behavior. This project involves analyzing user preferences, travel patterns, and attraction features to achieve three primary objectives: regression, classification, and recommendation.

## Business Use Cases
1. **Personalized Recommendations**: Suggest attractions based on users' past visits, preferences, and demographic data, improving user experience.
2. **Tourism Analytics**: Provide insights into popular attractions and regions, enabling tourism businesses to adjust their offerings accordingly.
3. **Customer Segmentation**: Classify users into segments based on their travel behavior, allowing for targeted promotions.
4. **Increasing Customer Retention**: By offering personalized recommendations, businesses can boost customer loyalty and retention.

## Objectives
### 1. Regression: Predicting Attraction Ratings
**Aim:**
Develop a regression model to predict the rating a user might give to a tourist attraction based on historical data, user demographics, and attraction features.

**Use Cases:**
- Travel platforms can use this model to estimate the satisfaction level of users visiting specific attractions.
- Personal travel guides can provide users with attractions most aligned with their preferences.

**Features:**
- User demographics: Continent, region, country, city.
- Visit details: Year, month, mode of visit (e.g., business, family, friends).
- Attraction attributes: Type (e.g., beaches, ruins), location, and previous average ratings.

**Target:**
- Predicted rating (on a scale, e.g., 1-5).

### 2. Classification: User Visit Mode Prediction
**Aim:**
Create a classification model to predict the mode of visit (e.g., business, family, couples, friends) based on user and attraction data.

**Use Cases:**
- Travel platforms can use this model to tailor marketing campaigns.
- Hotels and attraction organizers can plan resources based on predicted visitor types.

**Features:**
- User demographics: Continent, region, country, city.
- Attraction characteristics: Type, popularity, previous visitor demographics.
- Historical visit data: Month, year, previous visit modes.

**Target:**
- Visit mode (categories: Business, Family, Couples, Friends, etc.).

### 3. Recommendations: Personalized Attraction Suggestions
**Objective:**
Develop a recommendation system to suggest tourist attractions based on a user's historical preferences and similar users’ preferences.

**Use Cases:**
- Travel platforms can implement this system to guide users toward attractions they are most likely to enjoy.
- Destination management organizations can identify emerging trends and promote attractions accordingly.

**Types of Recommendation Approaches:**
- **Collaborative Filtering**: Recommend attractions based on similar users’ ratings and preferences.
- **Content-Based Filtering**: Suggest attractions similar to those already visited by the user.
- **Hybrid Systems**: Combine collaborative and content-based methods for enhanced accuracy.

**Features:**
- User visit history: Attractions visited, ratings given.
- Attraction features: Type, location, popularity.
- Similar user data: Travel patterns and preferences.

**Output:**
- Ranked list of recommended attractions.

## Approach
### 1. Data Cleaning
- Handle missing values in the transaction, user, and city datasets.
- Resolve discrepancies in categorical variables like VisitMode, AttractionTypeId.
- Standardize date and time format.
- Handle outliers in rating or other columns.

### 2. Preprocessing
- Encode categorical variables such as VisitMode, Continent, Country, and AttractionTypeId.
- Aggregate user-level features to represent each user's profile.
- Join relevant data from transaction, user, city, and attraction tables to create a consolidated dataset.
- Normalize numerical features such as Rating.

### 3. Exploratory Data Analysis (EDA)
- Visualize user distribution across continents, countries, and regions.
- Explore attraction types and their popularity based on user ratings.
- Investigate correlation between VisitMode and user demographics.
- Analyze distribution of ratings across different attractions and regions.

### 4. Model Training
#### **Regression Task:**
- Train a model to predict ratings based on user, attractions, transaction features, etc.

#### **Classification Task:**
- Train a classifier (e.g., Random Forest, LightGBM, or XGBoost) to predict VisitMode based on user and transaction features.

#### **Recommendation Task:**
- Implement collaborative filtering (using user-item matrix) to recommend attractions based on user ratings and preferences.
- Use content-based filtering based on attractions' attributes (e.g., location, type).

### 5. Model Evaluation
- Evaluate classification model performance using accuracy, precision, recall, and F1-score.
- Evaluate regression model using R2, MSE, etc.
- Assess recommendation system accuracy using metrics like Mean Average Precision (MAP) or Root Mean Squared Error (RMSE).

### 6. Deployment
- Build a Streamlit app that allows users to input their details (location, preferred visit mode) and receive:
  - A prediction of their visit mode (Business, Family, etc.).
  - Recommended attractions based on their profile and transaction history.
  - Display visualizations of popular attractions, top regions, and user segments in the app.

## End Output
A user-friendly Streamlit application where tourists can input their data and receive personalized recommendations for attractions, as well as get predictions on their likely visit mode.

