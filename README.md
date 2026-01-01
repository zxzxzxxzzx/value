# 🏠 HDB Valuation Calculator
## 🔥 Description
The HDB Valuation Calculator builds on existing data sets to provide users with an accurate estimation of their flats resale value based on various factors. This will serve as a decision-making tool to help our users make better decisions regarding the resale value of their property, based on pre-existing market trends. 

In this current market, it is difficult to make proper estimations for a flats resale price without having access to the data required to assess various factors. This can be troubling for newer homeowners who aren’t experienced in the field, leading to inaccurate valuations that can tank their overall return.

By centralizing and analysing publicly available resale transaction data, the HDB Valuation Calculator provides a transparent and data-driven solution. It empowers users with a clearer understanding of how factors influence resale value. This reduces reliance on guessing, ultimately promoting fairer transactions and better financial outcomes.

## 🔨 Backend
To understand the influence of each feature/input over the output obtained, we will be using Statistical Analysis & Exploratory Data Analysis to build a predictive model to calculate the resale price to a level of high accuracy. We will be building a regression model to carry out this requirement. 

There are initially a total of 9 features/inputs available from the dataset: 

- Date 
- Town
- Flat Type 
- Block 
- Street Name 
- Story Range 
- Floor Area 
- Flat Model 
- Lease Details

Out of these 9 features, we carried out testing to figure out which primary features we should focus on. This was done based on the dependency of each feature on the output. We landed on **Town, Flat Type, Storey Range, Floor Area, Flat Model, & Remaining Lease**

## ✨ Features
- 📈 **Price Prediction** – Polynomial regression model trained on historical resale data.
- 🛠 **Data Processing** – Automated cleaning, transformation, and feature engineering.
- 💻 **CLI Interface** – Simple command-line interaction for running predictions and analyses.
- 📊 **Visualization** – Generate charts for market trends, flat type analysis, and price vs. floor area.
- 📂 **Sample Dataset** – Included for quick testing and demonstration.

## 📂 Project Structure
```text
HDBPricePredictor
├── main.py                    # Entry point for the application
├── cli_interface.py           # Command-line interface
├── data_processor.py          # Data cleaning and transformation
├── hdb_polynomial_model.py    # Model training & prediction
├── visualizer.py              # Chart and graph generation
├── sample_data.csv            # Example dataset
└── pyproject.toml             # Dependencies & build configuration
```

## ⭐ Credits
https://www.w3schools.com/python/python_ml_getting_started.asp  
https://www.geeksforgeeks.org/  
https://data36.com/polynomial-regression-python-scikit-learn/  

https://replit.com/  
https://code.visualstudio.com/  
https://jupyter.org/  
https://colab.google/  

Datasets from https://data.gov.sg/  
Debugging assisted by https://replit.com/ai & https://chatgpt.com/ (_minor usage_)  
Final release in https://github.com/  

This project was developed by Souritra Samanta  
souritrasamanta@gmail.com  
Commonwealth Secondary School  

**PS: The testing notebook that I used to create the polynomial model will be released in Version 1.1 of the project.**
