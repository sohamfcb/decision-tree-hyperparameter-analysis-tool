# House Price Predictor
### Overview

This project is a machine learning model designed to predict house prices using a RandomForestRegressor with 200 estimators. The dataset contains various features related to houses, and the model predicts the house price based on these features. Matplotlib and Seaborn were used for Exploratory Data Analysis (EDA), which helps visualize data trends and patterns before building the model.

### Features

 - Predicts house prices based on relevant features.
 - Uses RandomForestRegressor for robust and accurate predictions.
 - Includes detailed EDA for better understanding of the data.


### Exploratory Data Analysis (EDA)

 - Matplotlib: Used for creating histograms, bar charts, and scatter plots.
 - Seaborn: Used for heatmaps and pair plots to understand correlations between features.

### Model Details

  - RandomForestRegressor:
  - n_estimators: 200
  - Max depth and other hyperparameters are tuned for performance.
  - Trained and validated on a portion of the dataset to ensure accuracy.

### Requirements

    numpy
    pandas
    scikit-learn
    matplotlib
    seaborn
