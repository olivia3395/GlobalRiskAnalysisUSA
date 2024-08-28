# GlobalRiskAnalysisUSA
# Financial Exposure Forecasting Using Time Series Models

This project utilizes various time series models, including the **Prophet** model, to forecast financial exposure trends for the United States based on historical data. The project explores traditional time series approaches such as Exponential Smoothing and ARIMA, as well as machine learning methods like Long Short-Term Memory (LSTM) networks.

## Table of Contents
- [Dataset](#dataset)
- [Project Setup](#project-setup)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset contains historical financial exposure data by different CBS bank types over time. The data is sourced from the [Bank for International Settlements (BIS)].

### Download Dataset

- You can download the dataset used in this project from the following link: [BIS Data][(https://www.bis.org/statistics/credstat.htm)](https://www.bis.org/statistics/consstats.htm).





### Forecasting for Specific CBS Bank Types

The dataset contains different CBS bank types. You can adjust the models to forecast for a specific bank type, such as `4B` (Domestic Banks). The script automatically handles filtering by the desired bank type.

## Models

### Holt-Winters Exponential Smoothing

A traditional time series model that captures trend and seasonality in the data. The additive model is used here to account for quarterly seasonality.

### ARIMA

ARIMA models are used to capture the autoregressive and moving average components of the time series, after differencing to remove non-stationarity.

### LSTM (Long Short-Term Memory)

A deep learning model particularly well-suited for time series forecasting due to its ability to handle sequential data and long-term dependencies.

## Results

### Visualization of Forecasts

Several models are trained and visualized for comparison. Below are sample results from the Holt-Winters and LSTM models:

- **Holt-Winters Forecast:**
  ![Holt-Winters Forecast](./hw_forecast1.png)
  ![Holt-Winters Forecast](./hw_forecast2.png)
   ![Holt-Winters Forecast](./hw_forecast3.png) 
  
- **LSTM Model Forecast:**
  ![LSTM Forecast](./lstm_forecast.png) 

### Residual Analysis

Residuals from the models are analyzed to assess the model's performance and ensure that no significant patterns remain in the residuals.

## Acknowledgments

This project is built using several open-source libraries including:
- **pandas**: For data manipulation.
- **matplotlib**: For data visualization.
- **statsmodels**: For time series modeling.
- **tensorflow**: For building deep learning models.


Special thanks to the [Bank for International Settlements (BIS)] for providing the financial exposure data.

## Future Work

- Improve model performance by exploring advanced hybrid models or ensemble methods.
- Expand the dataset to include more countries or regions for a broader analysis.
- Incorporate external variables (e.g., macroeconomic indicators) to improve forecasting accuracy.



---
