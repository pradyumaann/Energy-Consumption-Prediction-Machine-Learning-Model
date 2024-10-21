# Energy Consumption Forecasting with XGBoost

This project focuses on forecasting energy consumption using the XGBoost machine learning algorithm. The model leverages historical hourly energy data from the PJM interconnection region, using time-series features to predict future energy usage.

## Project Overview

The objective of this project is to develop an accurate model for predicting energy demand, which is crucial for resource planning and grid management. The workflow includes:

- **Data Preprocessing**: Clean and prepare the dataset, including setting the correct datetime index. The data is split into training (2008-2015) and testing (2015-2018) sets.
- **Feature Engineering**: Time-based features (hour, day of the week, month, quarter, year, etc.) are created to account for seasonality and trends in the energy usage data.
- **Model Training**: The XGBoost regressor is trained on the extracted features, with early stopping enabled to prevent overfitting. Evaluation is performed on the test set to monitor the model's performance.
- **Prediction & Visualization**: Predictions are compared to actual data, both for the entire test period and a specific week (April 2018), to visualize the accuracy of the forecasts.
- **Evaluation**: The model's performance is measured using the Root Mean Squared Error (RMSE) on the test set.

## Features

- **XGBoost Model**: Utilizes the powerful XGBoost algorithm for time-series forecasting.
- **Custom Feature Engineering**: Time-based features are engineered to capture important patterns in the data.
- **Visualization**: Detailed plots showing both actual and predicted energy usage for transparent model evaluation.

## Dependencies

To run this project, install the following dependencies:

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- xgboost
- scikit-learn

You can install these using:

pip install -r requirements.txt

## How to Use

1. Clone the repository:

   git clone https://github.com/your-username/energy-consumption-forecasting.git
   
2. Navigate to the project directory:

   cd energy-consumption-forecasting
   
3. Ensure the dataset (`PJME_hourly.csv`) is in the project folder.

4. Run the script to forecast energy consumption:

   python forecast_energy.py
   
## Results

The model is evaluated using the RMSE metric, and the predictions are visualized to show the performance on a weekly level. A specific week in April 2018 is highlighted to demonstrate how closely the model's predictions align with actual energy consumption patterns.

## Example Plots

- **Energy Consumption Predictions**: Visual comparison between the true energy usage and model predictions for the test set.
- **Weekly Forecast**: Detailed visualization of the model's forecast for a single week in April 2018.

## License

This project is licensed under the MIT License.

---

