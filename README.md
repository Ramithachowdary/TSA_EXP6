# Ex.No: 06 HOLT WINTERS METHOD

## Date: 16-05-2026
## Reg no: 212224240130

## AIM:
To implement the Holt Winters Method Model using Python.

## ALGORITHM:
1. Import the necessary libraries.
2. Load a CSV file containing airline passenger data into a DataFrame, parse the 'Month' column as datetime, set it as index, and perform initial data exploration.
3. Resample the data to monthly frequency (beginning of the month).
4. Plot the time series data and determine whether it has additive/multiplicative trend/seasonality.
5. Split into train and test data, create a model using Holt-Winters method, train with train data and evaluate the model predictions against test data.
6. Create the final model, predict future data and plot it.

## PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset, perform data exploration
data = pd.read_csv('/content/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
data.head()

# Resample and plot
data_monthly = data.resample('MS').sum()
data_monthly.head()
data_monthly.plot()
plt.title('Monthly Passengers')
plt.show()

# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
                        index=data_monthly.index)
scaled_data.plot()
plt.title('Scaled Data')
plt.show()

# Check for seasonality - decompose
decomposition = seasonal_decompose(data_monthly, model='additive')
decomposition.plot()
plt.show()

# multiplicative seasonality can't handle non-positive values
scaled_data = scaled_data + 1

# Split train and test data
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data  = scaled_data[int(len(scaled_data) * 0.8):]

# Create and train Holt-Winters model
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# TEST PREDICTION plot
ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(['train_data', 'test_predictions_add', 'test_data'])
ax.set_title('Visual Evaluation')
plt.show()

# Model performance metrics
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f'RMSE: {rmse}')
print(f'Standard Deviation: {scaled_data.std()}, Mean: {scaled_data.mean()}')

# Final model on full data and predict future
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul',
                                   seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly) / 4))

# FINAL PREDICTION plot
ax = data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(['data_monthly', 'final_predictions'])
ax.set_xlabel('Months')
ax.set_ylabel('Number of Monthly Passengers')
ax.set_title('Prediction')
plt.show()
```

## OUTPUT:

### Monthly Passengers Plot:
*(Plot of raw monthly airline passenger data from 1949–1960)*
<img width="509" height="417" alt="image" src="https://github.com/user-attachments/assets/475b68ae-c265-44fa-aa9f-8f0ee0208801" />

### Scaled Data Plot:
*(MinMaxScaler normalized time series scaled between 0 and 1)*
<img width="506" height="424" alt="image" src="https://github.com/user-attachments/assets/ccd22992-3794-4310-8494-b5a12c8f3a3b" />

### Decomposed Plot:
*(Four-panel plot showing Original, Trend, Seasonal, and Residual components)*
<img width="580" height="434" alt="image" src="https://github.com/user-attachments/assets/ee9fee42-3007-4b62-83a1-cabf7eac6772" />

### Test Prediction:
*(Line plot comparing train_data, test_predictions_add, and test_data)*
<img width="498" height="416" alt="image" src="https://github.com/user-attachments/assets/125cff73-5152-48ff-b568-d52ee35b771c" />

### Model Performance Metrics:
<img width="533" height="28" alt="image" src="https://github.com/user-attachments/assets/25673359-a493-43e4-8432-dcd9d4582a4d" />

### Final Prediction:
*(Line plot showing data_monthly vs final_predictions for the next quarter)*
<img width="528" height="414" alt="image" src="https://github.com/user-attachments/assets/254d6a9e-d4af-44ab-b893-a1f5d27d67f3" />

## RESULT:
Thus the program run successfully based on the Holt Winters Method model.
