import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the data
df = pd.read_csv('c:/users/Mohammed/Desktop/project ML/monthly_milk_production.csv', index_col='Date', parse_dates=True)
df.index.freq = 'MS'

# Plot the data
df.plot(figsize=(12, 6))

# Perform seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(df['Production'])
results.plot()

# Train-test split
train = df.iloc[:156]
test = df.iloc[156:]

# Scale the data
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Create time series generator
n_input = 3
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# Build LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(generator, epochs=50)

# Plot training loss
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)

# Make predictions on test data
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred) 
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse scaling for predictions
true_predictions = scaler.inverse_transform(test_predictions)

# Evaluate predictions
rmse = sqrt(mean_squared_error(test['Production'], true_predictions))
print("Root Mean Squared Error:", rmse)
