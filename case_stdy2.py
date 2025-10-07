import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


normal = np.random.normal(0, 1, (1000, 20))
fraud = np.random.normal(4, 1, (100, 20))
X = np.vstack([normal, fraud])
y = np.hstack([np.zeros(1000), np.ones(100)])  # 0=normal, 1=fraud

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]
autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(10, activation='relu'),
    layers.Dense(5, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

X_train = X_scaled[y == 0]
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0)

reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

threshold = np.percentile(mse, 95)
y_pred = (mse > threshold).astype(int)

print("Reconstruction error threshold:", round(threshold, 4))
print(classification_report(y, y_pred, digits=4))
