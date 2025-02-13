# autoencoder_model.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load and fill missing values in the dataset."""
    data = pd.read_csv(file_path)
    data.fillna(method='ffill', inplace=True)
    return data

def preprocess_data(data):
    """Extract numeric features and scale them."""
    features = data.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def build_autoencoder(input_dim):
    """Build a deep autoencoder model."""
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    bottleneck = Dense(4, activation='relu')(encoded)
    # Decoder
    decoded = Dense(8, activation='relu')(bottleneck)
    decoded = Dense(16, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

if __name__ == '__main__':
    file_path = 'data.csv'
    data = load_data(file_path)
    processed_data, scaler = preprocess_data(data)
    input_dim = processed_data.shape[1]
    
    autoencoder = build_autoencoder(input_dim)
    es = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history = autoencoder.fit(processed_data, processed_data,
                              epochs=50,
                              batch_size=32,
                              shuffle=True,
                              callbacks=[es])
    
    # Calculate reconstruction error for each sample
    reconstructions = autoencoder.predict(processed_data)
    mse = np.mean(np.power(processed_data - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold
    
    print("Autoencoder reconstruction error threshold: {:.4f}".format(threshold))
    print("Number of anomalies detected by autoencoder: {}".format(np.sum(anomalies)))
    
    # Save the autoencoder model
    autoencoder.save('autoencoder_model.h5')
