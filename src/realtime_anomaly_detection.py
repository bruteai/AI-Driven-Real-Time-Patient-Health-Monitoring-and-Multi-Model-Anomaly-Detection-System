# realtime_anomaly_detection.py
import pandas as pd
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
# Import the generator and preprocessing functions from datapreprocessing.py
from datapreprocessing import load_data, preprocess_data, realtime_data_generator

def load_models():
    """Load all trained models from disk."""
    with open('isolation_forest_model.pkl', 'rb') as f:
        isolation_forest = pickle.load(f)
    with open('one_class_svm_model.pkl', 'rb') as f:
        one_class_svm = pickle.load(f)
    autoencoder = load_model('autoencoder_model.h5')
    return isolation_forest, one_class_svm, autoencoder

def detect_anomalies(batch, isolation_forest, one_class_svm, autoencoder, autoencoder_threshold):
    """Apply each model on the batch and return their predictions."""
    # Work with numeric features only
    features = batch.select_dtypes(include=['float64', 'int64'])
    # IsolationForest prediction: 1 (normal) or -1 (anomaly)
    preds_if = isolation_forest.predict(features)
    # OneClassSVM prediction: 1 (normal) or -1 (anomaly)
    preds_svm = one_class_svm.predict(features)
    # Autoencoder prediction based on reconstruction error
    reconstructions = autoencoder.predict(features)
    mse = np.mean(np.power(features.values - reconstructions, 2), axis=1)
    preds_autoencoder = np.where(mse > autoencoder_threshold, -1, 1)
    return preds_if, preds_svm, preds_autoencoder

if __name__ == '__main__':
    file_path = 'data.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    generator = realtime_data_generator(data, batch_size=10)
    isolation_forest, one_class_svm, autoencoder = load_models()
    
    # Set the autoencoder threshold.
    # (In practice, you might calculate this on a validation set; here we use a fixed value for demonstration.)
    autoencoder_threshold = 0.5

    print("Starting real-time anomaly detection...\n")
    for i in range(5):  # simulate processing 5 batches
        batch = next(generator)
        preds_if, preds_svm, preds_autoencoder = detect_anomalies(batch, isolation_forest, one_class_svm, autoencoder, autoencoder_threshold)
        print(f"Batch {i+1} results:")
        print("  IsolationForest predictions:", preds_if)
        print("  OneClassSVM predictions:   ", preds_svm)
        print("  Autoencoder predictions:   ", preds_autoencoder)
        # Combine results: flag as anomaly if at least two models flag it (-1)
        combined = np.vstack([preds_if, preds_svm, preds_autoencoder])
        majority_anomaly = np.apply_along_axis(lambda x: np.sum(x == -1) > 1, axis=0, arr=combined)
        print("  Combined anomaly detection:", majority_anomaly)
        print("-" * 50)
        time.sleep(1)
