# ml_model2.py
import pandas as pd
from sklearn.svm import OneClassSVM
import pickle

def load_data(file_path):
    """Load and fill missing values in the dataset."""
    data = pd.read_csv(file_path)
    data.fillna(method='ffill', inplace=True)
    return data

def train_one_class_svm(data):
    """Train a OneClassSVM model on the numeric features."""
    features = data.select_dtypes(include=['float64', 'int64'])
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    model.fit(features)
    return model

def evaluate_model(model, data):
    """Evaluate the model by printing counts of predicted anomalies."""
    features = data.select_dtypes(include=['float64', 'int64'])
    predictions = model.predict(features)
    data['anomaly'] = predictions
    print("OneClassSVM anomaly prediction counts:")
    print(data['anomaly'].value_counts())

if __name__ == '__main__':
    file_path = 'data.csv'
    data = load_data(file_path)
    model = train_one_class_svm(data)
    evaluate_model(model, data)
    # Save the model to disk
    with open('one_class_svm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
