# ml_model1.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

def load_data(file_path):
    """Load and fill missing values in the dataset."""
    data = pd.read_csv(file_path)
    data.fillna(method='ffill', inplace=True)
    return data

def train_isolation_forest(data):
    """Train an IsolationForest model on the numeric features."""
    features = data.select_dtypes(include=['float64', 'int64'])
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(features)
    return model

def evaluate_model(model, data):
    """Evaluate the model by printing counts of predicted anomalies."""
    features = data.select_dtypes(include=['float64', 'int64'])
    predictions = model.predict(features)
    # Note: In IsolationForest, 1 = normal, -1 = anomaly
    data['anomaly'] = predictions
    print("IsolationForest anomaly prediction counts:")
    print(data['anomaly'].value_counts())

if __name__ == '__main__':
    file_path = 'data.csv'
    data = load_data(file_path)
    model = train_isolation_forest(data)
    evaluate_model(model, data)
    # Save the model to disk
    with open('isolation_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
