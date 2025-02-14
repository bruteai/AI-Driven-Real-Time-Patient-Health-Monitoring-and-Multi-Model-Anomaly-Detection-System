# datapreprocessing.py
import pandas as pd
import numpy as np
import time

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Fill missing values and normalize numeric columns."""
    data.fillna(method='ffill', inplace=True)
    # Select numeric columns for normalization
    vitals = data.select_dtypes(include=[np.number])
    normalized_data = (vitals - vitals.mean()) / vitals.std()
    data[normalized_data.columns] = normalized_data
    return data

def realtime_data_generator(data, batch_size=10):
    """
    Generator that simulates real-time streaming by yielding data batches.
    Loops over the dataset and pauses briefly between batches.
    """
    num_rows = data.shape[0]
    index = 0
    while True:
        if index >= num_rows:
            index = 0  # Restart for simulation
        batch = data.iloc[index:index+batch_size]
        index += batch_size
        yield batch
        time.sleep(1)  # simulate delay

if __name__ == '__main__':
    file_path = 'data.csv'  # Make sure this is the correct path to your dataset file
    data = load_data(file_path)
    data = preprocess_data(data)
    generator = realtime_data_generator(data)
    for i in range(5):  # simulate printing 5 batches
        batch = next(generator)
        print("Batch {}:\n{}".format(i+1, batch))
