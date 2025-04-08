# FULL STACK REAL-TIME PATIENT HEALTH MONITORING AND MULTIMODEL ANOMALY DETECTION

## Overview
This project provides an AI-driven remote patient monitoring system that collects real-time vitals from IoT wearable devices and applies multiple machine learning techniques for anomaly detection. It enables proactive healthcare intervention by identifying potential health risks.

## Repository Structure
```
AI-Remote-Patient-Monitoring/
│── data/                             # Sample dataset
│── src/                              # Data collection & AI model scripts
│   ├── datapreprocessing.py          # Data cleaning, normalization, and real-time streaming
│   ├── models/                       # Machine learning and deep learning models
│   │   ├── ml_model1.py              # Isolation Forest-based anomaly detection
│   │   ├── ml_model2.py              # One-Class SVM anomaly detection
│   │   ├── autoencoder_model.py      # Autoencoder-based deep learning anomaly detection
│   ├── realtime_anomaly_detection.py # Combines all models for real-time anomaly detection
│── models/                           # Saved trained models
│── README.md                         # Setup and usage instructions
│── requirements.txt                   # Dependencies
```

## Features
- **Real-time Data Collection & Preprocessing**: Cleans and normalizes incoming data streams.
- **Multiple AI-Powered Anomaly Detection Models**: Uses multiple techniques to detect abnormal vital signs.
- **Machine Learning-Based Detection**: Implements Isolation Forest and One-Class SVM for anomaly detection.
- **Deep Learning-Based Autoencoder**: Learns normal patterns and flags deviations as anomalies.
- **Real-time Anomaly Detection Pipeline**: Processes incoming patient vitals in real-time.
- **Custom Dataset Support**: Users can replace the sample dataset with their own.
- **Modular Codebase**: Organized using modules for maintainability and scalability.
- **Scalability**: Designed to integrate with real IoT devices in production.

## Setup
### Prerequisites
Ensure you have Python installed. Recommended version: Python 3.8+

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/AI-Remote-Patient-Monitoring.git
   ```
2. Navigate to the project folder:
   ```sh
   cd AI-Remote-Patient-Monitoring
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Running Data Preprocessing & Real-Time Streaming
To preprocess data and simulate real-time streaming:
```sh
python src/datapreprocessing.py
```

### Training and Running AI Anomaly Detection Models
#### Running Isolation Forest Model
1. Train and evaluate the Isolation Forest anomaly detection model:
   ```sh
   python src/models/ml_model1.py
   ```
2. The script will analyze patient vitals and indicate whether an anomaly is detected.

#### Running One-Class SVM Model
1. Train and evaluate the One-Class SVM anomaly detection model:
   ```sh
   python src/models/ml_model2.py
   ```
2. The script will analyze patient vitals and classify them as normal or anomalous.

#### Running the Autoencoder-Based Deep Learning Model
1. Train and evaluate the autoencoder anomaly detection model:
   ```sh
   python src/models/autoencoder_model.py
   ```
2. The script will train an autoencoder to learn normal patterns and flag anomalies based on reconstruction error.

### Running Real-Time Anomaly Detection Pipeline
To run real-time anomaly detection using all models:
```sh
python src/realtime_anomaly_detection.py
```
This script will load trained models, process incoming data batches, and flag anomalies using an ensemble method.

### Using a Custom Dataset
- Replace the sample dataset in `data/sample_vitals.csv` with your own data.
- Ensure the dataset follows the same structure (columns: heart_rate, spo2, ecg, bp_systolic, bp_diastolic).

## Example Output
When an anomaly is detected:
```
Anomaly Detected: Possible health risk identified!
```
For normal vitals:
```
Normal: No immediate health concerns.
```

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License.

