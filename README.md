# FraudLens — Fraud Detection System

An explainable transaction fraud detection system combining rule-based detection, Isolation Forest anomaly detection, and risk-based case prioritization.

## Features

- Rule-based transaction monitoring
- Isolation Forest anomaly detection
- Transaction risk scoring and alert generation
- Case prioritization for investigation
- Streamlit dashboard for exploring alerts
- SQLite storage and Power BI reporting exports

## Technology Stack

- Python
- Pandas and Scikit-learn
- Streamlit
- SQLite and SQL
- Power BI

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fraudlens-fraud-detection.git
cd fraudlens-fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the demo

```bash
python run_demo.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

## Demo Results

On the included synthetic dataset, the demo processed 18,042 transactions and generated 157 alerts grouped into 71 cases.

Reported evaluation metrics against the included labels:

- Precision: 89.2%
- Recall: 94.6%
- F1 score: 0.92
- Case precision: 77.5%

These results are specific to the included synthetic data and should not be interpreted as real-world fraud detection performance.

## Limitations

The system is a demonstration project, not a production-ready banking fraud detection service. Performance must be validated on independent, representative, labelled transaction data before real-world use.

## Disclaimer

Use only authorized datasets. Do not commit sensitive customer information, credentials, or confidential financial records.
