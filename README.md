## Overview

This project predicts whether a hotel reservation will be fulfilled (1) or canceled/no-show (0) using historical booking data.
The pipeline follows MLOps best practices to ensure reproducibility, scalability, and maintainability — covering:

- Data ingestion
- Data preprocessing
- Model training
- Model evaluation
- Experiment tracking & model versioning (MLflow)
- Deployment-ready monitoring setup

The chosen algorithm, LightGBM, provides high performance, low memory usage, and efficient handling of large datasets with mixed feature types.

## Problem Statement

Hotels often lose revenue due to last-minute cancellations or no-shows, which cause wasted inventory and missed opportunities.
Accurate prediction of whether a client will honor a reservation enables hotels to:

- Take preventive actions for high-risk bookings
- Optimize overbooking strategies
- Improve resource allocation and occupancy rates

# Goal:
Given historical reservation features, predict the booking status:

- 1 → Reservation fulfilled (client honours and shows up)
- 0 → Reservation canceled or no-show

## Dataset

Features:

Feature	Description
- lead_time: Number of days between booking date and arrival date
- no_of_special_requests: Count of special requests made by the customer
- avg_price_per_room:	Average price per room in USD
- arrival_month: Month of arrival (1–12)
- arrival_date: Date of arrival within the month
- market_segment_type: Type of market segment (e.g., Online, Offline, Corporate)
- no_of_week_nights: Number of weekday nights booked
- no_of_weekend_nights: Number of weekend nights booked
- type_of_meal_plan: Meal plan chosen by the customer
- room_type_reserved: Reserved room type
- booking_status: Target: 1 = honoured, 0 = canceled/no-show

## MLOps Pipeline

- Data Ingestion

Raw CSV file stored in data/raw/
Ingestion script (src/data_ingestion.py) reads data, performs schema validation, and saves processed copy in data/processed/

- Data Preprocessing

- - Handle missing values
- - Encode categorical variables (market_segment_type, type_of_meal_plan, room_type_reserved)
- - Scale/transform numeric features where needed
- - Train-test split (e.g., 80/20)
- - Save preprocessing pipeline using joblib for reproducibility

- Model Training

- - Train LightGBM classifier
- - Use hyperparameter tuning via Optuna or GridSearchCV
- - Track parameters, metrics, and artifacts in MLflow

- Model Evaluation

- - Metrics:
- - - Accuracy
- - - Precision, Recall, F1-score
- - - ROC-AUC
- - - Save confusion matrix and feature importance plots
- - - Register best model in MLflow Model Registry

- Model Versioning & Tracking (MLflow)

- - Log:
- - - Dataset version
- - - Preprocessing pipeline version
- - - Model parameters & metrics
- - - Trained model artifacts

- Monitoring

- - Set up MLflow UI to monitor experiments
- - Potential integration with EvidentlyAI or custom scripts for data drift detection (future extension)