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

  - Handle missing values
  - Encode categorical variables (market_segment_type, type_of_meal_plan, room_type_reserved)
  - Scale/transform numeric features where needed
  - Train-test split (e.g., 80/20)
  - Save preprocessing pipeline using joblib for reproducibility

- Model Training

  - Train LightGBM classifier
  - Use hyperparameter tuning via Optuna or GridSearchCV
  - Track parameters, metrics, and artifacts in MLflow

- Model Evaluation

  - Metrics:
    - Accuracy
    - Precision, Recall, F1-score
    - ROC-AUC
    - Save confusion matrix and feature importance plots
    - Register best model in MLflow Model Registry

- Model Versioning & Tracking (MLflow)

  - Log:
    - Dataset version
    - Preprocessing pipeline version
    - Model parameters & metrics
    - Trained model artifacts

- Monitoring

  - Set up MLflow UI to monitor experiments
  - Potential integration with EvidentlyAI or custom scripts for data drift detection (future extension)
 
## How the Model Learns and Adapts

- The current model was trained on historical booking data, which captures past customer behavior patterns. However, booking behavior can change over time as new client patterns emerge. To ensure the model stays accurate and relevant, the system must continuously learn and adapt.

- Continuous Data Ingestion & Retraining:

  - Now: The model predicts based on patterns it saw in the past (e.g., long lead times often mean more cancellations).

  - Improvement: Continuously feed the system with new reservations and their actual outcomes (fulfilled or canceled). For example: A client books today → the model predicts → after the stay date, the true outcome is logged as new labeled data.
  - Benefit: Retraining periodically (weekly or monthly) ensures the model evolves with both old and new data.

- Online Learning (Advanced Option):
  - Instead of waiting for bulk retraining, incremental learning algorithms (like River, Vowpal Wabbit, or LightGBM in online mode) can be used.
  - How it works: The model updates itself each time a new data point arrives.
  - Benefit: The model adapts in near real time to shifting customer behaviors, such as sudden changes during holiday periods.

- Data & Concept Drift Monitoring:
  - Data Drift: Input features change (e.g., average room price distributions shift).
  - Concept Drift: The relationship between features and outcomes changes (e.g., previously, “many special requests = more likely to honor,” but this flips later).
  - Tools: EvidentlyAI or WhyLogs can detect these drifts.
  - Action: Once drift is detected, retraining is triggered automatically.

- Active Learning Loop:
  - The model is deployed with a feedback mechanism. For example: If it predicts high cancellation risk, hotel staff can reconfirm with the guest.
  - The system records whether this action was effective and feeds the result back into training.
  - Benefit: The model learns not only from raw booking outcomes but also from business interventions.

- Practical Setup in the Project
  - Data ingestion pipeline: Collect fresh booking data + outcomes.
  - Model registry (MLflow): Track and version old and new models.
  - Retraining schedule: Batch (weekly/monthly) or online (continuous).
  - Deployment monitoring: Log prediction distributions and compare with actual outcomes.
  - Feedback loop: Trigger retraining when model performance drops.
