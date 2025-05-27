# 🍱 Restaurant Visitor Forecasting

This project forecasts daily visitor counts for restaurants using time series and machine learning techniques. Built as a solution to the [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) challenge hosted by Recruit Holdings, it simulates a real-world scenario where restaurants must predict customer demand to optimize staffing and ingredient purchasing.

---

## 🔍 Problem Overview

Restaurants often struggle with accurately forecasting customer traffic, especially around holidays or special events. Recruit Holdings, which owns restaurant reservation and POS platforms in Japan, provided a unique dataset combining reservation logs, store metadata, and visit history.

The goal: Predict the number of visitors for a given restaurant on a future date to support operational planning.

---

## 🧠 Solution Approach

This project builds a robust time series forecasting pipeline using:

### ✅ Data Sources
- `air_visit_data.csv`: Daily visit logs (target)
- `air_reserve.csv`, `hpg_reserve.csv`: Reservation logs
- `store_id_relation.csv`: Mapping between air and hpg systems
- `air_store_info.csv`, `hpg_store_info.csv`: Genre, area, geolocation
- `date_info.csv`: Day-of-week, holiday flags

---

## 🛠️ Pipeline Steps

### 1. **Data Preprocessing**
- Merges multiple relational files into a unified format.
- Handles reservation lead times and ensures consistent store/date joins.

### 2. **Feature Engineering**
- Adds rich time-based features including:
  - Lag features: visitors from the last 1, 7, 14 days
  - Rolling stats: mean/std over 3/7/14-day windows
  - Reservation aggregates: lead times, total reserve visitors
  - Categorical encodings: genre, area, day-of-week, holiday flags

### 3. **Modeling with LightGBM**
- Trains a LightGBM regressor on all stores combined.
- Uses time-based validation and early stopping.
- Evaluates using RMSLE (Root Mean Squared Log Error), consistent with the competition metric.

### 4. **Prediction & Submission**
- Automatically generates predictions for all restaurant/date pairs in the test set.
- Outputs `predictions/prediction.csv` in the required format.

---

## 🔢 Evaluation Metric

All models are evaluated using **Root Mean Squared Logarithmic Error (RMSLE)**:

$$
RMSLE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( \log(p_i + 1) - \log(a_i + 1) \right)^2 }
$$

This metric penalizes underestimation more than overestimation and handles skewed visitor distributions well.

---

## 📈 Results

- Achieved an average RMSLE **< 0.40** using a tuned LightGBM model.
- Explored Prophet and ARIMA for ensembling, but settled on LightGBM due to its superior performance and generalization.
- Applied cross-validation and best global alpha tuning.

---

## 💡 Key Takeaways

- Real-world demand forecasting requires temporal awareness, holiday sensitivity, and careful feature crafting.
- Gradient-boosted trees like LightGBM work exceptionally well with tabular + temporal data.
- Ensemble models (Prophet + LightGBM) can help—but tuning and validation are critical.
