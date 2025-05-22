# Restaurant Visitor Forecasting

## 1. Understand the Objective (Problem Framing)
### Goal: Predict the number of visitors to air restaurants for specific future dates (the test set).

- Predicting → Visitor counts on given days (test set)
- Granularity → Daily (air_store_id)
- Per each prediciton time → Visitor history, Reservations, Calendar Info, Metadata
- Seasonal Events → Japan's Golden Week

### Takeaway: This is a multi-store time series regression problem with metadata and exogenous signals like holidays and reservations.

## 2. Map Out Data Sources (Data Inventory)
I have:
- Visitor logs (air_visit_data.csv)
- Reservation logs (air & hpg)
- Calendar info (with holidays)
- Metadata (genre, area, lat/lon)
- Restaurant ID mappings (store_id_relation.csv)

Thought Process

```
air_visit_data
    |
    |-- join air_reserve on air_store_id + visit_date
    |-- join hpg_reserve (via store_id_relation) on visit_date
    |-- join air_store_info on air_store_id
    |-- join date_info on visit_date
```

### Takeaway: Before modeling, need to build a unified table per restaurant and date.

## 3. Feature Design (Signal Extraction)
- Past behavior that might influence the future → Rolling averages, lags (ex: last week’s traffic)
- External factors affect visits → Holidays, weekends, genre, reservations
- Store-specific patterns that matter → Area, genre (ex: some restaurants get busy on weekends, others don’t)

Build:
- Lag features: visitors last 1, 7, 14 days
- Rolling means: 7-day rolling average
- Calendar features: holiday flag, weekend, day-of-week
- Reservation features: total reserved visitors, average lead time
- Store metadata: one-hot encode or target encode genre/area

### Takeaway: Better features → better signal → better model generalization.

## 4. Choose and Justify Models (Model Design)
Which Models:
- Prophet: handles trend + seasonality well, easy holiday integration, great for individual stores.
- ARIMA: classical, works well with small, stationary series (ex: for stores with stable traffic).
- LightGBM: captures interactions across stores, dates, holidays, reservations, scales to full dataset.

Strategy: Combine model strengths—Prophet/ARIMA for pattern-specific insights, LightGBM for high-performing generalization.

### Takeaway: Multiple model families capture different dynamics. Ensemble or compare.

## 5. Validation Design (Test Strategy)
The test set is time-based, and includes holidays, need simulate this in training.

- Cannot randomly split the data, time series needs forward-only splits
- RMSLE (log-scale penalizes underpredictions more than over)

- Expanding window CV (e.g., train on Jan–Feb, validate on March; then train on Jan–March, validate on April)
- Metrics: RMSLE, MAPE for interpretability

### Takeaway: Evaluation setup must reflect real-world deployment—can’t use future data to predict the past.

## 6. Iterate and Improve (Modeling Cycle)

After the baseline is complete:
- Analyze residuals: where do I under/overpredict?
- Segment performance: which genres/areas perform worst?
- Try feature interaction terms (e.g., weekend × genre)
- Consider stacking models or ensembling predictions

### Takeaway: The best models come from iteration + intuition-driven debugging.