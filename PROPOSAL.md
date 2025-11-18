## Swiss Population Forecasting: A Model Comparison Approach

Category: Statistical Modeling • Machine Learning • Time Series Analysis

**1. Problem Motivation**

Population forecasting is essential for public policy, housing planning, pension system sustainability, and long-term infrastructure investment. Switzerland publishes annual demographic data, but forecasting future values is non-trivial: population trends depend on birth rates, mortality, and migration flows.

Most official forecasts rely on complex demographic models.
In contrast, this project focuses on data-driven statistical forecasting and investigates:

Can simple machine learning models outperform a naïve constant-growth benchmark?

This question is directly aligned with the course’s goal: implement, compare, and evaluate several predictive models using real data.

**2. Data & Legality**

All data will come from fully legal and open official sources:

Data Type	Source	Access
Annual Swiss Population (Total)	Swiss Federal Statistical Office (BFS)	Open data (CSV)
Birth rate, mortality, migration (optional)	BFS	Open data
Age-group population (optional)	BFS	Open data

No scraping is needed.
All datasets can be downloaded as public CSV files from the BFS website.

**3. Methodology**
Step 1 — Data Preparation

Load BFS population time series (e.g., 1950–2023)

Clean and transform into annual time-series format

Engineer features such as:

lagged population values

yearly growth rate

multi-year moving averages

Step 2 — Forecasting Models
Baseline Model (Required)

Naïve Constant-Growth Model
Uses historical % average growth to predict future population.

This model serves as the benchmark:

If machine learning cannot beat this, it is not useful.

Machine Learning & Statistical Models
Model	Purpose
Linear Regression	Model population as linear function of time & lagged values
AR / ARIMA (statsmodels)	Capture persistence in population growth
Random Forest Regression	Learn non-linear relationships in the time series

All models will be trained on historical data and evaluated on held-out years.

Step 3 — Model Evaluation

Each model will be evaluated using a formal model comparison procedure:

RMSE (root mean squared error)

MAE (mean absolute error)

MAPE (percentage error)

Train/test split (e.g., train on 1950-2010, test on 2011-2023)

A ranking table will show which model performs best.

**4. Expected Challenges**

Time-series forecasting with limited sample size

Overfitting risk in non-linear ML models

Ensuring fair comparison across models

Feature engineering for long-term demographic trends

**5. Success Criteria**

The project is successful if:

At least one ML model clearly outperforms the naïve baseline

Results are validated, plotted, and statistically interpretable

Code is modular, documented, and tested

A clear conclusion is obtained:

Which model forecasts Swiss population most accurately?

**6. Stretch Goals (Optional/not sure yet)**

Age-group-specific forecasting (0–19, 20–64, 65+)

Migration-only modeling scenario

Cross-country comparison (Switzerland vs France/Germany)

Streamlit dashboard for interactive forecasting