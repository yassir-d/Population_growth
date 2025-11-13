# Project Proposal  
**Title:** Population Growth & Migration Simulator — Switzerland  
**Category:** Simulation & Modeling + Data Science + Machine Learning  

## 1. Problem Motivation

Population growth and migration are central to many economic and policy questions: pressure on housing, pensions, schools, and healthcare all depend on how the population evolves over time. Switzerland (and especially urban areas like Geneva) faces ageing, immigration, and changing birth rates, which make future population paths highly uncertain.

The goal of this project is to build a **data-driven population simulator** that models how the Swiss (or Geneva) population evolves over the next 20–30 years. The core idea is to combine:

- **Historical demographic data** (births, deaths, migration by age group),
- **Statistical / ML models** to estimate future demographic rates,
- **Simulation techniques** to generate many possible future population trajectories.

The final tool will answer questions such as:

> “Under current trends, what is the probability that the share of 65+ exceeds X% by year Y?”  
> “How do different migration scenarios affect total population and age structure?”

---

## 2. Data & Legality

All data will come from **open and official sources**:

- Swiss Federal Statistical Office (BFS): historical population by age, births, deaths, migration  
- (Optionally) Open Data Genève: canton-level demographics  

These are publicly available as CSV/Excel and can be used legally for analysis. No scraping of commercial websites is needed.

---

## 3. Methodology (Data Science + ML + Simulation)

### Step 1 — Data Preparation

- Load historical yearly (or quarterly) data on:
  - Population by age group
  - Birth rates, death rates
  - Net migration (inflows – outflows)
- Construct a panel/time-series of demographic rates:
  - Age-specific fertility rates
  - Age-specific mortality rates
  - Net migration rates by age or group
- Split data into **training and test periods** (e.g. 1990–2015 train, 2016–2023 test).

### Step 2 — Statistical / Machine Learning Estimation

For each rate (birth, death, migration), fit predictive models:

- **Time-series regression models** (e.g. AR, ARIMA-like or linear regression with lags)  
- And/or **tree-based models** (Random Forest / Gradient Boosting) using:
  - Past values of the rate
  - Overall economic indicators (if available, e.g. unemployment, GDP growth)

Validation:

- Rolling-origin or expanding-window time-series cross-validation  
- Compare to simple baselines (e.g. “last year’s rate”)  
- Metrics: MAE / RMSE for rate predictions, plus visual inspection of fitted vs actual rates.

These estimated and validated models give **forecast distributions** for future demographic rates.

### Step 3 — Population Simulation (Core Simulation Component)

Use a standard **cohort-component model** / discrete-time simulation:

- Start from the most recent observed population by age
- For each simulated year:
  - Apply predicted age-specific mortality to compute survivors  
  - Apply predicted fertility to estimate births and entry into age 0–4 group  
  - Apply predicted migration rates to adjust each age group  

To model **uncertainty**, combine the point predictions from ML with stochastic noise:

- Draw rates from distributions centered at the ML predictions  
- Run **many Monte Carlo simulations** (e.g. 1,000–10,000 paths)

Outputs per scenario:

- Total population  
- Age structure (e.g. share of 0–19, 20–64, 65+)  
- Dependency ratios (old-age / working-age)

From this, compute probabilities, e.g.:

> “In 2045, in 82% of simulations, the 65+ share exceeds 25%.”

### Step 4 — Visualization & Analysis

- Time-series plots of simulated population and age shares  
- Fan charts (prediction intervals) for key variables  
- Comparison of baseline vs alternative scenarios (e.g. high vs low migration)

---

## 4. Expected Challenges

- Handling data gaps and changes in age group definitions over time  
- Designing a robust validation strategy for time-series forecasts  
- Keeping the model interpretable while incorporating uncertainty

---

## 5. Success Criteria

The project will be considered successful if:

- At least one validated model can reasonably forecast key demographic rates  
- The simulator produces multiple plausible population trajectories  
- The tool can answer “what if” questions about ageing and migration with clear visualizations  
- Code is modular, documented, and tested (unit tests for core functions).

Stretch goals (if time permits):

- Add scenario comparison UI (e.g. Streamlit)  
- Include simple economic indicators (e.g. pension dependency ratio)
