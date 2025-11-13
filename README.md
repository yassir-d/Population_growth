# 🌍 Population Growth & Migration Simulator — Switzerland  
**Simulation • Time Series Modeling • Demography**

---

## 📘 Overview

This project builds a **data-driven population simulator** for Switzerland (or a specific canton such as Geneva). The goal is to model how the population and its age structure may evolve over the next 20–30 years, under uncertainty in:

- Birth rates  
- Death rates  
- Migration flows  

The project combines:

- **Official demographic data** (Swiss Federal Statistical Office, BFS)  
- **Statistical / machine learning models** to forecast demographic rates  
- **Simulation (cohort-component model + Monte Carlo)** to generate many possible future population paths  

Example questions the tool aims to answer:

> “What is the probability that the 65+ population share exceeds **X%** by year **Y**?”  
> “How do different migration assumptions affect the total population and ageing?”

---

## 🎯 Objectives

- Collect and clean **historical population, birth, death, and migration data**  
- Estimate **time-series models** for demographic rates (fertility, mortality, net migration)  
- Simulate **thousands of future population trajectories** using a cohort-component model  
- Quantify and visualize **uncertainty** (fan charts, confidence bands)

---

## 🧠 Methods (High-Level)

### Data
- Swiss Federal Statistical Office (BFS): population by age, births, deaths, migration  
- (Optional) Open Data Genève: canton-level details  

All sources are **open and legal**. No scraping of commercial websites.

### Modeling & Simulation
- Time-series / regression models for demographic rates (e.g. lagged linear models, simple AR-style models, or tree-based models such as Random Forest for rates)  
- Rolling or expanding **time-series validation** (train on early years, test on later years)  
- Cohort-component population model:
  - Apply predicted mortality → survivors  
  - Apply predicted fertility → new births  
  - Apply predicted migration → adjusted age groups  
- **Monte Carlo**: add stochastic noise to rates and simulate many future paths.

### Outputs
- Population projections (total and by age group)  
- Age structure metrics (e.g. share of 0–19, 20–64, 65+)  
- Dependency ratios (old-age / working-age)  
- Uncertainty intervals: “In X% of simulations, indicator Y exceeds threshold Z”

---


