# Direct Marketing Revenue Optimization - Executive Summary

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:367/format:webp/1*DbTurClYzPthFSM8ZDfUmw.jpeg" alt="Direct Marketing Strategy">
</p>

## Executive Overview

This analysis applies a data-driven strategy to optimize revenue from direct marketing campaigns. Leveraging a dataset of 1,615 bank clients, **6 models** were trained and applied to identify high-propensity targets and assign optimal offers.

The solution targets the top 15% of clients with personalized offers, achieving an expected revenue of **$691.39** from 96 targeted clients. 47 clients (49.0%) are assigned Consumer Loan offers, 38 clients (39.6%) receive Mutual Fund offers, and 11 clients (11.5%) are assigned Credit Card offers.

## Dataset Preparation
Data from the `soc_dem`, `products`, `inflow`, and `sales` spreadsheets were merged to create a consolidated dataset of 1,615 bank clients for analysis.

The dataset was split into a training set (60%, 969 clients) and a test set (40%, 646 clients) for targeting evaluation. The top 15% of clients from the test set (96 individuals) were selected for targeting.

## Methodology: 6-Stage Targeting Strategy

| Stage                            | Model(s) used                                                                     | Output                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **1. Score purchase propensity** | 3 **binary classifiers**<br> • `Sale_CL_model`: Predicts Consumer Loan purchase probability <br> • `Sale_CC_model`: Predicts Credit Card purchase probability <br> • `Sale_MF_model`: Predicts Mutual Fund purchase probability | For every client *i* a calibrated probability p̂<sub>ip</sub> (e.g. "Jane has a **31 %** chance of taking a consumer loan").    |
| **2. Score conditional revenue** | 3 **regressors**<br> • `Rev_CL_model`: Predicts revenue if Consumer Loan sale occurs<br> • `Rev_CC_model`: Predicts revenue if Credit Card sale occurs <br> • `Rev_MF_model`: Predicts revenue if Mutual Fund sale occurs           | For the same client *i* a point estimate of revenue **if** the sale occurs, r̂<sub>ip</sub> (e.g. "expected loan margin \$135"). |
| **3. Combine to expected value** | —                                                                    | EV<sub>ip</sub> = p̂<sub>ip</sub> × r̂<sub>ip</sub> for each product *p*                                                |
| **4. Pick the best offer**       | —                                                                        | `best_offer_i` = argmax<sub>p</sub> EV<sub>ip</sub> <br>`best_EV_i` = max<sub>p</sub> EV<sub>ip</sub>        One optimal offer per client with expected value justification                         |
| **5. Build the final list**      | —                                                                   | Rank clients by `best_EV_i`, slice the top 15 %.                                                                               |
| **6. Revenue forecast / lift**   | —                                                                           | Sum of `best_EV_i` in the list                                                                                                 |


## Key Results

### Model Performance
Six machine learning models using XGBoost were trained to support marketing optimization across three financial products:
#### 📊 Revenue Regression Models
- Consumer Loan (`Revenue_CL`)
- Credit Card (`Revenue_CC`)
- Mutual Fund (`Revenue_MF`)
#### 🛍️ Sales Classification Models
- Consumer Loan (`Sale_CL`)
- Credit Card (`Sale_CC`)
- Mutual Fund (`Sale_MF`)

| Product | Sales Model F1 | Revenue Model RMSE |
|---------|----------------|-------------------|
| Consumer Loan | 0.526 | 4.725 |
| Credit Card | 0.538 | 5.117|
| Mutual Fund | 0.429 | 8.590 |

### Propensity Analysis by Product

**Consumer Loan:**
  - **Average Propensity**: 33.2% across all clients
  - **Targeting Strategy**: 47 clients (49.0% of targets) assigned CL offers

**Credit Card:**
  - **Average Propensity**: 11.4% across all clients
  - **Targeting Strategy**: 11 clients (11.5% of targets) assigned CC offers

**Mutual Fund:**
  - **Average Propensity**: 8.0% across all clients
  - **Targeting Strategy**: 38 clients (39.6% of targets) assigned MF offers

### Optimal Offer Assignment Strategy

**Client Targeting Decision Logic:**
- **Consumer Loans**: Assigned to clients with highest CL expected value (47 clients)
- **Credit Cards**: Assigned to clients with highest CC expected value (11 clients)
- **Mutual Funds**: Assigned to clients with highest MF expected value (38 clients)

**Expected Revenue by Strategy:**
- **Total Expected Revenue**: $691.39 from 96 targeted clients
- **Average Expected Revenue per Client**: $2.78
- **Lift vs Baseline Targeting**: 159.0%
- **Revenue Distribution**: CL (49.0%), MF (39.6%), CC (11.5%)


## Recommendations
1. **Deploy Targeting Strategy**: Implement the 96-client targeting list ([targeted_clients.csv](./targeted_clients.csv)) for next campaign
2. **Monitor Performance**: Track actual vs. expected revenue for model validation
3. **Iterate Models**: Retrain with new campaign results for continuous improvement
