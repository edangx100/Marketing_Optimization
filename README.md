# Direct Marketing Revenue Optimization - Executive Summary

## Executive Overview

This analysis applies a data-driven strategy to optimize revenue from direct marketing campaigns. Leveraging a dataset of 1,615 bank clients, **6 models** were trained and applied to identify high-propensity targets and assign optimal offers.

The solution targets the top 15% of clients with personalized offers, achieving an expected revenue of **$699.54** from 96 targeted clients. 95 clients (99.0% of targets) are assigned Consumer Loan offers, 1 client (1.0%) receives Credit Card offers, and 0 clients are assigned Mutual Fund offers due to lower expected revenue per client.

## Dataset Preparation
Data from the `soc_dem`, `products`, `inflow`, and `sales` spreadsheets were merged to create a consolidated dataset of 1,615 bank clients for analysis.

The dataset was split into a training set (60%, 969 clients) and a test set (40%, 646 clients) for targeting evaluation. The top 15% of clients from the test set (96 individuals) were selected for targeting.

## Methodology: 6-Stage Targeting Strategy

| Stage                            | Model(s) used                                                                     | Output                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **1. Score purchase propensity** | 3 **binary classifiers**<br> • `Sale_CL_model`: Predicts Consumer Loan purchase probability <br> • `Sale_CC_model`: Predicts Credit Card purchase probability <br> • `Sale_MF_model`: Predicts Mutual Fund purchase probability | For every client *i* a calibrated probability  $\hat p_{ip}$ (e.g. "Jane has a **31 %** chance of taking a consumer loan").    |
| **2. Score conditional revenue** | 3 **regressors**<br> • `Rev_CL_model`: Predicts revenue if Consumer Loan sale occurs<br> • `Rev_CC_model`: Predicts revenue if Credit Card sale occurs <br> • `Rev_MF_model`: Predicts revenue if Mutual Fund sale occurs           | For the same client *i* a point estimate of revenue **if** the sale occurs, $\hat r_{ip}$ (e.g. "expected loan margin \$135"). |
| **3. Combine to expected value** | —                                                                    | $\hat{\text{EV}}_{ip} \;=\; \hat p_{ip}\times \hat r_{ip}$ for each product *p*                                                |
| **4. Pick the best offer**       | —                                                                        | `best_offer_i` = $$argmax_p \hat{\text{EV}}_{ip}$$ <br>`best_EV_i` = $$\max_p \hat{\text{EV}}_{ip}$$        One optimal offer per client with expected value justification                         |
| **5. Build the final list**      | —                                                                   | Rank clients by `best_EV_i`, slice the top 15 %.                                                                               |
| **6. Revenue forecast / lift**   | —                                                                           | Sum of `best_EV_i` in the list                                                                                                 |


## Key Results: High-Propensity Client Lists

### Model Performance
| Product | Sales Model F1 | Revenue Model R² | Revenue Model RMSE | Training Set |
|---------|----------------|------------------|-------------------|--------------|
| Consumer Loan | 0.720 | 0.103 | 10.914 | 773 clients |
| Credit Card | 0.685 | -0.027 | 6.887 | 773 clients |
| Mutual Fund | 0.669 | -0.001 | 7.013 | 773 clients |

### Propensity Analysis by Product

**Consumer Loan High-Propensity Clients:**
- **Average Propensity**: 33.7% across all clients
- **Targeting Strategy**: 95 clients (99.0% of targets) assigned CL offers

**Credit Card High-Propensity Clients:**
- **Average Propensity**: 26.4% across all clients
- **Targeting Strategy**: 1 client (1.0% of targets) assigned CC offers

**Mutual Fund High-Propensity Clients:**
- **Average Propensity**: 20.0% across all clients
- **Targeting Strategy**: 0 clients assigned MF offers (lower expected value)

### Optimal Offer Assignment Strategy

**Client Targeting Decision Logic:**
- **Consumer Loans**: Assigned to clients with highest CL expected value (95 clients)
- **Credit Cards**: Assigned to clients with highest CC expected value (1 client)  
- **Mutual Funds**: Not assigned due to lower expected revenue per client

**Expected Revenue by Strategy:**
- **Total Expected Revenue**: $699.54 from 96 targeted clients
- **Average Expected Revenue per Client**: $2.76
- **Lift vs Baseline Targeting**: 164.2%
- **Revenue Distribution**: CL (99.0%), CC (1.0%), MF (0.0%)


## Recommendations

### Immediate Actions
1. **Deploy Targeting Strategy**: Implement the 96-client targeting list ([targeted_clients.csv](./targeted_clients.csv)) for next campaign
2. **Monitor Performance**: Track actual vs. expected revenue for model validation
3. **Iterate Models**: Retrain with new campaign results for continuous improvement

### Long-term Strategy
1. **Expand Feature Set**: Incorporate additional behavioral and transactional data
2. **Advanced Modeling**: Experiment with ensemble methods and deep learning
