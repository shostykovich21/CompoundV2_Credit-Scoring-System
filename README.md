# Zeru Credit Score: A DeFi Creditworthiness Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A data-driven credit scoring model for Ethereum wallets based on their transaction history with the Compound V2 protocol. This project leverages a hybrid machine learning strategy to transform raw on-chain data into an interpretable and robust credit score. This model uses the three lagest json files: compoundV2_transactions_etherium_chunk_0, 1 and 2 as its datasets, which have not been uploaded for privacy concerns.  

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Machine Learning Strategy](#machine-learning-strategy)
- [Model Validation and Performance](#model-validation-and-performance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Generate Credit Scores](#1-generate-credit-scores)
  - [2. Run Sensitivity Analysis](#2-run-sensitivity-analysis)
  - [3. Perform In-depth Model Validation](#3-perform-in-depth-model-validation)
- [License](#license)

## Overview

In Decentralized Finance (DeFi), assessing the creditworthiness of an anonymous wallet is a critical challenge. This project implements a complete pipeline to address this problem by analyzing on-chain behavior. It ingests transaction data (deposits, borrows, repays, etc.), engineers features that reflect financial health and risk, and applies a transparent scoring model to generate a final score from 0 to 100.

The result is a reliable and data-driven metric for quantifying wallet risk and reliability.

## Key Features

*   **Robust Data Ingestion**: Efficiently loads and cleans data from multiple JSON files containing raw transaction records.
*   **Comprehensive Feature Engineering**: Creates over 20 meaningful features for each wallet, covering financial health, account history, activity patterns, and risk indicators.
*   **Interpretable Scoring Model**: The final score is a weighted average of three core components: **Health**, **Trust**, and **Risk**.
*   **Expert-in-the-Loop Logic**: The model incorporates domain-specific business rules to penalize high-risk events like liquidations or loan defaults, ensuring scores are practical and realistic.
*   **Weight Sensitivity Analysis**: Includes a script to test the model's stability by measuring how the top-ranked wallets change when model weights are adjusted.
*   **Detailed Validation Notebook**: A Notebook is provided for full model validation, visualizing score distributions and confirming the profiles of the highest and lowest-scoring wallets.

## Machine Learning Strategy

This project employs a hybrid approach that combines statistical modeling with expert-defined business rules. This strategy ensures the model is both data-driven and aligned with fundamental principles of credit risk.

1.  **Feature Engineering**: Raw transaction logs are aggregated to create a wallet-level feature set. Logarithmic transformations are applied to normalize skewed distributions (e.g., transaction amounts, account age).
2.  **Non-Linear Scaling**: A `sklearn.preprocessing.QuantileTransformer` is used to scale all features to a uniform distribution between 0 and 1. This is a powerful, non-parametric method that is robust to outliers and does not assume a specific distribution for the input data.
3.  **Component-Based Scoring**: Features are grouped into three logical components (Health, Trust, Risk). An intermediate score is calculated for each component, providing model interpretability.
4.  **Weighted Aggregation**: The final raw score is a linear combination of the three component scores, allowing for easy tuning of their relative importance.
5.  **Rule-Based Overrides**: After the statistical score is calculated, a set of deterministic rules is applied. For instance, any wallet that has been liquidated or has a very low repayment ratio receives a significant score penalty. This "expert-in-the-loop" step ensures that critical, unambiguous risk factors are never missed by the model.

## Model Validation and Performance

The model's accuracy and logical consistency were validated using the `tests/model_performance_analysis.ipynb` notebook. The analysis confirms that the model performs as expected and successfully separates user profiles based on risk.

**Key Performance Findings:**

*   **High-Scoring Wallets (Score 85-95):** The model correctly identifies top-tier users. These wallets consistently exhibit **zero liquidations** and a **perfect repayment ratio (1.0)**, demonstrating their reliability.
*   **Low-Scoring Wallets (Score 10-25):** The model accurately pinpoints high-risk users. This group is characterized by one of two failure modes:
    1.  A history of being **liquidated**.
    2.  A **near-zero repayment ratio**, indicating a loan default.
*   **Effective Risk Differentiation:** The model demonstrates a strong ability to differentiate between distinct types of risk, proving its logic is robust and aligned with real-world credit assessment principles. The score distribution shows clear separation between low-risk, average, and high-risk user groups.

## Project Structure
zero-credit-score/
├── data/
│ └── *.json # Input transaction data files (took the three largest chunks)
├── output/
│ └── top_1000_wallets.csv # output
├── src/
│ └── zeru_credit_score/
│ ├── loader.py # Data loading and cleaning
│ ├── features.py # Feature engineering logic
│ ├── scoring.py # Scoring model and overrides
│ ├── main.py # Main CLI entrypoint for scoring
│ └── run_sensitivity.py# CLI for sensitivity analysis
├── tests/
│ └── model_performance_analysis.ipynb # Notebook for model validation
├── README.md # This file
└── requirements.txt # Project dependencies


## Getting Started

### Prerequisites
*   Python 3.8+
*   `git`
*   A virtual environment manager (`venv` or `conda`)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/zero-credit-score.git
    cd zero-credit-score
    ```

2.  **Create and activate a virtual environment:**
    *Using `conda` (recommended):*
    ```sh
    conda create -n zeru_env python=3.10
    conda activate zeru_env
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 1. Generate Credit Scores
To run the main pipeline and generate a CSV file with the top-scoring wallets, use `main.py`.

```sh
python src/zeru_credit_score/main.py --data-dir data/ --output-dir output/ --topk 1000
```

This will process the data in `data/` and save `top_1000_wallets.csv` to the `output/` directory.

### 2. Run Sensitivity Analysis
To check how stable the leaderboard is to changes in model weights, run the `run_sensitivity.py` script.
```sh
python src/zeru_credit_score/run_sensitivity.py
```

This will output the Jaccard similarity between the base model and two alternatives, indicating model robustness.

To visually inspect and validate the model's performance, use the provided notebook under tests as test.ipynb

## License
This project is licensed under the MIT License. See the LICENSE file for details.
