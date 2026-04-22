# Apple M1 Purchase Prediction: End-to-End Machine Learning Pipeline

This repository contains a comprehensive 5-stage machine learning project aimed at predicting consumer purchase behavior for Apple M1-powered devices. The project progresses from a baseline statistical analysis to a highly optimized Gaussian Naive Bayes model, addressing challenges like class imbalance and high-dimensional feature selection.

## Project Overview
The goal is to identify whether a consumer will purchase an Apple M1 device based on demographics, brand trust, and technical feature preferences. 

### Project Structure
The analysis is divided into five distinct phases:
1. **Baseline Modeling:** Establishing initial benchmarks using Logistic Regression.
2. **Exploratory Data Analysis (EDA):** Deep dive into 133 observations to identify drivers like ecosystem loyalty and income.
3. **Class Imbalance Mitigation:** Evaluating resampling (SMOTE, ROS) vs. algorithmic weighting to handle skewed target classes.
4. **Feature Engineering & Selection:** Using Recursive Feature Elimination (RFE) and Factor Analysis to reduce dimensionality.
5. **Model Selection & Tuning:** Comparative analysis of multiple classifiers to find the optimal production-ready model.

## Key Findings
*   **Ecosystem Power:** Mac users are significantly more likely to own multiple Apple devices, which is the strongest predictor of M1 adoption.
*   **Feature Latency:** Technical specs (Neural engine, multitasking) act as a single "Power & Tech" factor in the consumer's mind.
*   **Performance:** The final optimized model achieved an **80% Accuracy** and an **0.84 F1-score** for purchasers.

## Tech Stack
*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, LazyPredict, Scipy
*   **Techniques:** Factor Analysis, RFE, SMOTE, Gaussian Naive Bayes, GridSearchCV

## Detailed Pipeline

### Part 1: Logistic Regression Baseline
Initial model to check linear separability. 
*   **Result:** AUC of 0.66. Showed a strong bias toward the majority class (Purchased).

### Part 2: Exploratory Data Analysis
Identified that `trust_apple`, `user_pcmac`, and `income_group` were critical predictors. Discovered that performance features are often evaluated by consumers as a combined "package" rather than individual specs.

### Part 3: Handling Class Imbalance
Tested multiple strategies to improve the Minority Class (Not Purchased) F1-score:
*   **Winner:** Logistic Regression with `class_weight='balanced'` (F1 improved from 0.59 to 0.70).

### Part 4: Feature Selection & Factor Analysis
Used a "Voting System" combining Variance Threshold, Chi-Square, and Tree-based importance. 
*   **Factor Analysis:** Reduced features into 3 latent factors: *Power & Tech*, *Lifestyle & Portability*, and *Optimization & Value*.

### Part 5: Final Model Selection
Used `LazyPredict` for rapid prototyping.
*   **Final Model:** Gaussian Naive Bayes (GaussianNB).
*   **Optimization:** Hyperparameter tuning + Balanced Sample Weights.
*   **Final Metrics:** 0.88 Recall for 'Not Purchased' (Minority) and 0.93 Precision for 'Purchased'.

## 📈 Performance Summary
| Metric | Not Purchased (Minority) | Purchased (Majority) |
| :--- | :--- | :--- |
| Precision | 0.64 | 0.93 |
| Recall | 0.88 | 0.76 |
| F1-Score | 0.74 | 0.84 |

## ⚙️ Usage
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run 'python main.py'.