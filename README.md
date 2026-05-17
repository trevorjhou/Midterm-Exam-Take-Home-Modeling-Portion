# Model Workflow

## 1. Data Preparation

No missing values were found in the dataset, so no imputation was required.

Categorical variables were encoded using one-hot encoding, and train/test datasets were aligned after encoding to ensure consistent feature structures.

Because the target variable was imbalanced (approximately 11% positive class), Stratified 5-Fold Cross-Validation was used throughout the modeling process to preserve label distribution during validation.

---

## 2. Feature Engineering

Several additional features were created to capture customer interaction behavior and prior contact information more effectively:

- Interaction feature: `contact_time_minutes × contact_attempt_count`
- Ratio feature: `time_per_attempt`
- Binary indicator: `has_prior_contact`

Feature engineering improved the performance of several tree-based models, particularly Random Forest, XGBoost, and LightGBM, indicating that interaction-based behavioral features provided useful predictive information.

---

## 3. Feature Selection

Feature selection was performed after feature engineering using XGBoost permutation importance.

The top 20 most important features were selected to reduce lower-importance variables and simplify the feature space.

Different models responded differently to feature selection. XGBoost and CatBoost showed slight performance improvements after feature selection, while LightGBM performed better using the full feature-engineered dataset.

---

## 4. Baseline Models

Multiple models were evaluated to capture different patterns in the data:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Tree-based boosting models consistently outperformed the linear baseline model, suggesting the presence of non-linear relationships and feature interactions within the dataset.

---

## 5. Hyperparameter Tuning

Hyperparameter tuning was performed on the strongest boosting models:

- XGBoost
- LightGBM
- CatBoost

GridSearchCV with Stratified 5-Fold Cross-Validation was used to optimize model performance using F1 score as the primary evaluation metric.

Key parameters tuned included:

- Number of estimators
- Learning rate
- Tree depth / number of leaves
- Subsampling ratios

Different feature sets were used for different models based on earlier validation performance. LightGBM performed best using the full feature-engineered dataset, while XGBoost and CatBoost slightly benefited from the reduced feature-selected dataset.

---

## 6. Ensemble Learning

Probability blending ensemble methods were applied using the tuned boosting models:

- LightGBM
- XGBoost
- CatBoost

Two ensemble approaches were evaluated:

- Soft probability blending
- Weighted probability blending

Weighted probability blending achieved the strongest overall performance by assigning larger weights to stronger-performing models, particularly LightGBM.

The ensemble approach demonstrated that combining multiple boosting models could capture complementary predictive patterns more effectively than any single model alone.

---

# Final Model

The final selected model was the weighted probability blending ensemble combining tuned LightGBM, XGBoost, and CatBoost models.

This model achieved the highest overall validation performance with a Mean CV F1 score of approximately 0.656 using Stratified 5-Fold Cross-Validation.

The final ensemble leveraged the complementary strengths of multiple tuned boosting algorithms while assigning larger ensemble weights to stronger-performing models.
