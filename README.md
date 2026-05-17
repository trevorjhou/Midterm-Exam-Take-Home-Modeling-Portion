# Model Workflow

## 1. Exploratory Data Analysis

The dataset was first explored to understand its structure and modeling challenges.

The EDA included:

- checking dataset shape and data types
- checking missing values
- checking duplicate rows
- comparing train and test distributions
- exploring categorical variable distributions
- exploring numerical variable distributions
- analyzing target class imbalance
- examining feature distributions by target class
- reviewing numerical correlations using a heatmap

The EDA showed that the dataset had no missing values or duplicate rows, but contained several categorical variables requiring encoding. The target variable was imbalanced, with only about 11% of observations belonging to the positive class. Several numerical variables, such as `contact_time_minutes` and `contact_attempt_count`, were highly right-skewed, while `days_since_prior_contact` contained a special value around 999. These observations guided later preprocessing, feature engineering, and model evaluation choices.

Train and test distributions were generally consistent across both categorical and numerical variables, suggesting that the datasets followed similar underlying data distributions.

---

## 2. Data Preparation

No imputation was required because no missing values were found.

Categorical variables were encoded using one-hot encoding, and train/test datasets were aligned after encoding to ensure consistent feature structures.

Because the target variable was imbalanced, Stratified 5-Fold Cross-Validation was used throughout the modeling process to preserve label distribution during validation. F1 score was used as the primary evaluation metric.

---

## 3. Feature Engineering

Several additional features were created based on EDA findings:

- Interaction feature: `contact_time_minutes × contact_attempt_count`
- Ratio feature: `time_per_attempt`
- Binary indicator: `has_prior_contact`

These features were designed to capture customer engagement intensity and prior contact information more effectively.

Feature engineering improved the performance of several tree-based models, particularly Random Forest, XGBoost, and LightGBM, indicating that interaction-based behavioral features contained useful predictive information.

---

## 4. Feature Selection

Feature selection was performed after feature engineering using XGBoost permutation importance.

The top 20 features were selected to reduce lower-importance variables and simplify the feature space.

Different models responded differently to feature selection. XGBoost and CatBoost showed slight performance improvements after feature selection, while LightGBM performed better using the full feature-engineered dataset.

---

## 5. Modeling and Hyperparameter Tuning

Five baseline models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Tree-based boosting models consistently achieved the strongest performance, indicating the presence of non-linear relationships and interaction effects within the dataset.

Hyperparameter tuning was therefore focused on the strongest boosting models:

- LightGBM
- XGBoost
- CatBoost

GridSearchCV with Stratified 5-Fold Cross-Validation was used to optimize model performance using F1 score as the primary evaluation metric.

Different feature sets were used for different models based on earlier validation performance. LightGBM achieved better results using the full feature-engineered dataset, while XGBoost and CatBoost slightly benefited from the reduced feature-selected dataset.

Key parameters tuned included:

- number of estimators
- learning rate
- tree depth / number of leaves
- subsampling ratios

---

## 6. Ensemble Learning

Soft probability blending and weighted probability blending were evaluated using the tuned LightGBM, XGBoost, and CatBoost models.

LightGBM used the full feature-engineered dataset, while XGBoost and CatBoost used the selected top features.

Weighted probability blending outperformed equal-weight soft blending because larger ensemble weights were assigned to stronger-performing models, particularly LightGBM.

The ensemble approach demonstrated that combining multiple boosting models could capture complementary predictive patterns more effectively than any single model alone.

---

# Final Model

The final selected model was the weighted probability blending ensemble combining tuned LightGBM, XGBoost, and CatBoost.

The final model achieved the highest validation performance with a Mean CV F1 score of approximately 0.656 using Stratified 5-Fold Cross-Validation.
