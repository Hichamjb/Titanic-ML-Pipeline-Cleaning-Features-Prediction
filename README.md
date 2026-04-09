

# 🚢 Titanic Survival Prediction: End-to-End Machine Learning Project

## 📌 Overview

This project is a complete machine learning pipeline for predicting the survival of passengers aboard the RMS Titanic, based on the famous Kaggle dataset. It covers the entire data science workflow: exploratory data analysis (EDA), data cleaning, feature engineering, model training, and evaluation. The final model is an optimized **XGBoost classifier**, achieving strong predictive performance.

The notebook demonstrates how raw, messy data can be transformed into a clean, structured format and used to train a high-performance model.

---

## 🎯 Objective

The primary goal is to build a binary classification model that predicts whether a passenger survived (`1`) or did not survive (`0`) the Titanic disaster. The model is trained on a subset of the data and evaluated on a holdout test set.

---

## 📁 Files

| File Name | Description |
|-----------|-------------|
| `cleaning.ipynb` | Jupyter notebook containing the full data cleaning, preprocessing, feature engineering, and imputation pipeline. |
| `models.ipynb`   | Jupyter notebook where the XGBoost classifier is trained, evaluated, and visualized. |
| `data_clean.csv` | The final cleaned dataset, ready for modeling. |

---

## 🧹 Data Cleaning & Preprocessing (`cleaning.ipynb`)

This notebook handles the transformation of the raw Titanic dataset into a clean, model-ready format.

### Key Steps:

1. **Loading & Initial Inspection**
   - Loaded the dataset using `pandas`.
   - Displayed basic statistics and data types.

2. **Text Cleaning (Name Column)**
   - Converted names to lowercase.
   - Removed punctuation and special characters.
   - Replaced titles like `mr.` with `x` to reduce cardinality and avoid confusion with gender.

3. **Cabin Feature Engineering**
   - Extracted only the first letter of each cabin (deck identifier).
   - Used `XGBRegressor` to predict missing cabin values based on other features (e.g., `Pclass`, `Fare`, `Embarked`).

4. **Age Imputation**
   - Used `RandomForestRegressor` and `XGBRegressor` to predict missing ages.
   - Selected the best-performing regressor (XGBoost) to fill missing `Age` values.

5. **Ticket Feature Engineering**
   - Extracted prefix (e.g., `PC`, `STON/O2.`) and ticket number.
   - Label-encoded both for modeling.

6. **Categorical Encoding**
   - Used `LabelEncoder` to convert `Sex`, `Embarked`, and `prefix` into numeric values.

7. **Handling Missing Values**
   - Dropped rows with missing `Embarked` values (only a few).
   - Imputed `Age` and `Cabin` using regression models.

8. **Final Clean Dataset**
   - All columns were numeric and free of missing values.
   - Saved as `data_clean.csv` for modeling.

---

## 📊 Exploratory Data Analysis (EDA) (`models.ipynb`)

Before training, the cleaned dataset was explored to understand patterns and relationships.

### Visualizations:

- **Histograms**: Distribution of all numerical features (`Age`, `Fare`, `SibSp`, `Parch`, etc.).
- **Boxplots**: Identification of outliers in numerical columns.
- **Correlation Heatmap**: Showed relationships between features and the target variable `Survived`.

### Key Insights:

- `Sex` and `Pclass` are strong predictors of survival.
- `Fare` and `Age` also show moderate correlation with survival.
- Cabin deck (first letter) is informative, especially for first-class passengers.

---

## 🧠 Model Training (`models.ipynb`)

### Algorithm:
- **XGBoost Classifier** – chosen for its speed, accuracy, and ability to handle mixed data types.

### Hyperparameters (tuned for performance):
```python
XGBClassifier(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=6,
    min_child_weight=2,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=42
)
```

### Training Setup:
- Features used: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Cabin`, `Embarked`, `prefix`, `ticket_nember`
- Target: `Survived`
- Train-test split: 80/20, stratified by target

---

## 📈 Model Evaluation

The model was evaluated on the test set using several classification metrics:

| Metric      | Score    |
|-------------|----------|
| Accuracy    | 0.8315   |
| Precision   | 0.8167   |
| Recall      | 0.7206   |
| F1-score    | 0.7656   |
| ROC AUC     | 0.8618   |

### Confusion Matrix:

```
[[97 15]
 [19 49]]
```
- True Negatives: 97  
- False Positives: 15  
- False Negatives: 19  
- True Positives: 49

### Visualizations:
- Bar chart of metrics
- Confusion matrix heatmap

---

## 🧪 Regressor Performance (Age & Cabin Imputation)

| Task      | Model Used        | Performance / Notes                          |
|-----------|-------------------|----------------------------------------------|
| Age       | XGBRegressor      | Used to predict missing ages effectively     |
| Cabin     | XGBRegressor      | Predicted cabin deck letters encoded as ints |

Both regressors were trained on complete cases and used to impute missing values in the final dataset.

---

## 📦 Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🚀 How to Run

1. Open `cleaning.ipynb` and run all cells to clean the data and generate `data_clean.csv`.
2. Open `models.ipynb` and run all cells to train the model and view results.

---

## 📁 Outputs

- `data_clean.csv` – Cleaned dataset ready for modeling.
- Trained XGBoost model (in notebook memory).
- Performance metrics and visualizations.

---

## 📚 Conclusion

This project demonstrates a complete machine learning pipeline:

- Raw data → cleaning → feature engineering → imputation → modeling → evaluation
- Achieved **83% accuracy** and **0.86 ROC AUC** using XGBoost
- Robust handling of missing values via regression imputation
- Clear, reproducible workflow suitable for real-world ML tasks

The final model is reliable and can be used for prediction on unseen Titanic passenger data.


