# Diabetes Hospital Readmission Prediction

This project uses logistic regression to predict whether a patient with diabetes will be readmitted to the hospital within 30 days.

The analysis uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

## Project Goal

Hospital readmissions can be costly and may indicate gaps in patient care. The goal of this project is to build and evaluate a binary classification model that predicts whether a patient will be readmitted within 30 days of discharge.

The target variable is defined as:

- `1`: Patient was readmitted within 30 days
- `0`: Patient was not readmitted within 30 days

## Dataset

The dataset is loaded using the `ucimlrepo` Python package.

Several columns are removed during preprocessing, including:

- `encounter_id`
- `patient_nbr`
- `weight`
- `payer_code`
- `medical_specialty`

Missing categorical values are represented by `"missing"`, and categorical features are converted using one-hot encoding.

## Methods

The project includes:

1. Data loading and preprocessing
2. Missing-value handling
3. Categorical feature encoding
4. Stratified train, validation, and test splits
5. Feature standardization
6. Logistic regression hyperparameter tuning
7. Model evaluation using:
   - Accuracy
   - ROC AUC
   - Precision
   - Recall
   - Confusion matrix

The logistic regression regularization parameter, `C`, is selected using validation-set ROC AUC.

## Repository Contents

```text
log_reg_diabetes/
├── log_reg_diabetes.ipynb
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/log_reg_diabetes.git
cd log_reg_diabetes
```

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn ucimlrepo jupyter
```

## Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
log_reg_diabetes.ipynb
```

The dataset is downloaded when the notebook runs, so an internet connection may be required.

## Results

The model is evaluated on a held-out test set using accuracy and ROC AUC.

Because the target classes are imbalanced, ROC AUC is emphasized over accuracy when assessing the model's ability to distinguish patients who are readmitted within 30 days.

Add your final results here after running the notebook:

```text
Best C:
Test accuracy:
Test ROC AUC:
```

## Limitations

- The model only captures linear relationships between the input features and the target.
- The dataset contains substantial class imbalance.
- Missing values are handled using a simplified categorical representation.
- The model is intended as an educational machine-learning project and should not be used for clinical decision-making.

## Future Improvements

Potential improvements include:

- Comparing logistic regression with random forests, K-nearest neighbors, and neural networks
- Addressing class imbalance through class weighting or resampling
- Performing feature selection
- Evaluating additional metrics such as the precision-recall curve
- Investigating feature importance and model interpretability

## Author

Created as a machine-learning classification project using Python and scikit-learn.