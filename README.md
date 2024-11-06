# Classification and Evaluation with Naive Bayes and XGBoost in Python

This repository includes a Python script to classify a dataset using both Naive Bayes and XGBoost classifiers. The script evaluates each model’s performance using accuracy, F1 score, confusion matrix, and ROC curve.

## Files

- **classification_analysis.py**: The main Python script that loads the dataset, trains classifiers, evaluates performance, and generates visualizations.

## Code Overview

1. **Data Loading and Preprocessing**:
   - The script reads data from `modified_dataset.csv`, assuming that 'Sale' is the target variable.
   - The target variable is isolated from the features, and the data is split into training and testing sets.

2. **Naive Bayes Classifier**:
   - Trains a Gaussian Naive Bayes classifier on the training set.
   - Evaluates the model on the test set using accuracy and F1 score.
   - Generates a confusion matrix and plots an ROC curve to evaluate model performance.

3. **XGBoost Classifier**:
   - Trains an XGBoost classifier on the same dataset.
   - Evaluates the model with accuracy, F1 score, confusion matrix, and ROC curve.
   - Provides additional visualization for a comprehensive comparison.

4. **Performance Metrics**:
   - **Accuracy**: Indicates the proportion of correctly classified samples.
   - **F1 Score**: A balanced metric that accounts for both precision and recall.
   - **Confusion Matrix**: Shows the counts of true positive, false positive, true negative, and false negative classifications.
   - **ROC Curve**: Plots the true positive rate against the false positive rate, with area under the curve (AUC) indicating model performance.

## Usage

1. **Prepare the Data**:
   - Ensure `modified_dataset.csv` is formatted correctly with 'Sale' as the target column.
   - Make sure the dataset is in the same directory or update the path in the script.

2. **Run the Script**:
   - Execute `classification_analysis.py` in a Python environment with required libraries installed.
   - The script will output accuracy and F1 scores for each model.
   - Confusion matrices and ROC curves will be displayed for detailed analysis.

3. **Example Output**:
   - Accuracy and F1 Score for both Naive Bayes and XGBoost classifiers.
   - Visualizations for confusion matrices and ROC curves for each model.

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
- xgboost

## Notes

- The script is modular, so additional classifiers can be easily added for comparison.
- Parameters for XGBoost and Naive Bayes can be adjusted to optimize performance.
- Ensure the dataset is balanced or consider stratified sampling to avoid bias in classification.

This code provides a comprehensive classification workflow, suitable for evaluating models on binary classification tasks.
