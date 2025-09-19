# SVM Classification on Loan Application Approval Dataset Using PCA

## Overview

This project implements a Support Vector Machine (SVM) classifier to predict whether loan would be approved for a particular applicant or not. The dataset undergoes preprocessing, dimensionality reduction using Principal Component Analysis (PCA), model training, evaluation, and visualization of decision boundaries.

## Project Structure

- **Data Loading**: The dataset is downsampled for computational efficiency.
- **Preprocessing**:
  - Categorical variables are encoded using Label Encoding.
- **Feature and Target Separation**: Features (`X`) and target variable (`y`) are separated.
- **Train-Test Split**: Data is split into training and testing sets.
- **Feature Scaling**: StandardScaler is applied to normalize features.
- **Dimensionality Reduction**: PCA reduces the feature space to 5 principal components.
- **Model Training**: An SVM classifier with a linear kernel is trained on the PCA-transformed data.
- **Model Evaluation**:
  - Accuracy score
  - Confusion matrix visualization
  - Classification report (precision, recall, f1-score)
- **Decision Boundary Visualization**: Decision boundaries of the SVM classifier are plotted on pairs of the first four PCA components for the test set.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## How to Run

1. Make sure you have the required libraries installed. You can install them using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Download the dataset `loan_data.csv` and place it in the working directory.

3. Run the notebook or script to perform the classification and visualize the results.

## Key Insights

- PCA effectively reduces the dimensionality of the dataset while preserving the variance.
- The SVM classifier achieves a reasonable accuracy in predicting status of loan application.
- Visualization of decision boundaries helps understand model behavior in the transformed PCA space.

## Notes

- The dataset is downsampled by taking every 5th record to reduce computational cost.
- Only the first 5 principal components from PCA are used for training and visualization.
- Decision boundaries are shown for pairs of PCA components, revealing how the classifier separates the classes in reduced dimensions.

## Author

Divyam Goel (2201CS23)