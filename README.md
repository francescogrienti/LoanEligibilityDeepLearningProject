# Loan Eligibility Deep Learning Project

This project implements a Deep Learning model to predict loan eligibility based on various applicant features. It compares the performance of a Neural Network with traditional machine learning models like Logistic Regression and Decision Trees.

The dataset used in this project is sourced from [Kaggle's Loan Eligible Dataset](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset).

## Project Structure

- `dataset/`: Contains the raw dataset (`loan-train.xls`).
- `data_preprocessing/`: Visualizations from data exploration (correlation matrix, histograms).
- `nn_architecture/`: Visual representation of the neural network architecture.
- `metrics/`: Plots showing accuracy and loss trends during training.
- `model/`: Python scripts for data processing and model training.
    - `main.py`: Main execution script including preprocessing, training, and evaluation.
    - `functions.py`: Helper functions for model building and plotting.

## Requirements

To run this project, you need the following libraries:

- Python 3.x
- pandas
- matplotlib
- seaborn
- tensorflow
- scikit-learn

## How to Run

1. Clone the repository.
2. Ensure you have the required dependencies installed.
3. Run the main script from the root directory:
   ```bash
   python model/main.py
   ```

## Model Architecture

The deep learning model is a Sequential Neural Network built using Keras:
- **Input Layer**: 8 features (Dependents, Self_Employed, ApplicantIncome, etc.)
- **Hidden Layer 1**: Dense layer with ReLU activation (tunable size).
- **Hidden Layer 2**: Dense layer with ReLU activation (tunable size).
- **Output Layer**: Dense layer with Sigmoid activation for binary classification.

The project also includes Hyperparameter Tuning using `RandomizedSearchCV`.

## Results

The project evaluates the model using:
- Accuracy and Loss plots over epochs.
- Comparison with Logistic Regression.
- Comparison with Decision Tree Classifier.

Current metrics and visualizations can be found in the `metrics/` and `data_preprocessing/` folders.

## Future Work

This project is a work in progress. Future improvements may include:
- Deeper evaluation of the model performance (Precision, Recall, F1-Score).
- Implementation of more advanced architectures.
- Enhanced feature engineering and data cleaning.
- Improved hyperparameter optimization strategy.
