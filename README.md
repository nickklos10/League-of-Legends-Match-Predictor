# League of Legends Match Predictor

**Predicting League of Legends Match Outcomes Using Logistic Regression and PyTorch.**

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

Welcome to **League-of-Legends-Match-Predictor**, a machine learning project that leverages logistic regression and PyTorch to predict the outcomes of League of Legends matches. By analyzing various in-game statistics, this model aims to provide insights into what factors most significantly influence match results, offering a valuable tool for players and analysts alike.

## Objectives

- **Data Loading and Preprocessing:** Efficiently load and prepare the League of Legends dataset for model training.
- **Model Implementation:** Develop a logistic regression model using PyTorch to predict match outcomes.
- **Model Training:** Train the model using optimized techniques to ensure accurate predictions.
- **Model Optimization:** Apply L2 regularization to prevent overfitting and enhance model generalization.
- **Visualization:** Utilize confusion matrices and ROC curves to interpret model performance.
- **Model Persistence:** Save and reload the trained model for future use or deployment.
- **Hyperparameter Tuning:** Identify the optimal learning rate to maximize model accuracy.
- **Feature Importance:** Analyze and visualize the impact of each feature on the model's predictions.

## Technologies Used

- **Python 3.8+**
- **PyTorch**
- **pandas**
- **NumPy**
- **scikit-learn**
- **matplotlib**
- **seaborn**
- **Jupyter Notebook**

## Dataset

The project utilizes the `league_of_legends_data_large.csv` dataset, which contains comprehensive in-game statistics from numerous LoL matches. The features include:

- `win`: Target variable indicating match outcome (1 for win, 0 for loss)
- `kills`
- `deaths`
- `assists`
- `gold_earned`
- `cs` (creep score)
- `wards_placed`
- `wards_killed`
- `damage_dealt`

Ensure that the dataset is placed in the root directory of the project or update the data path accordingly in the scripts.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nickklos10/League-of-Legends-Match-Predictor.git
   cd League-of-Legends-Match-Predictor
   ```
2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   
3. Install Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

4. Verify Installation:
   Launch Jupyter Notebook to ensure everything is set up correctly.
   ```bash
   jupyter notebook
   ```

## Usage

Open the LEoL.ipynb notebook in Jupyter Notebook and follow the steps sequentially:

1. `Data Loading and Preprocessing:` Load the dataset, handle missing values, split into training and testing sets, and standardize the features.
2. `Model Implementation:` Define the logistic regression model using PyTorch.
3. `Model Training:` Train the model and monitor loss.
4. `Model Optimization:` Apply L2 regularization and retrain the model.
5. `Visualization and Interpretation:` Generate confusion matrices, ROC curves, and classification reports.
6. `Model Saving and Loading:` Save the trained model (already in the repo saved as "logistic_regression_model.pth") and reload it to verify performance.
7. `Hyperparameter Tuning:` Experiment with different learning rates to find the optimal one.
8. `Feature Importance:` Analyze and visualize the importance of each feature in the model.

Ensure that the league_of_legends_data_large.csv file is in the data/ directory or adjust the file path in the notebook accordingly.


## Results

Throughout the project, various metrics and visualizations provide insights into the model's performance:

* `Training and Testing Accuracy:` Indicates how well the model performs on known and unseen data.
* `Confusion Matrices:` Show the distribution of correct and incorrect predictions across classes.
* `ROC Curves and AUC:` Assess the model's ability to distinguish between classes across different thresholds.
* `Classification Reports:` Provide detailed metrics including precision, recall, and F1-score.
* `Feature Importance Plot:` Highlights which in-game statistics most significantly impact match outcomes.


## Conclusion

The League-of-Legends-Match-Predictor project successfully builds and optimizes a logistic regression model to predict the outcomes of League of Legends matches. Through meticulous data preprocessing, model training, optimization, and comprehensive evaluation, the project demonstrates the application of machine learning techniques to real-world gaming data. The insights gained from feature importance analysis offer valuable understanding into the key factors that drive match results.
