# Heart Risk Assessment Model

This repository contains a machine learning project for predicting heart disease using patient data. The project involves data preprocessing, exploratory data analysis, model selection, training, evaluation, and deployment.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Heart disease is one of the leading causes of death worldwide. Early prediction of heart disease can help in taking preventive measures and saving lives. This project aims to build a machine learning model to predict the presence of heart disease based on various patient attributes.

## Dataset

The dataset used in this project contains the following attributes:

1. **age**: Age in years
2. **sex**: Sex (1 = male; 0 = female)
3. **chest pain type**: Chest pain type (1-4)
4. **resting bp s**: Resting blood pressure in mm Hg
5. **cholesterol**: Serum cholesterol in mg/dl
6. **fasting blood sugar**: Fasting blood sugar (1 if > 120 mg/dl, 0 otherwise)
7. **resting ecg**: Resting electrocardiogram results (0-2)
8. **max heart rate**: Maximum heart rate achieved
9. **exercise angina**: Exercise induced angina (1 = yes; 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **ST slope**: The slope of the peak exercise ST segment (0-2)
12. **target**: Presence of heart disease (1 = yes; 0 = no)

## Installation

To run this project, you will need Python and the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage
1. Clone the repository:

```bash
git clone https://github.com/your-username/HeartDiseasePrediction.git
cd HeartDiseasePrediction
```
2. Run the Jupyter Notebook or Python script to see the data processing, model training, and evaluation:

```bash
jupyter notebook heart_disease_prediction.ipynb
```

3. To preprocess data, train models, and make predictions, run:

```bash
python heart_disease_prediction.py
```

## Project Structure

```tree
HeartDiseasePrediction/
├── data/
│   └── heart_disease_data.csv
├── notebooks/
│   └── heart_disease_prediction.ipynb
├── scripts/
│   └── heart_disease_prediction.py
├── models/
│   └── trained_model.pkl
├── README.md
└── requirements.txt
```
data/: Contains the dataset.

notebooks/: Jupyter Notebook with detailed steps and explanations.

scripts/: Python scripts for data preprocessing, model training, and evaluation.

models/: Saved trained models.

README.md: Project documentation.

requirements.txt: List of required libraries.

## Modeling

The following steps are involved in the modeling process:

1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Selection
4. Model Training (Logistic Regression, Decision Tree, Random Forest, etc.)
5. Model Evaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

## Results

The final model achieved an accuracy of **`87.39%`** on the test set. Detailed evaluation metrics and plots are provided in the notebook and script.

## License
