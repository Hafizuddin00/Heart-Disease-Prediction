# Heart Disease Prediction Project

A machine learning project that predicts heart disease using various classification algorithms. This project performs comprehensive data preprocessing, feature selection, and model evaluation to identify the best-performing model for heart disease prediction.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Data Preprocessing](#data-preprocessing)
- [Results](#results)

## 🎯 Overview

This project implements a machine learning pipeline to predict heart disease based on patient clinical data. The project includes:

- Data cleaning and preprocessing
- Missing value imputation
- Feature scaling and normalization
- Outlier detection and handling
- Feature selection (categorical and numerical)
- Multiple model training and evaluation
- Interactive prediction interface

## ✨ Features

- **Data Preprocessing**: Handles missing values, outliers, and performs feature scaling
- **Feature Selection**: Uses Chi-square test for categorical features and ANOVA F-test for numerical features
- **Multiple Models**: Implements Logistic Regression, Random Forest, and K-Nearest Neighbor classifiers
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, F1-score, confusion matrices, and ROC curves
- **Interactive Prediction**: Command-line interface for making predictions on new patient data

## 📊 Dataset

The dataset (`latestHeart.csv`) contains **962 patient records** with the following features:

- **Age**: Patient's age
- **Sex**: Patient's sex (M/F)
- **ChestPainType**: Type of chest pain (ATA, NAP, ASY, TA)
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol (mg/dl)
- **FastingBS**: Fasting blood sugar (0/1)
- **RestingECG**: Resting electrocardiographic results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise
- **ST_Slope**: Slope of the peak exercise ST segment
- **HeartDisease**: Target variable (0 = No disease, 1 = Disease)

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Jupyter Notebook**: Interactive development environment

## 📦 Installation

1. Clone or download this repository

2. Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Or use the requirements file (if available):

```bash
pip install -r requirements.txt
```

3. Ensure the dataset file `latestHeart.csv` is in the project directory

## 🚀 Usage

### Running the Notebook

1. Start Jupyter Notebook:

```bash
jupyter notebook
```

2. Open `heartDisease.ipynb`

3. Run all cells sequentially to:
   - Load and explore the data
   - Preprocess the dataset
   - Train and evaluate models
   - Make predictions

### Making Predictions

The notebook includes an interactive prediction function. When you run the last cell, you'll be prompted to enter:

- Age
- Sex (0 = Female, 1 = Male)
- Chest Pain Type (0-3)
- Resting Blood Pressure
- Fasting Blood Sugar (0/1)
- Max Heart Rate
- Exercise Induced Angina (0/1)
- ST Depression (Oldpeak)
- ST Slope (0-2)

The model will then predict whether the patient has heart disease and provide a probability score.

## 📁 Project Structure

```
Project_SKIH2013/
│
├── heartDisease.ipynb      # Main Jupyter notebook with all code
├── latestHeart.csv          # Dataset file
└── README.md               # Project documentation
```

## 🎯 Model Performance

The project evaluates three machine learning models:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 86.61% | 86.82% | 86.61% | 86.60% |
| **Random Forest** | 85.71% | 86.17% | 85.71% | 85.69% |
| **K-Nearest Neighbor** | 81.25% | 81.55% | 81.25% | 81.24% |

**Best Model**: Logistic Regression achieves the highest accuracy of **86.61%**

## 🔧 Data Preprocessing

### 1. Missing Value Handling
- Numeric columns: Filled with median values
- Categorical columns: Filled with mode values

### 2. Feature Scaling
- **Standardization** (Z-score normalization): Applied to Age, RestingBP, Cholesterol, MaxHR
- **Min-Max Normalization**: Applied to Oldpeak

### 3. Outlier Handling
- Detected outliers using IQR (Interquartile Range) method
- Outliers were capped at upper and lower bounds

### 4. Feature Encoding
- Categorical features encoded using Label Encoding

### 5. Feature Selection
- **Categorical features**: Chi-square test
- **Numerical features**: ANOVA F-test
- Removed low-importance features: Cholesterol, RestingECG

## 📈 Results

The project includes visualizations for:
- Missing value heatmaps
- Boxplots for outlier detection
- Correlation matrices
- Feature importance scores
- Confusion matrices for each model
- ROC curves comparing all models
- Accuracy comparison bar chart

## 📝 Notes

- The dataset is split into 70% training and 30% testing
- Random state is set to 42 for reproducibility
- The Random Forest model is used for interactive predictions
- All preprocessing steps are applied consistently to both training and test sets

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is open source and available for educational purposes.

---

**Note**: This project is for educational and research purposes. For actual medical diagnosis, please consult healthcare professionals.

