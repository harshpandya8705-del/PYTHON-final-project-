# Titanic Survival Analysis

## 1. Project Overview

The Titanic Survival Analysis project is a Python-based data analysis project. The main objective is to analyze the Titanic passenger dataset and understand the factors that influenced passenger survival.

This project uses Exploratory Data Analysis (EDA) techniques to examine passenger information and visualize survival patterns based on passenger class, gender, age, and ticket fare.

**Project Name:** Titanic Survival Analysis  
**Programming Language:** Python  
**Tools:** VS Code / Jupyter Notebook  
**Libraries:** Pandas, Matplotlib, Seaborn

## 2. Project Objectives

- Analyze the Titanic passenger dataset.
- Understand passenger survival rates.
- Compare survival rates by gender.
- Analyze survival rates across passenger classes.
- Examine passenger age distribution.
- Identify missing values and clean the dataset.
- Create meaningful charts and visualizations.

## 3. Dataset Information

The dataset contains information about passengers aboard the Titanic.

**Dataset Source:** Kaggle Titanic Dataset

https://www.kaggle.com/competitions/titanic/data

Important dataset columns:

- PassengerId: Unique passenger identification number.
- Survived: Survival status (0 = No, 1 = Yes).
- Pclass: Passenger class (1st, 2nd, 3rd).
- Sex: Passenger gender.
- Age: Passenger age.
- SibSp: Number of siblings or spouses aboard.
- Parch: Number of parents or children aboard.
- Fare: Passenger ticket fare.
- Embarked: Port of embarkation.

Note: A small illustrative dataset is included for testing. Replace it with the official Kaggle dataset for the final analysis.

## 4. Technologies Used

**Python:** Main programming language.

**Pandas:** Data manipulation and analysis.

**Matplotlib:** Data visualization and charts.

**Seaborn:** Statistical visualization.

**NumPy:** Numerical operations.

## 5. Project Structure

```text
Titanic-Survival-Analysis/
│
├── data/
│   └── train.csv
│
├── outputs/
│   ├── key_metrics.csv
│   ├── survival_by_class.csv
│   ├── survival_by_gender.csv
│   ├── survival_by_class.png
│   ├── survival_by_gender.png
│   ├── age_distribution.png
│   ├── fare_by_class.png
│   └── correlation_heatmap.png
│
├── src/
│   └── analyze_titanic.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Installation and Setup

Step 1: Clone the GitHub repository.

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

Step 2: Open the project folder in VS Code.

Step 3: Install the required libraries.

```bash
pip install -r requirements.txt
```

Step 4: Download the Titanic dataset from Kaggle and place `train.csv` inside the data folder.

Step 5: Run the Python program.

```bash
python src/analyze_titanic.py
```

## 7. Data Analysis Performed

### 1. Data Exploration
- Display the first five records.
- Check dataset dimensions.
- Identify column data types.
- Find missing values.

### 2. Data Cleaning
- Handle missing age values using the median.
- Handle missing embarked values using the mode.
- Clean gender values.
- Create a survival status column.

### 3. Survival Analysis
- Calculate total passengers.
- Calculate total survivors.
- Calculate non-survivors.
- Calculate overall survival percentage.

### 4. Data Visualization
- Survival rate by passenger class.
- Survival rate by gender.
- Passenger age distribution.
- Fare distribution by class.
- Correlation heatmap.

## 8. Expected Project Outputs

After executing the Python program, the following outputs are generated:

- Key passenger statistics.
- Overall survival percentage.
- Class-wise survival analysis.
- Gender-wise survival analysis.
- Five data visualization charts.
- CSV summary reports.

All generated files are stored in the outputs folder.

## 9. Conclusion

The Titanic Survival Analysis project demonstrates how Python can be used to analyze real-world datasets.

Through data cleaning, exploratory data analysis, statistical calculations, and visualization, this project helps identify patterns in passenger survival.

It also provides practical experience with Pandas, Matplotlib, and Seaborn, which are important tools for a beginner data analyst.

## 10. Future Improvements

- Build an interactive dashboard using Streamlit.
- Add more detailed passenger analysis.
- Apply Machine Learning models to predict survival.
- Improve visualizations with interactive charts.

## 11. Author

**Name:** Your Name  
**Course:** Your Course Name  
**Project:** Titanic Survival Analysis  
**Language:** Python
