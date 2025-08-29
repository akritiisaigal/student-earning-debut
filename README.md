# student-earning-debut

Student Earning Debut 💸🎓

A Machine Learning project predicting post-graduation earnings and debt-to-earnings ratios by field of study and institution.

📌 Project Overview

This project applies Machine Learning to analyze the relationship between students’ post-graduation earnings, student debt, and field of study across various institutions.

The goal is to:

Predict earnings after graduation based on program and institution-level data.

Model the debt-to-earnings ratio (DER) as an indicator of loan burden.

Highlight key factors shaping repayment outcomes, such as program type, institution, and student demographics.

By uncovering insights into student loan affordability, this project contributes to data-driven discussions on higher education costs, equity, and long-term financial well-being.

⚙️ Features

📊 Data Preprocessing: Cleaning and handling missing data from institutional/student datasets.

🔎 Exploratory Data Analysis (EDA): Understanding distributions, correlations, and outliers.

🤖 Machine Learning Models:

Regression models (Linear, Ridge, Lasso) for continuous earnings predictions.

Classification models for debt-to-earnings risk categories.

📈 Model Evaluation: MAE, RMSE, R² for regression; Accuracy, Precision, Recall for classification.

🏫 Program & Institution Analysis: Identifies high-burden programs and institutions with strong/weak repayment outcomes

📊 Dataset

The dataset is derived from:

College Scorecard (U.S. Department of Education) – providing institutional data on student debt and earnings.

Supplementary program-level datasets (if applicable).

⚠️ Ensure compliance with dataset licensing when using this project.

🧠 Models Used

Linear Regression, Ridge, Lasso – for continuous earnings predictions.

Random Forest, XGBoost – for more robust predictions.

Logistic Regression, Decision Trees – for classification of debt-to-earnings outcomes.

📈 Results

Predicted earnings with X% accuracy (R² score ~0.78).

Identified fields of study with the highest and lowest debt-to-earnings ratios.

Highlighted institutional factors most correlated with repayment success.

(Detailed results are in the results/ folder.)

🛠️ Tech Stack

Python 3.x

Pandas, NumPy – Data processing

Scikit-learn, XGBoost – Modeling

Matplotlib, Seaborn – Visualization

Jupyter Notebook – Analysis

📌 Future Work

Incorporate student demographic data for deeper fairness analysis.

Deploy as an interactive dashboard (Streamlit/Flask).

Extend to international datasets for broader comparisons.
