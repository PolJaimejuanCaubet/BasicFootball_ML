# ⚽ Football Match Predictor

## 📌 Project Overview

This repository contains a Machine Learning model for predicting football match outcomes using a **Decision Tree Regressor**. The model processes historical match data, encodes categorical features, and optimizes the decision tree depth to achieve better predictions.

## 📂 Dataset

The dataset used for training is located at:

>LeaguesDataset/LaLiga.csv

It includes match statistics such as:

- Home Team & Away Team
- Goals scored by each team
- Match date (converted into numerical features)
- Final match result (encoded as a categorical variable)

## 🛠️ Technologies Used

- **Python** 🐍
- **Pandas** (Data manipulation)
- **Scikit-learn** (Machine Learning algorithms)

## 🚀 Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/PolJaimejuanCaubet/BasicFootball_ML.git
   cd BasicFootball_ML

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   
3. Run the notebook:

   Open and execute the steps inside **Football_Predictor.ipynb**.

## 🔍 Model Training & Prediction

The model follows these steps:

**1.** Load and clean the dataset.

**2.** Convert categorical values to numerical ones using LabelEncoder.

**3.** Split the data into training and validation sets.

**4.** Train a Decision Tree Regressor and evaluate it using Mean Absolute Error (MAE).

**5.** Optimize the model by tuning the number of leaf nodes.

**6.** Convert numerical predictions back to categorical results.

## 📊 Example Prediction Output

>Match: Real Madrid vs Barcelona - Predicted Result: Draw

>Match: Sevilla vs Atletico Madrid - Predicted Result: Home Win

## 📌 Future Improvements

- **Implement** Random Forest for better accuracy.

- **Add** more detailed match statistics.

- **Extend** predictions to multiple leagues.

## 🤝 Contributing

Feel free to fork this repository and **submit pull requests** to improve the model! 🚀
