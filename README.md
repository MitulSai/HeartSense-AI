# ❤️ Heart Disease Prediction System

An AI-powered Heart Disease Prediction web application built using **Python, Streamlit, Scikit-Learn, Pandas, and Machine Learning**. The application analyzes patient health parameters and predicts the likelihood of heart disease using a trained **K-Nearest Neighbors (KNN)** model.

## 🚀 Features

* Interactive and modern Streamlit user interface
* Real-time heart disease risk prediction
* KNN Machine Learning model for classification
* Data preprocessing and feature scaling
* Patient health summary dashboard
* Instant prediction results with risk indicators
* Responsive design with custom styling

## 📊 Input Parameters

The model uses the following health indicators:

* Age
* Gender
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol Level
* Fasting Blood Sugar
* Resting ECG Results
* Maximum Heart Rate
* Exercise-Induced Angina
* Oldpeak (ST Depression)
* ST Slope

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Machine Learning (KNN Classifier)

## 📁 Project Structure

├── app.py
├── knn_heart_model.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
├── requirements.txt
└── README.md

## ▶️ Run Locally

1. Clone the repository

```bash
git clone https://github.com/your-username/HeartSense-AI.git
cd HeartSense-AI
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start the application

```bash
streamlit run app.py
```

## 🎯 Future Improvements

* Probability-based risk score
* Model comparison (KNN, Random Forest, XGBoost)
* Data visualization dashboard
* PDF health report generation
* Cloud deployment support

## 👨‍💻 Developer

Developed by **Mitul Sai** as a Machine Learning and Healthcare Analytics project.
