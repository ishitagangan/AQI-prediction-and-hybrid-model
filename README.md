# AQI Prediction and Hybrid Model

A machine learning web application that predicts the Air Quality Index (AQI) using a hybrid ensemble approach combining Linear Regression, Random Forest, and Gradient Boosting models.

## 🚀 Features
* **Hybrid Modeling:** Combines multiple ML models for more accurate AQI forecasting.
* **Web Interface:** Built with Flask (`app.py`) to easily input weather/pollutant factors and get instant predictions.

## 📁 Project Structure
* `app.py` - The main Flask application.
* `Air_quality_prediction.ipynb` - Jupyter Notebook containing data analysis and model training.
* `*.pkl` - Trained model checkpoints (Linear, RF, GB, Scaler, and the Hybrid model).
* `templates/` & `static/` - HTML/CSS files for the web frontend.

## 🛠️ Installation & Setup
1. Clone this repository or download the files.
2. Install the required dependencies:
   ```bash
   pip install flask scikit-learn numpy pandas
