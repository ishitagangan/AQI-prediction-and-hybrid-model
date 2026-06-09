import matplotlib
matplotlib.use('Agg')  # 🔥 non-GUI backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load models
lr_model = pickle.load(open("linear_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    city = data['city']

    # Load dataset
    df = pd.read_csv("static/city_day.csv")
    df.columns = df.columns.str.strip()

    pollutants = ['PM2.5','PM10','NO2','CO','SO2','O3']
    df[pollutants] = df[pollutants].fillna(df[pollutants].mean())

    city_data = df[df['City'] == city]
    avg = city_data.mean(numeric_only=True)

    features = np.array([[
        avg['PM2.5'], avg['PM10'], avg['NO2'],
        avg['CO'], avg['SO2'], avg['O3']
    ]])

    lr_pred = lr_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    final_pred = (lr_pred + rf_pred) / 2

    # 🔥 MATPLOTLIB GRAPH
    last5 = city_data['AQI'].dropna().tail(5)

    plt.figure()
    plt.plot(last5.values, marker='o')
    plt.title(f"AQI Trend - {city}")
    plt.xlabel("Last 5 Records")
    plt.ylabel("AQI")

    # Save graph image inside static folder
    graph_path = "static/aqi_graph.png"
    plt.savefig(graph_path)
    plt.close()

    return jsonify({
        'aqi': round(final_pred,2),
        'graph': "/static/aqi_graph.png"
    })


if __name__ == "__main__":
    app.run(debug=True)