# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:06:09 2023

@author: kmerl
"""

import numpy as np
from flask import Flask, request, render_template
import joblib

import pandas as pd

app = Flask(__name__)

# Define the path for loading the pickle files
load_path = '/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/Further Changes/'

# Load the model and pipeline
model = joblib.load(load_path + "G1_model.pkl")
pipeline = joblib.load(load_path + "G1_pipeline.pkl")

# Load the column names from the pickle file
with open(load_path + 'G1_column.pkl', 'rb') as f:
    cols = joblib.load(f)

print("Model, pipeline, and column names loaded successfully.")
print("Column names:")
print(cols)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    
    AG_DRIV = np.array([request.form['AG']])
    ALCOHOL = np.array([request.form['Alc']])
    AUTOMOBILE = np.array([request.form['Auto']])
    CYCLIST = np.array([request.form['Cycl']])
    DAY = np.array([request.form['Day']])
    DAYOFWEEK = np.array([request.form['DOW']])
    DISABILITY = np.array([request.form['Dis']])
    DISTRICT = np.array([request.form['DISTRICT']])
    EMERG_VEH = np.array([request.form['EMERG_VEH']])
    HOUR = np.array([request.form['Hr']])
    LATITUDE = np.array([request.form['lati']])
    LIGHT = np.array([request.form['Light']])
    LONGITUDE = np.array([request.form['longi']])
    MINUTE = np.array([request.form['Min']])
    MONTH = np.array([request.form['Mon']])
    MOTORCYCLE = np.array([request.form['Mot']])
    PASSENGER = np.array([request.form['Pass']])
    PEDESTRIAN = np.array([request.form['Ped']])
    RDSFCOND = np.array([request.form['RDSF']])
    REDLIGHT = np.array([request.form['Red']])
    SPEEDING = np.array([request.form['Speed']])
    TRSN_CITY_VEH = np.array([request.form['TRSN']])
    TRUCK = np.array([request.form['Tr']])
    VISIBILITY = np.array([request.form['Vis']])
    YEAR = np.array([request.form['Year']])


    final = np.concatenate([AG_DRIV,ALCOHOL,AUTOMOBILE,CYCLIST,DAY,
                           DAYOFWEEK,DISABILITY,DISTRICT,EMERG_VEH,HOUR,LATITUDE,
                           LIGHT,LONGITUDE,MINUTE,MONTH,MOTORCYCLE,PASSENGER,PEDESTRIAN,RDSFCOND,
                           REDLIGHT,SPEEDING,TRSN_CITY_VEH,TRUCK,VISIBILITY,YEAR])
    
    final = np.array(final)
    data = pd.DataFrame([final], columns=cols)
    data_trans = pipeline.transform(data)
    prediction = model.predict(data_trans)
    return render_template("result.html", prediction = round(prediction[0],3))

if __name__ == "__main__":
    app.run(debug=True, port=5000)









