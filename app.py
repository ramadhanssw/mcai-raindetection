from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import pickle

with open("model/predict.pkl", "rb") as file:
    model = pickle.load(file)
with open("model/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        date = request.form["date"]
        year = int(pd.to_datetime(date, format="%Y-%m-%dT").year)
        day = int(pd.to_datetime(date, format="%Y-%m-%dT").day)
        month = int(pd.to_datetime(date, format="%Y-%m-%dT").month)
        location = int(request.form["loc"])
        windgustdir = int(request.form["windgustdir"])
        winddir9am = int(request.form["winddir9am"])
        winddir3pm = int(request.form["winddir3pm"])
        mintemp = float(request.form["mintemp"])
        maxtemp = float(request.form["maxtemp"])
        rainfall = float(request.form["rainfall"])
        evaporation = float(request.form["evaporation"])
        sunshine = float(request.form["sunshine"])
        windgustspeed = float(request.form["windgustspeed"])
        windspeed9am = float(request.form["windspeed9am"])
        windspeed3pm = float(request.form["windspeed3pm"])
        humidity9am = float(request.form["humidity9am"])
        humidity3pm = float(request.form["humidity3pm"])
        pressure9am = float(request.form["pressure9am"])
        pressure3pm = float(request.form["pressure3pm"])
        temp9am = float(request.form["temp9am"])
        team3pm = float(request.form["team3pm"])
        cloud9am = float(request.form["cloud9am"])
        cloud3pm = float(request.form["cloud3pm"])
        raintoday = int(request.form["raintoday"])
        predict_list = [
            [
                location,
                mintemp,
                maxtemp,
                rainfall,
                evaporation,
                sunshine,
                windgustdir,
                windgustspeed,
                winddir9am,
                winddir3pm,
                windspeed9am,
                windspeed3pm,
                humidity9am,
                humidity3pm,
                pressure9am,
                pressure3pm,
                cloud9am,
                cloud3pm,
                temp9am,
                team3pm,
                raintoday,
                year,
                month,
                day,
            ]
        ]
        predict_list = scaler.transform(predict_list)
        output = model.predict(predict_list)
        if output == 0:
            return render_template("brightdetected.html")
        else:
            return render_template("rainingdetected.html")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
