from flask import Flask, request, render_template
from flask_cors import cross_origin
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Debugging information
print("Current directory:", os.getcwd())
print("File exists:", os.path.exists('your_scaler.pkl'))

# Load scaler and model
scaler = joblib.load('your_scaler.pkl')  # Adjust path as necessary
model = xgb.Booster()
model.load_model('your_model.model')  # Adjust path as necessary

@app.route("/")
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == 'POST':
        Age = float(request.form['Age'])
        Gender = int(request.form['Gender'])
        Height = float(request.form['Height'])
        Weight = float(request.form['Weight'])
        Sys_BP = float(request.form['Sys_bp'])
        Dia_BP = float(request.form['Dia_bp'])
        Smoke = int(request.form['Smoke'])
        Alco = int(request.form['Alco'])
        Active = int(request.form['Active'])
        Chol_2 = int(request.form['Chol_lvl'] == "Level 2")
        Chol_3 = int(request.form['Chol_lvl'] == "Level 3")
        Gluc_2 = int(request.form['Gluc_lvl'] == "Level 2")
        Gluc_3 = int(request.form['Gluc_lvl'] == "Level 3")

        data = np.array([
            Age, Gender, Height, Weight, Sys_BP, Dia_BP, Smoke,
            Alco, Active, Chol_2, Chol_3, Gluc_2, Gluc_3
        ]).reshape(1, -1)

        # Scale the data
        data_scaled = scaler.transform(data)

        # Create DMatrix for XGBoost prediction
        data_DM = xgb.DMatrix(data_scaled)

        # Make prediction
        cardio_prob = model.predict(data_DM)[0]
        output = round(cardio_prob * 100, 2)

        return render_template('index.html', prediction_text=f"Your chances of having Cardiovascular Disease is {output}%.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)
