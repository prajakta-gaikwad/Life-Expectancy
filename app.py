import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, url_for, redirect, Markup, flash
from sklearn.externals import joblib
import pickle
import requests

from flask_sqlalchemy import SQLAlchemy
import xgboost as xgb

with open(f'model/thoracic_model_xgboost.pkl','rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
app.secret_key = '12345'

# Database setup
"""
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data/life_expectancy.sqlite"
db = SQLAlchemy(app)

class Surgery(db.Model):
    __tablename__ = 'surgery'

    id = db.Column(db.Integer, primary_key=True)
    diagnosis = db.Column(db.Float)
    fev = db.Column(db.Float)
    age = db.Column(db.Float)
    performance = db.Column(db.Float)
    tnm = db.Column(db.Float)
    pain = db.Column(db.Float)
    hae = db.Column(db.Float)
    dys = db.Column(db.Float)
    cough = db.Column(db.Float)
    weakness = db.Column(db.Float)
    dm = db.Column(db.Float)
    mi = db.Column(db.Float)
    pad = db.Column(db.Float)
    smoking = db.Column(db.Float)
    asthma = db.Column(db.Float)

    def __repr__(self):
        return '<Surgery %r>' % (self.name)

@app.before_first_request
def setup():
    db.create_all()

"""

@app.route("/")
def form():
	return render_template('index.html')


@app.route("/form", methods=['GET','POST'])
def getform():
    if request.method == "GET":
        return (render_template("form.html"))

    if request.method == 'POST':
        if 'submit-button' in request.form:
            diagnosis = request.form["diagnosis"]
            fev = request.form["fev"]
            age = request.form["age"]
            performance = request.form["performance"]
            tnm = request.form["tnm"]
            hae = request.form['hae']
            pain = request.form["pain"]
            dys = request.form["dys"]
            cough = request.form["cough"]
            weakness = request.form["weakness"]
            dm = request.form["dm"]
            mi = request.form["mi"]
            pad = request.form["pad"]
            smoking = request.form["smoking"]
            asthma = request.form["asthma"]

        #surgeryData = Surgery(diagnosis=diagnosis, fev=fev, age=age,performance=performance,tnm=tnm,pain=pain, hae=hae, dys=dys,cough=cough, weakness=weakness, dm=dm, mi=mi, pad=pad,smoking=smoking, asthma=asthma)
        
        #db.session.add(surgeryData)

        #db.session.commit()

            input_variables = pd.DataFrame([[performance, dys, cough, tnm, dm]], columns=['Performance', 'Dyspnoea', 'Cough', 'TNM', 'DM'], dtype=float)

            prediction = model.predict(input_variables)[0]

            if int(prediction) == 1:
                prediction = "Patient is at High Risk"
                flash("Patient is at High Risk")
                
            else:
                prediction = "Patient is Not at Risk"
                flash("Patient is Not at Risk")
            
            return render_template("form.html", prediction = prediction)
    
    return render_template("form.html")



if __name__ == "__main__":
    app.run(debug = True)     

"""    

@app.route("/send", methods=['GET','POST'])
def send():
    if request.method == 'POST':
        if 'submit-button' in request.form:
            diagnosis = request.form["diagnosis"]
            fev = request.form["fev"]
            age = request.form["age"]
            performance = request.form["performance"]
            tnm = request.form["tnm"]
            hae = request.form['hae']
            pain = request.form["pain"]
            dys = request.form["dys"]
            cough = request.form["cough"]
            weakness = request.form["weakness"]
            dm = request.form["dm"]
            mi = request.form["mi"]
            pad = request.form["pad"]
            smoking = request.form["smoking"]
            asthma = request.form["asthma"]

        #surgeryData = Surgery(diagnosis=diagnosis, fev=fev, age=age,performance=performance,tnm=tnm,pain=pain, hae=hae, dys=dys,cough=cough, weakness=weakness, dm=dm, mi=mi, pad=pad,smoking=smoking, asthma=asthma)
        
        #db.session.add(surgeryData)

        #db.session.commit()

            input_variables = pd.DataFrame([[performance, dys, cough, tnm, dm]], columns=['Performance', 'Dyspnoea', 'Cough', 'TNM', 'DM'], dtype=float)

            prediction = model.predict(input_variables)[0]

            return render_template("/", original_input = {'Performance': performance, 'Dyspnoea': dys, 'Cough': cough}, result = prediction)
        else:
            return ("Submit button not found")

"""
            
        

"""
@app.route("/data")
def X_data():
    results = db.session.query(Surgery.performance, Surgery.dys, Surgery.cough, Surgery.tnm, Surgery.dm).all()

    performance = [result[0] for result in results]
    dys = [result[1] for result in results]
    cough = [result[2] for result in results]
    tnm = [result[3] for result in results]
    dm = [result[4] for result in results]

    surgery_data = [{
        "performance": performance,
        "dys": dys,
        "cough":cough,
        "tnm": tnm,
        "dm": dm
    }]

    return jsonify(surgery_data)

@app.route("/api", methods=['POST'])
"""

#predictiom function
"""
def valuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1,5)
	loaded_model = pickle.load(open("model.pkl","rb"))
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route("/result/<postData>", methods=['GET','POST'])
def result(postData):
    
    data = list(postData)
    new_data = dict()
    #new_data["diagnosis"] = data[0]
    #new_data["fev"] = data[1]
    #new_data["age"] = data[2]
    new_data["performance"] = data[3]
    new_data["dys"] = data[7]
    new_data["cough"] = data[8]
    new_data["tnm"] = data[4]
    new_data["dm"] = data[10]
    #new_data["pain"] = data[5]
    #new_data["hae"] = data[6]
    #new_data["weakness"] = data[9]
    #new_data["mi"] = data[11]
    #new_data["pad"] = data[12]
    #new_data["smoking"] = data[13]
    #new_data["asthma"] = data[14]
    # data = to_dict(data)
    to_predict_list = new_data
    to_predict_list = list(to_predict_list.values())
    #to_predict_list = list(map(int, to_predict_list))
    result = valuePredictor(to_predict_list)

    if int(result) == 1:
        prediction = 'Patient is at High Risk'
    else:
        prediction = 'Patient is Not at Risk'

    print('at the end in py file')

    return render_template('result.html', prediction = prediction)

            data = []
            data.append(performance)
            data.append(dys)
            data.append(cough)
            data.append(tnm)
            data.append(dm)
      
            prediction = model.predict([data])

            if int(prediction) == 1:
                prediction = 'Patient is at High Risk'
            else:
                prediction = 'Patient is Not at Risk'

@app.route("/result")
def getresult():
    return render_template("result.html")
        
"""




        