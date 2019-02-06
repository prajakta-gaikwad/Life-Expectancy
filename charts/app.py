import pandas as pd
import json
from flask import (Flask, render_template)

app = Flask(__name__)

@app.before_first_request
def setup():
    df_lifeex = pd.read_csv("data/life-expectancy.csv")
    df_lifex = df_lifeex.rename(index=str, columns={"Entity":"Country","Life expectancy (Clio-Infra up to 1949; UN Population Division for 1950 to 2015)": "Life-ex"})
    df_lifex_filt = df_lifex.loc[df_lifex['Year'] == 2015]
    df_lifex2 = df_lifex_filt[['Country','Life-ex']]
    ret_lifex2 = df_lifex2.to_json(orient='records')

    return ret_lifeex2

@app.route("/")
def home():
	return render_template("Chloropleth.html")

# Returns selected year's life expectancy data for each country
@app.route("/<v_year>")
def year_data(v_year):

    df_lifex_filt = df_lifex.loc[df_lifex['Year'] == v_year]
    df_lifex2 = df_lifex_filt[['Country','Life-ex']]
    ret_lifex2 = df_lifex2.to_json(orient='records')

    return ret_lifeex2

@app.route("/<v_bmi>")
def bmi_calc(v_bmi):

    inputvalues =v_bmi.split("$$")    
    weight = inputvalues[0]
    height = inputvalues[1] 
    step1 = weight * .45
    step2 = height * 0.025
    step3 = step2 * step2
    ret_bmi = step1/step3
    
    return ret_bmi




if __name__ == "__main__":
	app.run(debug=True)

