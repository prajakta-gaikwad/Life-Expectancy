# Life-Expectancy
This project is devided in two parts:
* Developing a Machine Learning Web Application to determine Post-Operative Life Expectancy of Lung Cancer Patients
* Creating a Tableau dashboard to visualize trends in life expectancy (in general) in different countries over time

#### Project website is hosted on [https://life-expectancy.herokuapp.com/](https://life-expectancy.herokuapp.com/)

# Technologies Used
* Python
  * Pandas
  * NumPy
  * Matplotlib
  * Seaborn
 * Machine Learning Libraries
   * Scikit-Learn
   * XGBoost
 * Python Flask RESTful API
 * Tableau
 * HTML/CSS/Bootstrap
 * Javascript
 * Heroku Platform

# Part 1 - **LUNG**EVITY Machine Learning Web App
![Alt text](tho.jpg?raw=true "Optional Title")

### Problem Statement:
<p>
Lung cancer is the leading cause of cancer-related deaths in the world. In the United States, lung cancer claims more lives every year than colon cancer, prostate cancer, and breast cancer combined.
Patients who receive <b>Thoracic Surgery for Lung Cancer</b> do so with the expectation that their lives will be prolonged for a sufficient amount of time after the surgery.
</p><p>
The problem this app tries to solve is to find whether there is a way to determine <b>Post-Operative Life Expectancy</b> from patient attributes in the dataset. If there is pattern to be recognized, this would help physicians and patients make a more educated decision on whether they should proceed forward with the surgery.
<br>Not only would this influence physicians and patients but also health insurance companies, national health organizations and clinical researchers.</p>

### Scope of the Project:
The scope of this project is to examine the mortality of patients within a full year after the surgery. More specifically,
we are examining the underlying health factors of patients that could potentially be a powerful predictor
for surgery related deaths.

### Data Sources:
[https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data](https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data)
* The dataset obtained from UCI's website was collected retrospectively at Wroclaw Thoracic Surgery Centre for patients who underwent major lung resections for primary lung cancer in the years 2007-2011.

![Alt text](data_1.PNG?raw=true "Optional Title")

|   Attribute  |  Description  |
|--------------|--------------|
|   **Diagnosis**  | ICD-10 codes for primary and secondary as well multiple tumors if any |
|   **FVC**  | Amount of air which can be forcibly exhaled from the lungs after taking the deepest breath possible |
|   **Performance**  | Performance status on Zubrod scale, Good (0) to Poor (2) |
|   **Pain**  | Pain before surgery (T = 1, F = 0)  |
|   **Haemoptysis**  | Coughing up blood, before surgery (T = 1, F = 0) |
|   **Dyspnoea**  | Difficulty or labored breathing, before surgery (T = 1, F = 0)  |
|   **Cough**  | Symptoms of Coughing, before surgery (T = 1, F = 0)   |
|   **Weakness**  | Weakness, before surgery (T = 1, F = 0)  |
|   **TNM**  |  Size of the original tumor, 1 (smallest) to 4 (largest) |
|   **DM**  | Type 2 diabetes mellitus (T = 1, F = 0)   |\n",
|   **MI**  | Myocardial infarction (Heart Attack), up to 6 months prior(T = 1, F = 0)   |
|   **PAD**  | Peripheral arterial diseases (T = 1, F = 0)   |
|   **Smoking**  | Patient smoked (T = 1, F = 0)   |
|   **Asthma**  | Patient has asthma (T = 1, F = 0)   |
|   **Age**  | Age at surgery   |\n",
|   **Risk1Y**  | 1 year survival period - (T) value if died (T = 1, F = 0)    |

### Initial Data Cleaning and Exploration:
* Data did not have any Null values
* Many columns had categorical variables with most of them being True and Flase values. These were converted to 1 and 0 int data types. 

```
df.replace({
    'T':1,
    'F':0,
    'PRZ0':0,
    'PRZ1':1,
    'PRZ2':2,
    'OC10':0,
    'OC11':1,
    'OC12':2,
    'OC13':3,
    'OC14':4,
    'OC15':5
},inplace=True)
```
* Out of 470 patients, 70 **did not survive** 1 year after the surgery which is **14.89%** of the total sample size.

![Alt text](initial_de.PNG?raw=true "Optional Title")



![Alt text](initial_de4.PNG?raw=true "Optional Title")

* The most notable attributes for those who died are Dyspnoea, Diabetes Mellitus, Pain, PAD, and Haemoptysis, this indicates that for those who died, these features were strongly presented.

![Alt text](initial_de2.PNG?raw=true "Optional Title")

* It is clear that Cough and Smoking are strongly correlated to those patients who are to recieve thoracic surgery.

![Alt text](initial_de3.PNG?raw=true "Optional Title")



### Machine Learning
![Alt text](ml.jpeg?raw=true "Optional Title")
#### Libraries Used:
![Alt text](scikit.png?raw=true "Optional Title")
![Alt text](xgboost.jpg?raw=true "Optional Title")

#### Process:
* Exploratory analysis suggested that the data is highely imbalanced as the proportion of the patients that did not survive is just about 15%.
* Hence, only the attributes of significance are considered for building Machine Learning models

#### Models Trained:
* Logistic Regression
* Decision Tree
* Random Forests
* KNN
* SVM
* XGB Classifier

![Alt text](logistic_regression.PNG?raw=true "Optional Title")
![Alt text](tree_forest.PNG?raw=true "Optional Title")

#### Limitations faced and strategies used to improve model performance:
* The data had a significant class imbalance as the patients that did not survive(class B) are just 15% of the data-set unlike (class A) which is 85% of the data .
* Although all the models (from scikit-learn library) reached an accuracy of around 90%, it was simply by predicting class A every time.
* To deal with class imbalance, properly calibrated method may achieve a lower accuracy, but would have a substantially higher true positive rate(or Recall).
* Techniques used - Using the class weights and Oversampling the minority class i.e. to duplicate the minority class entries.
* Using class weights also did not improve models' performance.
* With oversampling and using XGBoost library's XGB Classifier algorithm accuracy of 69.33 % was achieved.
* Although several options are explored in this project, there are many more going beyond this result.
* More data and additional features will improve the scope of this project and model's performance.

![Alt text](oversampling.PNG?raw=true "Optional Title")

![Alt text](xgboost_model.PNG?raw=true "Optional Title")


### Web App:
![Alt text](architecture.png?raw=true "Optional Title")
#### Front-end:
* Created a web form using HTML, CSS and Bootstrap
#### Backend:
* The trained model is saved to disk using **pickle** 
* Python code loads the model and gets the user input from the web form.
* POST method in the web form tells the python app to expect an input and therefore process it. 
* python function takes input variables, transforms them into the appropriate format, and returns predictions.
* Flask web framework was used as RESTful API



# Part 2 - Tableau Dashboard
![Alt text](lifespan.jpg?raw=true "Optional Title")


![Alt text](tableau1.gif?raw=true "Optional Title")


## Data cleaning and merging

#### We had a very large geojson file (23mb)

![Alt text](img1.png?raw=true "Optional Title")

#### Running this file really bogged down the system. I found mapshaper.com
#### This site allows you to lower the resolution and merge borders of a file to shrink it.

## <a href="http://www.icdzn.com/misc/mapshaper.gif" target="_blank">Map Shaper!</a>

#### The new file is only 600kb
![Alt text](img2.png?raw=true "Optional Title")

#### The next step was to append the related data from a .csv file into the geojson.  
#### The output was logged to see what matched up and what did not.  
#### Instances like 'United States' vs 'United States of America' had to be cleaned in order for the data to match.

![Alt text](img3.png?raw=true "Optional Title")

#### LifeX was added in as a property of the feature.  for each matching feature we aded a list of years and life expectancies.

![Alt text](img4.png?raw=true "Optional Title")

#### A Flask was created but not used since we decided to move the charts to Tableau because of time contraints.  This included a BMI calculator.

![Alt text](img5.png?raw=true "Optional Title")

#### A choropleth was created in leaflet but was later moved to Tableau.
![Alt text](img6.png?raw=true "Optional Title")

#### Screenshots of the website
![Alt text](homepage.png?raw=true "Optional Title")
![Alt text](webform.png?raw=true "Optional Title")
![Alt text](dashboard.png?raw=true "Optional Title")


