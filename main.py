import uvicorn
from fastapi import FastAPI, Query
from typing import List
import joblib
import pandas as pd

app = FastAPI()
df=pd.read_csv('HeadthPred-Data.csv', encoding='cp1252')


def get_details(disease_name):
    details={}

    newdf = (df.loc[df['Disease'] == disease_name])
    desc = str(newdf.iloc[0]['Disease Description']).replace('\n', '')

    medications = str(newdf.iloc[0]['Medicine and Description']).replace('\n', '')
    medlist = medications.split('$')
    medications_name = []
    medications_desc = []

    # separate medications name and desc
    for medication in medlist:
        if medication != '':
            name_and_desc = medication.split(':')

            medications_name.append(name_and_desc[0])
            medications_desc.append(name_and_desc[1])
    details['description']=desc
    details['medications_names']=medications_name
    details['medications_desc']=medications_desc

    return details


@app.get('/')
def index():
    return {'message': 'Hello World'}


@app.get('/name')
def greet(name: str):
    return {'Welcome to FastAPI': f'{name}'}


@app.get('/predict-diabetes')
def predict_diabetes(arr: List[float] = Query(None)):
    # load classifier
    diabetes_classifier = joblib.load('models/diabetes-classifier.pkl')

    # make prediction
    saved_pred = diabetes_classifier.predict([arr])
    res = saved_pred[0]

    # Diabetes is True
    if res==1:
        data=get_details('Diabetes Disease')
        items=[{'prediction': str(res)}, {'description': data['description']},
               {'medication_names': data['medications_names']}, {'medications_desc': data['medications_desc']}]
        return items
    else:
        data = get_details('Diabetes Disease')
        items = [{'prediction': str(res)}, {'description': data['description']},
                 {'medication_names': []}, {'medications_desc': []}]
        return items


@app.get('/predict-diseases')
def predict_diseases(arr: List[float] = Query(None)):
    pred=[]
    # put zeroes
    for i in range(115):
        pred.append(0)

    # put actual symptoms
    for sym in arr:
        pred[int(sym)]=1

    # final input
    inp=[pred]

    # load classifier
    diseases_classifier = joblib.load('models/disease-classifier.pkl')

    # make prediction
    saved_pred = diseases_classifier.predict(inp)

    data = get_details(str(saved_pred[0]))
    items = [{'prediction': str(saved_pred[0])}, {'description': data['description']},
             {'medication_names': data['medications_names']}, {'medications_desc': data['medications_desc']}]

    return items


@app.get('/predict-heart')
def predict_heart(arr: List[float] = Query(None)):
    # load classifier
    heart_classifier = joblib.load('models/heart-classifier.pkl')

    # make prediction
    saved_pred = heart_classifier.predict([arr])
    res = saved_pred[0]

    if res == 1:
        data = get_details('Heart Disease')
        items = [{'prediction': str(res)}, {'description': data['description']},
                 {'medication_names': data['medications_names']}, {'medications_desc': data['medications_desc']}]
        return items
    else:
        data = get_details('Heart Disease')
        items = [{'prediction': str(res)}, {'description': data['description']},
                 {'medication_names': []}, {'medications_desc': []}]
        return items


@app.get('/predict-liver')
def predict_liver(arr: List[float] = Query(None)):
    # load classifier
    liver_classifier = joblib.load('models/liver-classifier.pkl')

    # make prediction
    saved_pred = liver_classifier.predict([arr])
    res = saved_pred[0]

    if res == 1:
        data = get_details('Liver Disease')
        items = [{'prediction': str(res)}, {'description': data['description']},
                 {'medication_names': data['medications_names']}, {'medications_desc': data['medications_desc']}]
        return items
    else:
        data = get_details('Liver Disease')
        items = [{'prediction': str(res)}, {'description': data['description']},
                 {'medication_names': []}, {'medications_desc': []}]
        return items


@app.get('/predict-cancer')
def predict_cancer(arr: List[float] = Query(None)):
    # load classifier
    cancer_classifier = joblib.load('models/cancer-classifier.pkl')

    # make prediction
    saved_pred = cancer_classifier.predict([arr])
    res = saved_pred[0]

    if res == "YES":
        data = get_details('Lung Cancer')
        items = [{'prediction': str(res)}, {'description': data['description']},
                 {'medication_names': data['medications_names']}, {'medications_desc': data['medications_desc']}]
        return items
    else:
        data = get_details('Lung Cancer')
        items = [{'prediction': str(res)}, {'description': data['description']},
                 {'medication_names': []}, {'medications_desc': []}]
        return items

# uvicorn main:app --reload
# python -m uvicorn main:app --reload
