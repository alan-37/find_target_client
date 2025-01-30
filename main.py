import dill
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


app = FastAPI()
with open('model/event_action.dill', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_browser: str
    device_screen_resolution: str
    geo_country: str
    geo_city: str
    visit_date: str
    visit_number: int



class Prediction(BaseModel):
    target: int


@app.get('/status')
def status():
    return 'Im OK'


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'target': y[0],
    }
