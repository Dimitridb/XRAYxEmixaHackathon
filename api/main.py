# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# Generate the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model1', 'model1.pkl')

# Load the model from the file
with open(model_path, 'rb') as f:
    model = pickle.load(f)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    predicted_class: int

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: IrisInput):
    data = np.array([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])
    prediction = model.predict(data)
    return PredictionOutput(predicted_class=int(prediction[0]))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Iris Prediction API"}
