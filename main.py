import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Carregar o modelo treinado
model = joblib.load('modelo_treinado.pkl')

# Criar instância da FastAPI
app = FastAPI()

# Definir o modelo Pydantic para a estrutura de entrada
class InputData(BaseModel):
    input: list[float]

# Criar a FastAPI com o modelo
@app.post('/predict')
async def predict(input_data: InputData):
    data = np.array(input_data.input)
    prediction = model.predict(data.reshape(1, -1))
    return {'prediction': prediction.tolist()}