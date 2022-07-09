from sre_constants import SUCCESS
from typing import Dict, List, Optional
from fastapi import FastAPI
from joblib import load
from tinydb import TinyDB, Query
from datetime import datetime
from tinydb.operations import set
import uvicorn
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
#correr la aplicación usando python main.py, corre en http://127.0.0.1:8000/
app = FastAPI(title="Lab 6")

# aquí carguen el modelo guardado (con load de joblib) y
model = load('modelo.joblib')
# el cliente de base de datos (con tinydb). Usen './db.json' como bbdd.
db = TinyDB('./db.json')

# Nota: En el caso que al guardar en la bbdd les salga una excepción del estilo JSONSerializable
# conviertan el tipo de dato a uno más sencillo.
# Por ejemplo, si al guardar la predicción les levanta este error, usen int(prediccion[0])
# para convertirla a un entero nativo de python.

# Nota 2: Las funciones ya están implementadas con todos sus parámetros. No deberían
# agregar más que esos.
vec = DictVectorizer()

@app.post("/potabilidad/")
async def predict_and_save(observation: Dict[str, float]):
    # probar desde una terminal linux usando:
    # curl -X POST http://127.0.0.1:8000/potabilidad/ -H 'Content-Type: application/json' -d '{"ph":10.316400384553162,"Hardness":217.2668424334475,"Solids":10676.508475429378,"Chloramines":3.445514571005745,"Sulfate":397.7549459751925,"Conductivity":492.20647361771086,"Organic_carbon":12.812732207582542,"Trihalomethanes":72.28192021570328,"Turbidity":3.4073494284238364}'
    
    #ordenamos los valores de la observación en el mismo ordwen que aparecen en la base de datos
    orden_columnas=[
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity"
    ]
    X=np.array([observation[columna] for columna in orden_columnas]).reshape(1, -1)
    pred_value = int(model.predict(X)[0])
    today = datetime.now()
    #actualizamos la base de datos en el mismo formato en que viene
    id=db.insert({"Day": today.day, "Month": today.month, "Year": today.year, "Prediction": pred_value})
    return { "potabilidad": pred_value,"id":id}

@app.get("/potabilidad/")
async def read_all():
    # implementar 2 aquí.
    return db.all()


@app.get("/potabilidad_diaria/")
async def read_by_day(day: int, month: int, year: int):
    # implementar 3 aquí
    consulta = Query()
    respuesta = db.search((consulta.Day==day)&(consulta.Month==month)&(consulta.Year==year))
    return respuesta


@app.put("/potabilidad/")
async def update_by_day(day: int, month: int, year: int, new_prediction: int):
    # implementar 4 aquí
    consulta = Query()
    ids = db.update(set('potabilidad',new_prediction),((consulta.Day==day)&(consulta.Month==month)&(consulta.Year==year)))
    updated = 0
    if len(ids)>0:
        updated = 1
    return {"success":updated,"id":ids}


@app.delete("/potabilidad/")
async def delete_by_day(day: int, month: int, year: int):
    # implementar 5 aquí
    #es lo mismo que el put pero con remove
    consulta = Query()
    ids = db.remove((consulta.Day==day)&(consulta.Month==month)&(consulta.Year==year))
    updated = 0
    if len(ids)>0:
        updated = 1
    return {"success":updated,"id":ids}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)