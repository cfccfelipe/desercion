# -*- coding: utf-8 -*-
"""
Creado 30 Agosto 2023

@author: Carlos Luis Mora Cañas - Carlos Felipe Cortés Cataño
"""

# Análisis exploratorio a cuotas y estados

# Importar librerias

import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import re
import matplotlib.pyplot as plt

# Lectura de Archivos

estados = pd.read_csv("../data/outputs/estados.csv")
estados = estados[["CodigoPrograma", "estado", "fecha"]]

cuotas = pd.read_csv("../data/outputs/cuotas.csv")

# Ajuste de Dataset de nombres y variables

estados = pd.concat([estados.drop("estado", axis=1),
                     estados["estado"].str.split("-", expand=True, n=1)
                     .rename(columns={0: "estado", 1: "motivo"})], axis=1)

# Eliminacion de Duplicados

new_estados = estados.drop_duplicates("CodigoPrograma")["CodigoPrograma"]

# Analisis estados Activos e Inactivos

temp = estados.groupby(["CodigoPrograma", "estado"]
                       ).count().sort_values("fecha").reset_index()
temp["estado"] = temp["estado"].str.replace(" ", "")
new_estados = temp[temp["estado"] == "Activo"][["CodigoPrograma",
                                                "estado", "fecha"]]\
    .merge(new_estados, on="CodigoPrograma")\
    .rename(columns={"fecha": "qactivos"})


temp = estados.groupby(["CodigoPrograma", "estado"]
                       ).count().sort_values("fecha").reset_index()
temp["estado"] = temp["estado"].str.replace(" ", "")
new_estados = temp[temp["estado"] == "Inactivo"][["CodigoPrograma",
                                                  "estado", "fecha"]]\
    .merge(new_estados, on="CodigoPrograma")\
    .rename(columns={"fecha": "qinactivos"})

new_estados = new_estados.drop(["estado_x", "estado_y"], axis=1)


# Se ordenan para obtener la fecha mas reciente y la fecha final para activos e inactivos

new_estados = new_estados.merge(estados[estados["estado"] == "Activo "]
                                .sort_values("fecha").drop_duplicates("CodigoPrograma", keep="first")
                                .drop("estado", axis=1).rename(columns={"fecha": "fechaactivoi",
                                                                        "motivo": "motivoactivoi"}), on="CodigoPrograma")\

new_estados = new_estados.merge(estados[estados["estado"] == "Inactivo "]
                                .sort_values("fecha").drop_duplicates("CodigoPrograma", keep="last")
                                .drop("estado", axis=1).rename(columns={"fecha": "fechainactivof",
                                                                        "motivo": "motivoinactivof"}), on="CodigoPrograma")

new_estados = new_estados.merge(estados[estados["estado"] == "Inactivo "]
                                .sort_values("fecha").drop_duplicates("CodigoPrograma", keep="first")
                                .drop("estado", axis=1).rename(columns={"fecha": "fechainactivoi",
                                                                        "motivo": "motivoinactivoi"}), on="CodigoPrograma")

new_estados = new_estados.merge(estados[estados["estado"] == "Activo "]
                                .sort_values("fecha").drop_duplicates("CodigoPrograma", keep="last")
                                .drop("estado", axis=1).rename(columns={"fecha": "fechaactivof",
                                                                        "motivo": "motivoactivof"}), on="CodigoPrograma")

# Nuevo Dataset con nuevas columnas de estado

new_estados["fechaactivof"] = new_estados["fechaactivof"].apply(lambda x: datetime.strptime(
    x[:10], "%Y-%m-%d"))
new_estados["fechaactivoi"] = new_estados["fechaactivoi"].apply(lambda x: datetime.strptime(
    x[:10], "%Y-%m-%d"))
new_estados["fechainactivoi"] = new_estados["fechainactivoi"].apply(lambda x: datetime.strptime(
    x[:10], "%Y-%m-%d"))
new_estados["fechainactivof"] = new_estados["fechainactivof"].apply(lambda x: datetime.strptime(
    x[:10], "%Y-%m-%d"))

# Se Exporta csv

new_estados.to_csv("../data/outputs/estados_v2.csv", index=False)


# Recaudos

# Lectura de Archivos

recaudos = pd.read_csv("../data/outputs/gestionrecaudo.csv")
recaudos = recaudos.dropna(
    subset=["fechaenvio", "fechaidealpago", "estado", "CodigoPrograma"])

# Ajuste de Dataset de nombres, variables y eliminacion de columnas innecesarias

recaudos = recaudos.drop(
    recaudos[recaudos["fechaidealpago"].str.isalpha()].index)
recaudos = recaudos.drop(
    recaudos[recaudos["fechaidealpago"].str.contains("a")].index)
recaudos = recaudos.drop(
    recaudos[recaudos["fechaidealpago"].apply(lambda x: len(x) < 10)].index)
recaudos = recaudos.drop(
    recaudos[recaudos["fechaenvio"].str.contains("a")].index)
recaudos = recaudos.drop(
    recaudos[recaudos["fechaenvio"].apply(lambda x: len(x) < 10)].index)
recaudos["fechaenvio"] = recaudos["fechaenvio"].apply(lambda x: datetime.strptime(
    x[:10], "%Y-%m-%d"))
recaudos["fechaidealpago"] = recaudos["fechaidealpago"].apply(lambda x: datetime.strptime(
    x[:10], "%Y-%m-%d"))

recaudos["gestion"] = recaudos["fechaenvio"] - recaudos["fechaidealpago"]

recaudos[recaudos["gestion"] < dt.timedelta(days=1)]


# Analisis De Estado

temp1 = recaudos[["CodigoPrograma", "estado"]][recaudos["estado"] == "Exitosa "]\
    .groupby(["CodigoPrograma"]).count().reset_index().rename(columns={"estado": "recauExitoso"})

temp2 = recaudos[["CodigoPrograma", "estado"]][recaudos["estado"] == "Fallido "]\
    .groupby(["CodigoPrograma"]).count().reset_index().rename(columns={"estado": "recauFallido"})

recaudoexp = temp1.merge(temp2, on="CodigoPrograma")

# Exportar Dataset

recaudoexp.to_csv("../data/outputs/gestionrecaudo_v2.csv", index=False)

# Merge con base de datos principal

# Lectura de Archivos procesados

principal = pd.read_csv("../data/outputs/principal_v0.1.csv")
pegarestados = pd.read_csv("../data/outputs/estados_v2.csv")
pegarrecaudo = pd.read_csv("../data/outputs/gestionrecaudo_v2.csv")
direcciones = pd.read_excel("../data/outputs/Direcciones_faltantes.xlsx")
datos_titulares = pd.read_excel(
    "../data/base_datos_titulares_e_inscritos.xlsx")
sentiment = pd.read_csv("../data/outputs/df_tweets_proc_df.csv", sep=";")
data0 = pd.read_csv("../data/consulta_promotora.csv", encoding='latin-1')

principal = principal.merge(pegarestados, on="CodigoPrograma").merge(
    pegarrecaudo, on="CodigoPrograma")

# Ajuste de los Dataset

direcciones = direcciones[["CodigoPrograma", "Direccion", "LocalidadVenta",
                           "Nivel Socio Economico", "Barrio", "Localidad", "Longitud", "Latitud"]]

principal = principal.drop(["Unnamed: 0", "Direccion", "longitud",
                            "latitud", "LocalidadVenta"], axis=1)

# Merge

principal = principal.merge(direcciones, on="CodigoPrograma")

# Exportar Dataset

principal.to_csv("../data/outputs/principal_v0.2.csv", index=False)

principal = pd.read_csv("../data/outputs/principal_v0.2.csv")

# Promedio edad afiliados tomador

prom_edad = datos_titulares[["NOMBRE_TOMADOR", "EDAD_INSCRITO"]]\
    .groupby("NOMBRE_TOMADOR").mean("EDAD_INSCRITO").reset_index()\
    .rename(columns={"NOMBRE_TOMADOR": "tomador", "EDAD_INSCRITO": "prom_edad_insc"})
prom_edad["tomador"] = prom_edad["tomador"].str.lower().str.replace(" ", "")
principal["tomador"] = principal["tomador"].str.lower().str.replace(" ", "")
data = principal.merge(prom_edad, on="tomador", how="left")

# Profesion y nombre de plan

nom_plan = datos_titulares[["NOMBRE_TOMADOR", "PLAN_EXEQUIAL", "PROFESION_TOMADOR"]].drop_duplicates()\
    .rename(columns={"NOMBRE_TOMADOR": "tomador", "PLAN_EXEQUIAL": "nom_plan",
                     "PROFESION_TOMADOR": "profesion_tomador"})
data = principal.merge(nom_plan, on="tomador", how="left")

moda_prof_inscritos = datos_titulares[["NOMBRE_TOMADOR", "PROFESION_INSCRITO"]]\
    .sort_values("PROFESION_INSCRITO")\
    .groupby("NOMBRE_TOMADOR").agg(pd.Series.mode).reset_index()\
    .rename(columns={"NOMBRE_TOMADOR": "tomador", "PROFESION_INSCRITO": "moda_prof_inscritos"
                     })

moda_prof_inscritos["moda_prof_inscritos"] = moda_prof_inscritos["moda_prof_inscritos"].apply(
    lambda x: x if type(x) == str else " - ".join(x))
data = data.merge(moda_prof_inscritos, on="tomador", how="left")

moda_parentesco = datos_titulares[["NOMBRE_TOMADOR", "PARENTESCO"]]\
    .sort_values("PARENTESCO")\
    .groupby("NOMBRE_TOMADOR").agg(pd.Series.mode).reset_index()\
    .rename(columns={"NOMBRE_TOMADOR": "tomador", "PARENTESCO": "moda_parentesco"
                     })

moda_parentesco["moda_parentesco"] = moda_parentesco["moda_parentesco"].apply(
    lambda x: x if type(x) == str else " - ".join(x))
data = data.merge(moda_parentesco, on="tomador", how="left")


# Datos Sentimientos

sentiment = sentiment[["CodigoPrograma", "Neutral Score",
                       "Negative Score", "Positive Score"]]
sentiment = sentiment.groupby("CodigoPrograma").mean().reset_index()
data = data.merge(sentiment, on="CodigoPrograma", how="left")

# Exportar Datos

data.to_csv("../data/outputs/principal_v0.3.csv", index=False)


# Reemplaza 'grupo' y 'A' con los nombres de tus columnas y categorías

data2 = pd.read_csv("../data/outputs/principal_v0.2.csv")

group1 = data[data["EstadoActual"].str.contains(
    "Activo - ")][["edad", "#_inscritos_activos"]]
group2 = data[data["EstadoActual"].str.contains(
    "Inactivo - ")][["edad", "#_inscritos_activos"]]

# Reemplaza 'grupo' y 'A' con los nombres de tus columnas y categorías
group1 = data2[data2["EstadoActual"].str.contains(
    "Activo - ")][["edad", "#_inscritos_activos"]]
group2 = data2[data2["EstadoActual"].str.contains(
    "Inactivo - ")][["edad", "#_inscritos_activos"]]
