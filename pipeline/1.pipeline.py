"""
-*- coding: utf-8 -*-
Creado 03 Septiembre 2023
@author: Carlos Luis Mora Cañas - Carlos Felipe Cortés Cataño
"""
# Importar librerias
import re
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from datetime import datetime
import numpy as np
import pandas as pd
import string  # Operaciones de cadenas de caracteres
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
# Manipulación datos

# parametros
umbral_reduccion_partesco = 0.03  # percentil 3
umbral_reduccion_profesiones = 0.01  # percentil 1
umbral_reduccion_gestion = 0.01  # percentil 1
dias_analisis_observaciones = 300  # Análsis texto ultimos días
# Modelo
# funciones
# Función para interpretar el sentimiento


def interpretar_sentimiento(row):
    if row["mean_neg"] > row["mean_pos"]:
        return "1"  # Negativo
    elif row["mean_pos"] > row["mean_neg"]:
        return "2"  # Positivo
    else:
        return "0"  # Neutro
# Función para generar los nombres de las columnas a partir del índice multi-nivel


def generate_column_names(columns):
    return [f"{col[1]}_{col[0]}" for col in columns]
# Función para limpiar el texto


def limpiar_texto(texto):
    # Poner el texto en minúsculas
    texto = texto.lower()
    # Quitar simbolos
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\n', '', texto)
    # Tokenizar el texto y quitar los signos de puntuación
    texto = [word.strip(string.punctuation) for word in texto.split(" ")]
    # Quitar las palabras que contengan números
    texto = [word for word in texto if not any(c.isdigit() for c in word)]
    # Quitar las stop words
    stop = stopwords.words('spanish')
    texto = [x for x in texto if x not in stop]
    # Quitar los tokens vacíos
    texto = [t for t in texto if len(t) > 0]
    # Unimos texto
    texto = ' '.join(texto)
    # Quitamos tildes
    texto = texto.replace("á", "a")
    texto = texto.replace("é", "e")
    texto = texto.replace("í", "i")
    texto = texto.replace("ó", "o")
    texto = texto.replace("ú", "u")
    return texto


# Archivo original, elimina duplicados para repetición codigo y estado actual
reporte = pd.read_excel(
    "../data/Consulta_promotora_082023.ods", engine="odf").sort_values("fechaSolicitud").drop_duplicates(
    subset=["CodigoPrograma", "EstadoActual"], keep="first")
# Creamos copia, eliminamos los registros que en la importación crean una nueva columna, o tiene errores y la columna

data = reporte.copy().drop(reporte[reporte["Unnamed: 21"].notna()].index).drop(
    ["Unnamed: 21"], axis=1)
data["idTomador"] = data["Documento_cliente"].replace(
    ["", "C", "M"], np.nan, regex=True)
data["valorCuota1"] = data["valorCuota_1"].replace(
    ["", "factura"], np.nan, regex=True).astype(float)
# Damos uniformidad a los tipos de valores y nombres de variables
# Definimos dia de pago
data["diaPagoCuota"] = data["fechaIdealPago_CuotaCancelada"].astype(str).str.split(
    "-", expand=True)[2].str.split(" ", expand=True)[0].fillna(np.nan).astype(float)
# Eliminamos variables no aplicables al modelo
data = data.drop(["fechaIdealPago_CuotaCancelada",
                 "Documento_cliente", "fechaNacimiento", "valorCuota_1", "coordenadas"], axis=1)
# Elimina registros vacios en tipo de programa necesario según reglas de negocio
data = data.drop(data[data["TipoPrograma"].isna()].index)
# Elimina registros con casilla corrida
data = data[data["FechaRescindido"].str.isspace().isna()]
# Elimina todos aquellos que sean empresariales, no hacen parte del estudio, solo familiares
data = data[data["TipoPrograma"] != "Empresarial"]
data = data.drop("TipoPrograma", axis=1)
# Imputamos valores del valor de la cuota, dependiendo de las medias de la cuota de inscritos y de mascotas
temp = data.groupby(["#_inscritos_activos", "#_mascotas_activas"]).mean()[
    ["valorCuota1", "valorUltimaCuota"]].reset_index()
temp = data[data["valorCuota1"].isna()].drop(
    ["valorCuota1", "valorUltimaCuota"], axis=1).merge(temp, on=["#_inscritos_activos", "#_mascotas_activas"])
data = data.drop(data[data["valorCuota1"].isna()].index)
data = pd.concat([data, temp], axis=0)
data = data.reset_index().drop("index", axis=1)
data = data.rename(columns={'#_inscritos_activos': "qPersonas",
                            '#_mascotas_activas': "qMascotas"})
# Calculamos duración del programa
# Si hay fecha de rescidindo usamos esa fecha sino hay valor nulo usamos la ultima fecha reportada en creación en estado
ultima_fecha_solicitud = pd.to_datetime(reporte["fechaSolicitud"].dropna(
).sort_values().tail(1).values[0])
data["FechaRescindido"] = data["FechaRescindido"].fillna(
    ultima_fecha_solicitud)
# Calculamos duración del programa
data["duracion"] = data["FechaRescindido"] - data["fechaSolicitud"]
data["duracion"] = data["duracion"].dt.days
# Existen 5 casos con valores negativos esto no debe ocurrir se igualan al promedio según cantidad mascotas y personas
temp = data.groupby(["qPersonas", "qMascotas"]).mean()[
    "duracion"].reset_index()
data = data.merge(temp, on=["qPersonas", "qMascotas"])
data["duracion"] = data["duracion_x"].where(
    data["duracion_x"] > 0, data["duracion_y"])
data = data.drop(["duracion_x", "duracion_y"], axis=1)
# Extraemos solo si se es activo o inactivo, se recomienda clasificar en más categorías en futuros proyectos
data["estado"] = data["EstadoActual"].str.split("-", expand=True)[0]
# Eliminamos columnas con información redudante capturada en variables extraidas
data = data.drop(["fechaSolicitud", "FechaRescindido",
                 "EstadoActual", "Estados"], axis=1)
# Conteo facturas generadas y promedio de descuentos otorgados
temp = data[["CodigoPrograma", "Cuotas"]].dropna()
temp["Cuotas"] = temp["Cuotas"].str.replace(
    "[", "", regex=True).str.replace("]", "", regex=True)
temp = temp.set_index("CodigoPrograma")
temp = temp["Cuotas"].str.split("},", expand=True).stack(
).reset_index().drop("level_1", axis=1).set_index("CodigoPrograma")
temp = temp[0].str.split(",", expand=True)
temp = temp.reset_index().rename(columns={0: "factura",
                                          1: "valorSinDescuento", 2: "valorcondescuento",
                                          3: "cuota", 4: "cuotaSinDescuento", 5: "cuotaConDescuento",
                                          6: "porcentajeDescuento", 7: "valorDescuento", 8: "periodoCuota"})
temp["factura"] = temp["factura"].str.replace(
    '{"factura":', "", regex=True).str.replace('"', "", regex=True)
temp["valorSinDescuento"] = temp["valorSinDescuento"].str.replace(
    '"valorTotalFacturaSinDescuento":', "", regex=True)
temp["valorcondescuento"] = temp["valorcondescuento"].str.replace(
    '"valorTotalFacturaConDescuento":', "", regex=True)
temp["cuota"] = temp["cuota"].str.replace(
    '"cuota":', "", regex=True).str.replace('"', "", regex=True)
temp["cuota"] = temp["cuota"].str.replace(
    '"cuota":', "", regex=True).str.replace('"', "", regex=True)
temp["cuotaSinDescuento"] = temp["cuotaSinDescuento"].str.replace(
    '"valorCuotaSinDescuento":', "", regex=True)
temp["cuotaConDescuento"] = temp["cuotaConDescuento"].str.replace(
    '"valorCuotaConDescuento":', "", regex=True)
temp["porcentajeDescuento"] = temp["porcentajeDescuento"].str.replace(
    '"porcentajeDescuento":', "", regex=True)
temp["valorDescuento"] = temp["valorDescuento"].str.replace(
    '"valorDescuento":', "", regex=True)
temp["periodoCuota"] = temp["periodoCuota"].str.replace(
    '"periodoCuota":', "", regex=True).str.replace('"', "", regex=True).str.replace(' ', "", regex=True)
cuota = temp["cuota"].str.replace("Contrato# ", "").str.replace("Cuota# ", "")\
    .str.split("-", expand=True).rename(columns={0: "contrato", 1: "cuota"})
cuotas = pd.concat([temp.drop("cuota", axis=1), cuota], axis=1)

temp = cuotas["periodoCuota"].str.replace(
    "\\", "", regex=True).str.replace(
    "}", "", regex=True).str.split("-", expand=True)\
    .rename(columns={0: "periodoInicial", 1: "periodoFinal"})
temp["periodoInicial"] = temp["periodoInicial"].apply(
    lambda x: datetime.strptime(x, "%d/%m/%Y"))
temp["periodoFinal"] = temp["periodoFinal"].apply(
    lambda x: datetime.strptime(x, "%d/%m/%Y"))
cuotas = pd.concat([cuotas.drop("periodoCuota", axis=1), temp], axis=1)
# Conteo de facturas generadas
new_cuotas = cuotas[["CodigoPrograma",
                     "factura"]].groupby("CodigoPrograma").count().reset_index().rename(columns={"factura": "qFacturas"})
# Promedio descuentos
new_cuotas["promPercDesc"] = cuotas[["CodigoPrograma", "porcentajeDescuento"]].astype("float")\
    .groupby("CodigoPrograma").mean().reset_index()["porcentajeDescuento"]
# Cantidad de descuento otorgados
cuotas["valorDescuento"] = cuotas["valorDescuento"].astype(float)
new_cuotas["qDescOtorgado"] = cuotas[cuotas["valorDescuento"]
                                     > 0].groupby("CodigoPrograma").count().reset_index()["valorDescuento"]

data = data.merge(new_cuotas, on="CodigoPrograma")
data["qDescOtorgado"] = data["qDescOtorgado"].fillna(0)
# Calculo de incremento cuota
data["cambioCuota"] = data["valorUltimaCuota"] - data["valorCuota1"]
# Eliminamos variables redundantes
data = data.drop(["Cuotas", "valorCuota1"], axis=1)
# Gestiones de recaudo
temp = data[["CodigoPrograma", "GestionesRecaudo"]].dropna()
temp["GestionesRecaudo"] = temp["GestionesRecaudo"].str.replace(
    "[", "", regex=True).str.replace("]", "", regex=True)
temp = temp.set_index("CodigoPrograma")
temp = temp["GestionesRecaudo"].str.split("},", expand=True).stack(
).reset_index().drop("level_1", axis=1).set_index("CodigoPrograma")
temp = temp[0].str.split(",", expand=True, n=3)
temp = temp.reset_index().rename(columns={0: "estado",
                                          1: "fechaenvio", 2: "fechaidealpago",
                                          3: "mensaje"})
temp["estado"] = temp["estado"].str.replace(
    '{"Estado":"', "", regex=True).str.replace('"', "", regex=True)
temp["fechaenvio"] = temp["fechaenvio"].str.replace(
    '"FechaEnvio":"', "", regex=True).str.replace('"', "", regex=True).str.replace('T', " ", regex=True)
temp["fechaidealpago"] = temp["fechaidealpago"].str.replace(
    '"fechaidealpago":"', "", regex=True).str.replace('"', "", regex=True).str.replace('T', " ", regex=True)
temp["mensaje"] = temp["mensaje"].str.replace(
    '"Mensaje":', "", regex=True).str.replace('"', "", regex=True)
gestion = temp["estado"].str.split(
    "-", expand=True).rename(columns={0: "estado", 1: "comentario"})
temp = temp.drop("estado", axis=1)
recaudos = pd.concat([temp, gestion], axis=1)
recaudos = recaudos.dropna(
    subset=["fechaenvio", "fechaidealpago", "estado", "CodigoPrograma"])
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
# Realizamos conteos de recaudos y los añadimos al dataframe
umbral = recaudos["comentario"].count()*umbral_reduccion_gestion
temp = recaudos[["CodigoPrograma", "comentario", "estado"]].groupby(
    ["CodigoPrograma", "comentario"]).count().reset_index()

# Si no cumplen el Umbral se reemplaza por otro para reducir la cantidad de atributos
no_umbral = recaudos.groupby("comentario").count()
no_umbral = no_umbral[no_umbral["estado"] < umbral].index
temp["comentario"] = temp["comentario"].where(
    ~temp["comentario"].isin(no_umbral), "otros")
temp = temp.drop_duplicates(subset=["CodigoPrograma", "comentario"])
temp = pd.pivot(temp, index=["CodigoPrograma"], columns=['comentario'])[
    "estado"].reset_index().drop("otros", axis=1)
temp = temp.rename(
    columns={" Gestión de recaudo cambio en cuotas de carpeta de pagos del cliente": "gestionCambioCuotas",
             " Gestión de recaudo exitosa": "gestionExitosa", " Gestión de recaudo no exitosa": "gestionNoExitoso",
             " Gestión no ejecutada para el día planeado": "gestionNoEjecutada"})
data = data.merge(temp, how="left", on="CodigoPrograma")
data[temp.columns] = data[temp.columns].fillna(0)
data = data.drop(["GestionesRecaudo", "Observaciones"], axis=1)
"""No se esta obteniendo adecuadamente los días de realización de la gestión
debido a que la fecha del envio no se reporta correctamente, se recomienda corregir este atributo
para calcular correctamente los días en que se realiza la gestión,
así mismo se evidencia que el atributo mensaje solo contiene 5 mensajes en una serie de recaudos
se recomienda dejar comentarios o mensajes para hacer análisis de texto también a este atributo"""
"""El nivel socioeconomico, variable significativa, se extrae de la latitud y longitud, sin embargo,
más del 75% de los datos no presentan estos datos, para complementarlos
se puede usar la dirección y la localidad de la venta pero se debe pagar
por el servicio ArcGis o uno similar, este presupuesto esta fuera del alcance del proyecto,
y no obtamos por perder esa cantidad de datos, se procede a eliminar las variables, solo usaremos la localidad"""
data = data.drop("Direccion", axis=1)
# Añade datos titulares
datos_titulares = pd.read_excel(
    "../data/base_datos_titulares_e_inscritos.xlsx")
prom_edad = datos_titulares[["NOMBRE_TOMADOR", "EDAD_INSCRITO"]]\
    .groupby("NOMBRE_TOMADOR").mean("EDAD_INSCRITO").reset_index()\
    .rename(columns={"NOMBRE_TOMADOR": "tomador", "EDAD_INSCRITO": "prom_edad_insc"})
# Se añade a la base de datos principal y los valores nulos se llenan con al menos la edad del tomador
data = data.merge(prom_edad, on="tomador", how="left")
data["prom_edad_insc"] = data["prom_edad_insc"].where(
    data["prom_edad_insc"].notna(), data["edad"])
# Nombre del plan y profesión del tomador
nom_plan = datos_titulares[["NOMBRE_TOMADOR", "PLAN_EXEQUIAL", "PROFESION_TOMADOR"]].drop_duplicates()\
    .rename(columns={"NOMBRE_TOMADOR": "tomador", "PLAN_EXEQUIAL": "nom_plan",
                     "PROFESION_TOMADOR": "profesion_tomador"})
data = data.merge(nom_plan, on="tomador", how="left")
# Llenamos nulos con identificador unico
data["profesion_tomador"] = data["profesion_tomador"].fillna("No Identificada")
data["nom_plan"] = data["nom_plan"].fillna("No Identificada")
# Reduciendo la cantidad de profesiones con limpieza texto
datos_titulares["PROFESION_INSCRITO"] = datos_titulares["PROFESION_INSCRITO"].astype(
    str).apply(limpiar_texto)
datos_titulares["PROFESION_INSCRITO"] = datos_titulares["PROFESION_INSCRITO"].fillna(
    "identificada").replace("", "otros").replace("nan", "otros").replace("ninguna", "otros")\
    .replace("ama casa", "hogar").replace("identificada", "otros")
# Reduciendo profesiones que no representen el percentil 1 de los datos
umbral = datos_titulares["SUCURSAL_VENTA"].count()*umbral_reduccion_profesiones
temp = datos_titulares[["NOMBRE_TOMADOR", "PROFESION_INSCRITO", "SUCURSAL_VENTA"]].groupby(
    ["NOMBRE_TOMADOR", "PROFESION_INSCRITO"]).count().reset_index()
# Si no cumplen el Umbral se reemplaza por otro para reducir la cantidad de atributos
no_umbral = temp.groupby("PROFESION_INSCRITO").count()
no_umbral = no_umbral[no_umbral["SUCURSAL_VENTA"] < umbral].index
temp["PROFESION_INSCRITO"] = temp["PROFESION_INSCRITO"].where(
    ~temp["PROFESION_INSCRITO"].isin(no_umbral), "otros")
temp = temp.drop_duplicates(subset=["NOMBRE_TOMADOR", "PROFESION_INSCRITO"])
temp = pd.pivot(temp, index=["NOMBRE_TOMADOR"], columns=['PROFESION_INSCRITO'])[
    "SUCURSAL_VENTA"].reset_index().drop("otros", axis=1)\
    .rename(columns={"NOMBRE_TOMADOR": "tomador", "PROFESION_INSCRITO": "moda_prof_inscritos"
                     })  # otros es categoría que generaliza
data = data.merge(temp, on="tomador", how="left")
data[temp.columns] = data[temp.columns].fillna(0)
# Reduciendo la cantidad de parentescos con limpieza texto
datos_titulares["PARENTESCO"] = datos_titulares["PARENTESCO"].astype(
    str).apply(limpiar_texto)
temp = datos_titulares[["NOMBRE_TOMADOR", "PROFESION_INSCRITO", "SUCURSAL_VENTA"]].groupby(
    ["NOMBRE_TOMADOR", "PROFESION_INSCRITO"]).count().reset_index()

datos_titulares["PARENTESCO"] = datos_titulares["PARENTESCO"].fillna(
    "otros").replace("", "otros")
# Reduciendo parentestos
temp = datos_titulares[["NOMBRE_TOMADOR", "PARENTESCO", "SUCURSAL_VENTA"]].groupby(
    ["NOMBRE_TOMADOR", "PARENTESCO"]).count().reset_index()
no_umbral = temp.groupby("PARENTESCO").count()
umbral = datos_titulares["SUCURSAL_VENTA"].count(
)*umbral_reduccion_partesco  # umbral al 3 en este caso
no_umbral = no_umbral[no_umbral["SUCURSAL_VENTA"] < umbral].index
temp["PARENTESCO"] = temp["PARENTESCO"].where(
    ~temp["PARENTESCO"].isin(no_umbral), "otros")
temp = temp.drop_duplicates(subset=["NOMBRE_TOMADOR", "PARENTESCO"])
temp = pd.pivot(temp, index=["NOMBRE_TOMADOR"], columns=['PARENTESCO'])[
    "SUCURSAL_VENTA"].reset_index().drop(["otros", "titular"], axis=1)\
    .rename(columns={"NOMBRE_TOMADOR": "tomador", "PARENTESCO": "moda_parentesco_inscritos"
                     })  # otros es categoría que generaliza
data = data.merge(temp, on="tomador", how="left")
data[temp.columns] = data[temp.columns].fillna(0)
data["profesion_tomador"] = data["profesion_tomador"].apply(
    limpiar_texto).replace(" ", "", regex=True).replace("identificada", "noIdentificado", regex=True).replace("", "noIdentificada")
data["nom_plan"] = data["nom_plan"].apply(
    limpiar_texto).replace(" ", "", regex=True).replace("identificada", "noIdentificado", regex=True)
data = data.rename(columns={"oficios varios": "oficiosVarios", "profesion_tomador": "profesionTomador",
                            "nom_plan": "nombrePlan", "prom_edad_insc": "promedioEdadInscritos"})
data["codigoPrograma"] = data["CodigoPrograma"]
# tenemos el id del tomador
data = data.drop("CodigoPrograma", axis=1).rename(
    columns={"LocalidadVenta": "localidad"})
data["localidad"] = data["localidad"].apply(limpiar_texto)
data = data.set_index("codigoPrograma")
# Hay valor de cuota con simbolo negativo
data = data[(data["valorUltimaCuota"] > 0) &
            (data["valorUltimaCuota"] < 1200000)]
# Existen más de 1 esposa y más de una madre
data = data[(data["madre"] < 2) & (data["esposa"] < 2) & (data["esposo"] < 2)]
# Reducir variables categoricas
variables_categoricas = ["qPersonas", "qMascotas", "nombrePlan", "agricultor",
                         "comerciante", "docente", "empleado", "estudiante", "localidad",
                         "hogar", 'independiente', 'oficiosVarios', 'pensionado', 'esposa', 'esposo',
                         'hermana', 'hermano', 'hija', 'hijo', 'madre']
# Definir umbral de baja frecuencia 1%
umbral = data.shape[0]*0.01
for variable_a_evaluar in variables_categoricas:
    data[variable_a_evaluar] = data[variable_a_evaluar].astype(str)
    frecuencia = data[variable_a_evaluar].value_counts()
    # Identificar categorías con baja frecuencia
    categorias_baja_frecuencia = frecuencia[frecuencia < umbral].index
    # Reemplazar categorías con baja frecuencia por un valor único
    data[variable_a_evaluar].loc[data[variable_a_evaluar].isin(
        categorias_baja_frecuencia)] = 'Otros'
data = data.reset_index()
# Modelado con reentrenamiento

variables_modelo = ['qPersonas', 'qMascotas', 'localidad', 'nombrePlan', 'agricultor',
                    'comerciante', 'docente', 'empleado', 'estudiante', 'hogar',
                    'independiente', 'oficiosVarios', 'pensionado', 'esposa', 'esposo',
                    'hermana', 'hermano', 'hija', 'hijo', 'madre', 'edad', 'diaPagoCuota',
                    'duracion', 'promPercDesc', 'gestionExitosa', 'gestionNoEjecutada',
                    'promedioEdadInscritos', 'estado']
data_model = data.copy()[variables_modelo]
variables_categoricas = ["qPersonas", "qMascotas",
                         "localidad", "inactivoadmin", "inactivocosto", "inactivofallecimiento",
                         "inactivoinactivo", "inactivoincumplimiento", "inactivoinfluencia", "inactivomala", "inactivorecuperado",
                         "inactivoubicacion", "inactivovoluntario", "sentimiento", "nombrePlan", "profesionTomador", "agricultor",
                         "comerciante", "docente", "empleado", "estudiante", "hogar", 'independiente', 'oficiosVarios', 'pensionado', 'esposa', 'esposo',
                         'hermana', 'hermano', 'hija', 'hijo', 'madre']

# Hot encoder para variables categoricas
labelencoder = LabelEncoder()
for column in variables_categoricas:
    try:
        data_model[column] = data_model[column].astype(str)
        data_model[column] = labelencoder.fit_transform(data_model[column])
    except:
        pass

# Encoder personalizado para estado 1 es inactivo
data_model["estado"] = data_model["estado"].replace(
    "Inactivo ", 1).replace("Activo ", 0)
# Balanceo


def random_undersample(X, y, majority_class):
    # Separa las instancias de la clase mayoritaria y minoritaria
    majority_X = X[y == majority_class]
    minority_X = X[y != majority_class]
    minority_y = y[y != majority_class]

    # Realiza undersampling aleatorio en la clase mayoritaria
    majority_X_undersampled = resample(majority_X,
                                       replace=False,  # No reemplazar las instancias
                                       # Igual número de instancias que la clase minoritaria
                                       n_samples=len(minority_y),
                                       random_state=42)  # Fijar una semilla para reproducibilidad

    # Combina las instancias undersampled de la clase mayoritaria con la clase minoritaria
    X_undersampled = np.concatenate([majority_X_undersampled, minority_X])
    y_undersampled = np.concatenate(
        [np.repeat(majority_class, len(minority_y)), minority_y])

    return X_undersampled, y_undersampled


X_undersampled, y_undersampled = random_undersample(
    data_model.drop("estado", axis=1), data_model["estado"], 0)
X_train, X_test, y_train, y_test = train_test_split(
    X_undersampled, y_undersampled, test_size=0.30, random_state=100)
# Escalar para facilitar calculos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Modelado
xgb_model = xgb.XGBClassifier(colsample_bytree=0.9,
                              learning_rate=0.1,
                              max_depth=7, n_estimators=300,
                              subsample=0.8)
xgb_model.fit(X_train, y_train)
# Realizar predicciones en el conjunto de prueba
predictions = xgb_model.predict(X_test)
# Evaluar
unique, counts = np.unique(
    labelencoder.inverse_transform(predictions), return_counts=True)
predictions_df = pd.DataFrame(
    counts, ["Activo", "Inactivo"]).rename(columns={0: "conteo"})
total = sum(counts)
predictions_df["porcentaje"] = round(predictions_df["conteo"] / total*100)
print("Metricas modelo XGB para identificar desertores")
print(predictions_df)
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
auc_roc = roc_auc_score(y_test, predictions,
                        average='weighted', multi_class='ovr')
print("Recall:", round(recall, 2))
print("F1-Score:", round(f1, 2))
print("AUC-ROC:", round(auc_roc, 2))
programas_activos = data_model[data_model["estado"] == 0]
descertores = xgb_model.predict(
    programas_activos.drop("estado", axis=1))
data_results = data[data["estado"] == "Activo "]
data_results["results"] = descertores
print("Los programas propensos a desertar son:")
codigodescertores = data_results[(data_results["estado"] == "Activo ") & (
    data_results["results"] == 1)]
print(codigodescertores["codigoPrograma"].values)
# Especifica la ruta y el nombre del archivo
# Obtener la fecha y hora actual
fecha_actual = datetime.now().date()
ruta_archivo = f'../data/outputs/desertores{fecha_actual}.txt'
ruta_excel = f'../data/outputs/desertores{fecha_actual}.csv'
# Abre el archivo en modo escritura
recall = round(recall, 2)*100
with open(ruta_archivo, "w") as archivo:
    archivo.write(
        f'Los programas propensos a desertar son {codigodescertores["codigoPrograma"].values} con una tasa de {recall}')

print("Texto exportado correctamente al archivo:", ruta_archivo)
codigodescertores.to_csv(ruta_excel, index=False)
