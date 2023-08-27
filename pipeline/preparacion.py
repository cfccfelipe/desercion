
"""
-*- coding: utf-8 -*-
Creado 21 Agosto 2023
@author: Carlos Luis Mora Cañas - Carlos Felipe Cortés Cataño
"""
# Importar librerias
# Manipulación datos
from textblob import TextBlob  # Traductor ingles para mayor precisión
# Analisis de sentimiento
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string  # Operaciones de cadenas de caracteres
import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt

# Tratamiento texto
import re
import nltk  # Procesamiento del lenguaje natural
nltk.download('averaged_perceptron_tagger')  # tagger
nltk.download('vader_lexicon')  # Lexicon
nltk.download('wordnet')  # Categorizacion de las palabras
nltk.download('stopwords')  # Quitar palabras comunes
# pip uninstall vaderSentiment
# pip install vader-multi

# parametros
umbral_reduccion_partesco = 0.03  # percentil 3
umbral_reduccion_profesiones = 0.01  # percentil 1
umbral_reduccion_gestion = 0.01  # percentil 1
dias_analisis_observaciones = 300  # Análsis texto ultimos días

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
data = data.drop(data[~data["EstadoActual"].str.contains("-")].index)
# Damos uniformidad a los tipos de valores
data["idTomador"] = data["Documento_cliente"].replace(
    ["", "C", "M"], np.nan, regex=True)
data["valorCuota1"] = data["valorCuota_1"].replace(
    ["", "factura"], np.nan, regex=True).astype(float)
data["diaPagoCuota"] = data["fechaIdealPago_CuotaCancelada"].astype(str).str.split(
    "-", expand=True)[2].str.split(" ", expand=True)[0].fillna(np.nan).astype(float)
# Se cambiaron los nombres y se elimina fecha de Nacimiento porque es redundante con edad
data = data.drop(["fechaIdealPago_CuotaCancelada",
                 "Documento_cliente", "valorCuota_1", "fechaNacimiento"], axis=1)

# Elimina registros vacios en tipo de programa necesario según reglas de negocio
data = data.drop(data[data["TipoPrograma"].isna()].index)

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
# División del diccionario longitud y latitud en 2 atributos
temp = data[["CodigoPrograma", "coordenadas"]].dropna()
temp = temp.set_index("CodigoPrograma")
temp["coordenadas"] = temp["coordenadas"].apply(
    lambda indice: indice.split(',"pov":')[0])
temp["coordenadas"] = temp["coordenadas"].apply(
    lambda indice: indice.replace('{"pos":{"latitud":', "").
    replace("}", "").replace('"longitud":', ""))
temp = temp["coordenadas"].str.split(",", expand=True).rename(
    columns={0: "latitud", 1: "longitud"}).reset_index()
data = data.merge(temp, on="CodigoPrograma",
                  how="left").drop("coordenadas", axis=1)

# Estados
# Existen estados que son casos especiales, no hacen parte del estudio
estados_especiales = ["Activo - Pendiente de autorización",
                      "Activo - Programa con inconsistencia", "Activo - Activo para verificación",
                      "Inactivo - Venta no efectiva", "Inactivo - Pendiente primer pago",
                      "Inactivo - Pendiente de autorización", 'Inactivo - Programa con inconsistencia severa',
                      "Inactivo - Programa con inconsistencia severa", "Inactivo - Alianza Olivos Promollano 2021"
                      'Inactivo - Programa pendiente de activación por empresa', "Inactivo - Desvinculación de la empresa",
                      "Inactivo - Trámites para realizar contrato", "Inactivo - Programa cedido a Santa Rosa o Alto de occidente",
                      "Inactivo - Cambio de forma de pago (Alto Occidente_Santa _Rosa)", "Inactivo - Alianza Olivos Promollano 2021",
                      "Inactivo - Equipos", "Inactivo - Plenitud 50 pendiente por definir", "Activo - Programa pendiente de activación por empresa",
                      "Inactivo - Cancelado Propietario CHEC"
                      ]

data = data.drop(data[data["EstadoActual"].astype(
    str).isin(estados_especiales)].index)
# Definición y validación estados del programa
estados = pd.DataFrame(columns=["CodigoPrograma", "estado", "fecha"])
# Primer inactivo - Fecha de rescindido
temp = data[["CodigoPrograma", "FechaRescindido", "EstadoActual"]].rename(
    columns={"FechaRescindido": "fecha", "EstadoActual": "estado"})
temp = temp.drop(temp[temp["fecha"].isna()].index)
estados = pd.concat([estados, temp])
# Estados contenidos en el atributo Estados
temp = data[["CodigoPrograma", "Estados"]].dropna()
temp["Estados"] = temp["Estados"].str.replace(
    "[", "", regex=True).str.replace("]", "", regex=True)
temp = temp.set_index("CodigoPrograma")
temp = temp["Estados"].str.split("},", expand=True).stack(
).reset_index().drop("level_1", axis=1).set_index("CodigoPrograma")
temp = temp[0].str.split(",", expand=True)
temp[1] = temp[2].where(~temp[2].isna(), temp[1])
temp = temp.reset_index().rename(columns={0: "estado",
                                          1: "fecha"}).drop(2, axis=1)
temp["estado"] = temp["estado"].str.replace(
    '{"Estado":"', "", regex=True).str.replace('"', "", regex=True)
temp["fecha"] = temp["fecha"].str.replace(
    '"fechainicio":"', "", regex=True).str.replace(
    '"fechacancelacion":"', "", regex=True).str.replace('"}', "",
                                                        regex=True).str.replace('"', "", regex=True).str.replace('T', " ", regex=True)
estados = pd.concat([estados, temp])
# Eliminamos nuevamente estados que son especiales pero esta vez de nuestro nuevo dataframe
estados = estados.drop(estados[estados["estado"].astype(
    str).isin(estados_especiales)].index)
# Eliminamos estados duplicados por fecha y estado
estados = estados.drop_duplicates(subset=["CodigoPrograma", "estado"])
# Agrupación motivos
estados = pd.concat([estados.drop("estado", axis=1),
                     estados["estado"].str.split("-", expand=True, n=1)
                     .rename(columns={0: "estado", 1: "motivo"})], axis=1)
remplazo_motivos = {
    "Percepción negativa de la empresa por comentarios de un tercero": "Influencia de seres cercanos",
    "Cambio de programa- inscritos pasan a plan nuevo Aurora": "Admin", "Terminación y uso del contrato": "Admin",
    "Cambio de lugar de vivienda": "Ubicación", "Dificultad para ubicarlo": "Ubicación",
    "Se retira por 50% del servicio por atraso": "Incumplimiento", "Sala de velación": "Mala",
    "Reportado por la CHEC": "Incumplimiento", "Cliente de Alto Riesgo": "Incumplimiento",
    "Cobertura del servicio": "Mala", "Cobros indebidos": "Mala", "Inconformidad en el recaudo": "Mala",
    "Precio del plan": "Costo", "PENDIENTE DEFINIR RETIRO": "Inactivo", "No Interesado": "Voluntario",
    "Problemas económicos": "Voluntario", "Pago extendido y no uso del servicio": "Voluntario",
    "Programa pendiente de activación por empresa": "Inactivo", "Mejoramiento del estilo de vida": "Influencia", "Doblemente afiliado en la aurora": "Admin",
    "Parque cementerio": "Mala", "Parque crematorio": "Mala", "Experimentación": "Admin"}

estados = estados.replace(remplazo_motivos, regex=True)
estados["motivo"] = estados["motivo"].str.split(" ", expand=True)[1]
estados["concat"] = estados["estado"].str.cat(
    estados["motivo"], "- ").apply(limpiar_texto).replace(" ", "", regex=True)
# Los estados activos no aportan a la medición
estados = estados.drop(estados[(estados["concat"] == "activoactivo")].index)
# Añadiendo los conteos al dataframe principal
temp = estados.groupby(["CodigoPrograma", "concat"]).count()[
    "fecha"].reset_index()
temp = pd.pivot(temp, index=["CodigoPrograma"], columns=['concat'])[
    "fecha"].reset_index().fillna(0)
data = data.merge(temp, how="left", on="CodigoPrograma")
data[temp.columns] = data[temp.columns].fillna(0)

# La fecha de rescindido la usamos para obtener la duración del programa
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


# Análisis de texto con observaciones
temp = data[["CodigoPrograma", "Observaciones"]].dropna()
temp["Observaciones"] = temp["Observaciones"].str.replace(
    "[", "", regex=True).str.replace("]", "", regex=True)
temp = temp.set_index("CodigoPrograma")
temp = temp["Observaciones"].str.split("},", expand=True).stack(
).reset_index().drop("level_1", axis=1).set_index("CodigoPrograma")
temp = temp[0].str.split(",", expand=True, n=2)
temp = temp.reset_index().rename(columns={0: "fecha",
                                          1: "empleado", 2: "observacion",
                                          })
temp["fecha"] = temp["fecha"].str.replace(
    '{"fechaIngreso":"', "", regex=True).str.replace('"', "", regex=True)\
    .str.replace('\\', "", regex=True).str.replace(' ', "", regex=True)
temp["fecha"] = temp["fecha"].apply(lambda x: datetime.strptime(x, "%d/%m/%Y"))
temp["empleado"] = temp["empleado"].str.replace(
    '"empleado":"', "", regex=True).str.replace('"', "", regex=True)
temp["observacion"] = temp["observacion"].str.replace(
    '"observacion":"', "", regex=True).str.replace('"', "", regex=True)
# Descartamos observaciones ultimo año posteriores a la ultima solicitud y los atributos empleado-fecha
observaciones = temp[temp["fecha"] > ultima_fecha_solicitud -
                     dt.timedelta(dias_analisis_observaciones)][["CodigoPrograma", "observacion"]]
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
# Para futuro análisis de texto, se concatena con observaciones
recaudos = recaudos[["CodigoPrograma", "mensaje"]]
# Análisis de textos en observaciones
observaciones["observacion"] = observaciones["observacion"].apply(
    limpiar_texto)
# Falla constantemente la API para traducir los textos
analizador = SentimentIntensityAnalyzer()
temp = observaciones["observacion"].apply(
    lambda x: analizador.polarity_scores(x))
resultado = pd.concat([observaciones, temp.apply(pd.Series)], axis=1)

# Agrupar por nb_words y realizar operaciones de suma y promedio en las columnas neg, neu, pos, compound
temp = resultado.groupby("CodigoPrograma").agg({"neg": ["sum", "mean"], "neu": [
    "sum", "mean"], "pos": ["sum", "mean"], "compound": ["sum", "mean"]})
# Resetear el índice del DataFrame
temp = temp.reset_index()
temp.columns = generate_column_names(temp.columns)

# Aplicar la función al DataFrame y crear una nueva columna "sentimiento"
temp["sentimiento"] = temp.apply(
    interpretar_sentimiento, axis=1)
temp["CodigoPrograma"] = temp["_CodigoPrograma"]
data = data.merge(temp[["CodigoPrograma", "sentimiento"]],
                  how="left", on="CodigoPrograma")
data["sentimiento"] = data["sentimiento"].fillna(0)

"""El nivel socioeconomico, variable significativa, se extrae de la latitud y longitud, sin embargo,
más del 75% de los datos no presentan estos datos, para complementarlos
se puede usar la dirección y la localidad de la venta pero se debe pagar
por el servicio ArcGis o uno similar, este presupuesto esta fuera del alcance del proyecto,
y no obtamos por perder esa cantidad de datos, se procede a eliminar las variables, solo usaremos la localidad"""
data = data.drop(["Direccion", "longitud", "latitud"], axis=1).rename(
    columns={"LocalidadVenta": "localidad"})
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
data["localidad"] = data["localidad"].apply(limpiar_texto)
data["profesion_tomador"] = data["profesion_tomador"].apply(
    limpiar_texto).replace(" ", "", regex=True).replace("identificada", "noIdentificado", regex=True).replace("", "noIdentificada")
data["nom_plan"] = data["nom_plan"].apply(
    limpiar_texto).replace(" ", "", regex=True).replace("identificada", "noIdentificado", regex=True)
data = data.rename(columns={"oficios varios": "oficiosVarios", "profesion_tomador": "profesionTomador",
                            "nom_plan": "nombrePlan", "prom_edad_insc": "promedioEdadInscritos"})
data["codigoPrograma"] = data["CodigoPrograma"]
# tenemos el id del tomador
data = data.drop("CodigoPrograma", "tomador", axis=1)
data = data.set_index("codigoPrograma")
# Hay valor de cuota con simbolo negativo
data = data[(data["valorUltimaCuota"] > 0) &
            (data["valorUltimaCuota"] < 1200000)]
# Existen más de 1 esposa y más de una madre
data = data[(data["madre"] < 2) & (data["esposa"] < 2) & (data["esposo"] < 2)]
# Reducir variables categoricas
variables_categoricas = ["qPersonas", "qMascotas",
                         "localidad", "inactivoadmin", "inactivocosto", "inactivofallecimiento",
                         "inactivoinactivo", "inactivoincumplimiento", "inactivoinfluencia", "inactivomala", "inactivorecuperado",
                         "inactivoubicacion", "inactivovoluntario", "sentimiento", "nombrePlan", "profesionTomador", "agricultor",
                         "comerciante", "docente", "empleado", "estudiante", "hogar", 'independiente', 'oficiosVarios', 'pensionado', 'esposa', 'esposo',
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

data.to_csv("../data/outputs/desercion_version_1.csv")
