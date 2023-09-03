from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import math
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# Manipulación datos

# Balanceo
# Modelamiento


data = pd.read_csv("../data/outputs/desercion_version_1.csv")
variables_modelo = ['qPersonas', 'qMascotas', 'localidad', 'nombrePlan', 'agricultor',
                    'comerciante', 'docente', 'empleado', 'estudiante', 'hogar',
                    'independiente', 'oficiosVarios', 'pensionado', 'esposa', 'esposo',
                    'hermana', 'hermano', 'hija', 'hijo', 'madre', 'edad', 'diaPagoCuota',
                    'duracion', 'promPercDesc', 'gestionExitosa', 'gestionNoEjecutada',
                    'promedioEdadInscritos', ]
variables_categoricas = ["qPersonas", "qMascotas",
                         "localidad", "inactivoadmin", "inactivocosto", "inactivofallecimiento",
                         "inactivoinactivo", "inactivoincumplimiento", "inactivoinfluencia", "inactivomala", "inactivorecuperado",
                         "inactivoubicacion", "inactivovoluntario", "sentimiento", "nombrePlan", "profesionTomador", "agricultor",
                         "comerciante", "docente", "empleado", "estudiante", "hogar", 'independiente', 'oficiosVarios', 'pensionado', 'esposa', 'esposo',
                         'hermana', 'hermano', 'hija', 'hijo', 'madre']
data_model = data.copy()[variables_modelo+['estado']]
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
    "Inactivo", 1).replace("Activo", 0)
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
descertores = xgb_model.predict(data_model.drop("estado", axis=1))
data["results"] = descertores
print("Los programas propensos a desertar son:")
codigodescertores = data[(data["results"] == 1) & (
    data["estado"] == "Activo")][["codigoPrograma"]].values
print()
# Especifica la ruta y el nombre del archivo
# Obtener la fecha y hora actual
fecha_actual = datetime.now().date()
ruta_archivo = f'../data/outputs/desertores{fecha_actual}.txt'

# Abre el archivo en modo escritura
recall = round(recall, 2)*100
with open(ruta_archivo, "w") as archivo:
    archivo.write(
        f'Los programas propensos a desertar son {codigodescertores} con una tasa de {recall}')

print("Texto exportado correctamente al archivo:", ruta_archivo)
