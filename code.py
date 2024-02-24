# %% [markdown]
# - HasCrCard: el cliente tiene una tarjeta de crédito (1 - sí; 0 - no)
# - EstimatedSalary: salario estimado
# 
# ## Objetivo
# 
# - Exited: El cliente se ha ido (1 - sí; 0 - no)
# # Descripción del proyecto
# 
# Los clientes de Beta Bank se están yendo, cada mes, poco a poco. Los banqueros descubrieron que es más barato salvar a los clientes existentes que atraer nuevos.
# 
# Necesitamos predecir si un cliente dejará el banco pronto. Tú tienes los datos sobre el comportamiento pasado de los clientes y la terminación de contratos con el banco.
# 
# Crea un modelo con el máximo valor F1 posible. Para aprobar la revisión, necesitas un valor F1 de al menos 0.59. Verifica F1 para el conjunto de prueba. 
# 
# Además, debes medir la métrica AUC-ROC y compararla con el valor F1.
# 
# # Instrucciones del proyecto
# 
# 1. Descarga y prepara los datos.Explica el procedimiento.
# 2. Examina el equilibrio de clases. Entrena el modelo sin tener en cuenta el desequilibrio. - Describe brevemente tus hallazgos.
# 3. Mejora la calidad del modelo. Asegúrate de utilizar al menos dos enfoques para corregir el desequilibrio de clases. Utiliza conjuntos de entrenamiento y validación para encontrar el mejor modelo y el mejor conjunto de parámetros. Entrena diferentes modelos en los conjuntos de entrenamiento y validación. Encuentra el mejor. Describe brevemente tus hallazgos.
# 4. Realiza la prueba final.
# 
# # Descripción de los datos
# 
# Puedes encontrar los datos en el archivo  /datasets/Churn.csv file. Descarga el conjunto de datos.
# 
# ## Características
# 
# - RowNumber: índice de cadena de datos
# - CustomerId: identificador de cliente único
# - Surname: apellido
# - CreditScore: valor de crédito
# - Geography: país de residencia
# - Gender: sexo
# - Age: edad
# - Tenure: período durante el cual ha madurado el depósito a plazo fijo de un cliente (años)
# - Balance: saldo de la cuenta
# - NumOfProducts: número de productos bancarios utilizados por el cliente
# - IsActiveMember: actividad del cliente (1 - sí; 0 - no)

# %% [markdown]
# ## Librerías

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from pprint import pprint
from scipy.stats import randint
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

# %% [markdown]
# ## Carga de datos

# %%
data=pd.read_csv('datasets/Churn.csv')
data.head(5)

# %%
data.info()

# %%
#Pasamos las columnas al modo snake_case
data.columns=['Row_Number', 'Customer_Id', 'Surname', 'Credit_Score', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'Num_Of_Products', 'Has_Cr_Card',
       'Is_Active_Member', 'Estimated_Salary', 'Exited']
new_col=[]
for i in data.columns:
    x=i.lower()
    new_col.append(x)
data.columns=new_col


# %%
data.describe()

# %%
#Cambiamos el tipo de dato a string
data['row_number']=data['row_number'].astype(str)
data['customer_id']=data['customer_id'].astype(str)
data['balance']=data['balance'].astype(int)
data['estimated_salary']=data['estimated_salary'].astype(int)
data.info()

# %% [markdown]
# Podemos ver a simple vista con el data info, que nuestros datos están bien, y que solo tenemos ausentes en la columna 'tenure', sin embargo, veremos más a fondo los ausetes y los duplicados.
# 
# En cuanto a los tipos de datos vemos que las columnas 'RowNumber' y 'CustomerId' deberían ser tipo string para no cometer errores con estos números que no son significativos, sin embargo, debido a que el objetivo del proyecto es predecir los clientes que se van del banco, las tres primeras columnas no son relevantes para entrenar el modelo, por lo que no se tendrán en cuenta.

# %% [markdown]
# ## Limpieza de datos

# %%
#Calculamos los ausentes
print('Ausentes: \n',data.isna().sum())

# %%
#Calculamos el porcentaje de significancia de los ausentes
print('Porcentaje de significancia: \n',100*data.isna().sum()/data.shape[0])

# %% [markdown]
# Podeos ver que solo tenure tiene 909 ausentes el cual equivale al 9.09% de los datos, podemos trabajar con estos para imputarlos.

# %%
data[data['tenure'].isna()]

# %%
#Llenamos los ausentes con la mediana
data['tenure'].fillna(data['tenure'].median(),inplace=True)
print(100*data.isna().sum()/data.shape[0])


# %% [markdown]
# Llenamos los datos ausentes con la mediana.

# %%
print('Duplicados: \n',data.duplicated().sum())

# %% [markdown]
# No tenemos duplicados, y rellenamos los ausentes, por lo cual estamos listos para comenzar el modelo de clasificación.

# %% [markdown]
# ## Entrenamiento

# %% [markdown]
# ### Examinamos el balanceo

# %%
#Gráficamos las frecuencias relativas de cada clase
balance=data['exited'].value_counts(normalize=True)
plt=balance.plot(kind='bar')

# %% [markdown]
# Podemos ver que la clase "0" es predominante con un porcentaje aproximado del 80%, en cambio la clase "1" tiene solo el 20%

# %% [markdown]
# ### Escalado de caracteristicas

# %%
#Vamos a escalar las características para que nuestro modelo pueda tomar estas variables
numeric=['credit_score','age','tenure','balance','num_of_products','has_cr_card','is_active_member','estimated_salary']
scaler=StandardScaler()
scaler.fit(data[numeric])
data[numeric]=scaler.transform(data[numeric])

# %%
#Mostramos nuestro dataset escalado
data.head(5)

# %% [markdown]
# ### Codificación de las variables categoricas para los arboles
# 
# Vamos a codificar con etiquetas el dataframe para los arboles y utilizaremos ONE-HOT para la regresion logística.

# %%
#Usamos codificación de etiquetas para el modelo, debido a que usaremos un arbol de clasificación
data_model=data.drop(['customer_id','row_number','surname'],axis=1)
encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data_model), columns=data_model.columns)
data_ordinal.head(5)

# %%
#Codificamos los datos en one-hot para la regresion logistica
data_oh=pd.get_dummies(data_model,drop_first=True)
data_oh.head(5)

# %% [markdown]
# ### Separamos el dataframe en entrenamiento y testeo

# %%
seed=12345
df_train_ordinal,df_test_ordinal=train_test_split(data_ordinal,random_state=seed,test_size=0.2)

# %%
df_train_oh,df_test_oh=train_test_split(data_oh,random_state=seed,test_size=0.2)

# %% [markdown]
# ### Separamos el dataset en entrenamiento y validación

# %%
#Separamos los datos para los arboles
seed=12345
features= df_train_ordinal.drop(['exited'],axis=1)
target=df_train_ordinal['exited']

features_train_ordinal, features_valid_ordinal, target_train_ordinal, target_valid_ordinal=train_test_split(
    features,target,random_state=seed,test_size=0.2)

model=RandomForestClassifier()

# %%
#Separamos los datos para la regresion logística
seed=12345
features= df_train_oh.drop(['exited'],axis=1)
target=df_train_oh['exited']

features_train_oh, features_valid_oh, target_train_oh, target_valid_oh=train_test_split(
    features,target,random_state=seed,test_size=0.25)

model=RandomForestClassifier(random_state=seed)

# %% [markdown]
# ### Random Forest

# %%
#Ahora vamos a pasar la lista de parametros que queremos iterar:
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Número de features a considerar para cada separación
max_features = randint(1, 11)
# Máximo número de niveles a considerar en el arbol
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Número mínimo  de pruebas requeridas para las eparación de un nodo
min_samples_split = [2, 5, 10]
# Número minimo de pruebas requeridas para cada nodo hoja
min_samples_leaf = [1, 2, 4]
# Metodo de selección de pruebas para el entrenamiento de cada árbol
bootstrap = [True, False]
# Creación de la malla aleatoria
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# %%
#Ahora junto con la malla y el RandomizedCV vamos a generar el mejor modelo con los mejores hiperparametros 
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=seed, n_jobs = -1)
# Entrenamos el modelo
model_random.fit(features_train_ordinal,target_train_ordinal)
print(model_random.best_params_)

# %%
#Medimos la exactitud de nuestro modelo 
best_random = model_random.best_estimator_
random_prediction = best_random.predict(features_valid_ordinal)
random_accuracy=metrics.accuracy_score(target_valid_ordinal,random_prediction)
roc_auc_rf=roc_auc_score(target_valid_ordinal,random_prediction)
print("Accuracy:",random_accuracy)
print('ROC-AUC: ',roc_auc_rf)

# %% [markdown]
# Tenemos una buena exactitud aunque con datos desbalanceados, por lo que debemos balancearlos. Por otro lado, el area bajo la curva ROC nos indica que el modelo no es malo pero puede mejorar.

# %%
#Mostramos la matriz de confusión y el reporte con nuestras métricas de clasificación
print(classification_report(target_valid_ordinal,random_prediction))
confusion_matrix = metrics.confusion_matrix(target_valid_ordinal,random_prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()


# %% [markdown]
# Podemos ver en la matriz que los datos están desbalanceados, debido a que el recall de la clase 1 es de 0.42 y el f1 es 0.56 lo que indica que del total de la suma de verdaderos positivos y falsos negativos, solo se está acertanco al 42% de los datos con categoría 1, por lo cual debemos hacer labores de balanceo para mejorar este resultado.

# %% [markdown]
# ### Decision Tree

# %%
#Parametros del arbol de decisión
params = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
tree=DecisionTreeClassifier(random_state=seed) 

# %%
#Iteramos para hallar los mejores hiperparametros para el arbol de decisión
random_tree=RandomizedSearchCV(estimator = tree, param_distributions = params, n_iter = 100, cv = 3, verbose=2, random_state=seed, n_jobs = -1)
# Entrenamos el modelo
random_tree.fit(features_train_ordinal,target_train_ordinal)
print(random_tree.best_params_)

# %%
#Medimos la exactitud del modelo
best_tree = random_tree.best_estimator_
tree_prediction = best_tree.predict(features_valid_ordinal)
tree_accuracy=metrics.accuracy_score(target_valid_ordinal,tree_prediction)
roc_auc_dt=roc_auc_score(target_valid_ordinal,tree_prediction)
print("Accuracy:",tree_accuracy)
print('ROC-AUC: ',roc_auc_dt)

# %% [markdown]
# Tenemos una buena exactitud aunque con datos desbalanceados al igual que el Random Forest, por lo que debemos balancearlos. Por otro lado, el area bajo la curva ROC da 0.69, lo que nos indica que el modelo no es malo pero puede mejorar.

# %%
#Mostramos la matriz de confusión y el reporte de métricas de clasificación
print(classification_report(target_valid_ordinal,tree_prediction))
confusion_matrix = metrics.confusion_matrix(target_valid_ordinal,tree_prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Al igual que el bosque aleatorio, podemos ver que el desbalanceo afecta el recall y el f1 score de la clase 1, con valores de 0.47 y 0.53 respectivamente, incluso en este caso la precisión se vio afectada al bajar a un 62%. También debemos trabajar con estos datos para contraarrestar el balanceo.

# %% [markdown]
# ### Logistic Regression

# %%
#Parametros de iteración de la regresión logística
params={
    'max_iter': range(100, 500),
    'solver' : ['lbfgs', 'newton-cg', 'liblinear'],
    'warm_start' : [True, False],
    'C': np.arange(0.01, 1, 0.01)
}
log_reg=LogisticRegression(random_state=seed)

# %%
#Entrenamos el modelo
log_random=RandomizedSearchCV(estimator=log_reg,param_distributions=params,n_iter=100,cv = 3, verbose=2, random_state=seed, n_jobs = -1)
log_random.fit(features_train_oh,target_train_oh)
print(log_random.best_params_)


# %%
#Medimos la exactitud del modelo
best_log = log_random.best_estimator_
log_prediction = best_log.predict(features_valid_oh)
log_accuracy=metrics.accuracy_score(target_valid_oh,log_prediction)
roc_auc_lr=roc_auc_score(target_valid_oh,log_prediction)
print("Accuracy:",log_accuracy)
print('ROC-AUC: ',roc_auc_lr)

# %% [markdown]
# La exactitud de este modelo es la más baja de los tres modelos, al igual que el área bajo la curva, sin embargo, podemos mejorar este valor.

# %%
#Mostramos la matriz de confusión y el reporte de métricas de clasificación
print(classification_report(target_valid_oh,log_prediction))
confusion_matrix = metrics.confusion_matrix(target_valid_oh,log_prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Al análizar los resultados de la regresión logística, podemos ver que también es afectada por el desbalanceo, al tener un recall para la clase 1 de 0.17 y un recall de 0.27, lo que indica que tenemos que trabajar con el desbalanceo para lograr subir estas métricas.

# %% [markdown]
# ## Mejora del modelo

# %% [markdown]
# ### Sobremuestreo

# %%
#Definimos la función para arreglar el sobremuestreo
def upsample(features, target, repeat):
    #Primero dividimos el conjunto de datos de entrenamiento en positivos y negativos 
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    #Posteriormente multiplicamos los datos de la clase que tiene menos datos, en este caso la clase 1 y unimos todos los datos
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    #Por último, mesclamos todos los datos con la función shuffle y devolvemos los datos desbalanceados
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=seed
    )

    return features_upsampled, target_upsampled

# %%
#Desbalanceamos todos los conjuntos de datos
features_upsampled_ordinal_train, target_upsampled_ordinal_train = upsample(
    features_train_ordinal, target_train_ordinal, 3
)
features_upsampled_ordinal_valid, target_upsampled_ordinal_valid = upsample(
    features_valid_ordinal, target_valid_ordinal, 3
)
features_upsampled_oh_train, target_upsampled_oh_train = upsample(
    features_train_oh, target_train_oh, 3
)
features_upsampled_oh_valid, target_upsampled_oh_valid = upsample(
    features_valid_oh, target_valid_oh, 3
)

# %%
#Gráficamos las frecuencias relativas de cada clase
balance=target_upsampled_ordinal_train.value_counts(normalize=True)
plt=balance.plot(kind='bar')

# %%
balance=target_upsampled_ordinal_valid.value_counts(normalize=True)
plt=balance.plot(kind='bar')

# %% [markdown]
# ### Random Forest con sobremuestreo

# %%
# Entrenamos el modelo de nuevo con sobremuestreo
model_random.fit(features_upsampled_ordinal_train,target_upsampled_ordinal_train)
print(model_random.best_params_)

# %%
#Medimos la exactitud de nuestro modelo 
best_random = model_random.best_estimator_
random_prediction_balanced = best_random.predict(features_upsampled_ordinal_valid)
random_accuracy_balanced=metrics.accuracy_score(target_upsampled_ordinal_valid,random_prediction_balanced)
roc_auc_rfb=roc_auc_score(target_upsampled_ordinal_valid,random_prediction_balanced)
print("Accuracy:",random_accuracy_balanced)
print('ROC-AUC: ',roc_auc_rfb)
print('Diferencia accuracy: ',random_accuracy-random_accuracy_balanced)
print('Diferencia ROC-AUC: ',roc_auc_rf-roc_auc_rfb)


# %% [markdown]
# Podemos ver que al sobreajustar los datos se disminuye la exactitud en un 10% sin embargo, la calidad del modelo aumento en un 1.3% como lo vemos en la métrica ROC-AUC.

# %%
#Mostramos la matriz de confusión y el reporte con nuestras métricas de clasificación
print(classification_report(target_upsampled_ordinal_valid,random_prediction_balanced))
confusion_matrix = metrics.confusion_matrix(target_upsampled_ordinal_valid,random_prediction_balanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Podemos ver que al triplicar los datos de la clase uno, nuestro dataset se balanceo y obtuvimos un mejor recall y f1 score sobre esa clase, obteniendo 0.48 y 0.62 respectivamente, sacrificando la exactitud que ahora es del 76%. 

# %% [markdown]
# ### Decision Tree con sobremuestreo

# %%
random_tree.fit(features_upsampled_ordinal_train,target_upsampled_ordinal_train)
print(random_tree.best_params_)

# %%
#Medimos la exactitud del modelo
best_tree_balanced = random_tree.best_estimator_
tree_prediction_balanced = best_tree_balanced.predict(features_upsampled_ordinal_valid)
tree_accuracy_balanced=metrics.accuracy_score(target_upsampled_ordinal_valid,tree_prediction_balanced)
roc_auc_dtb=roc_auc_score(target_upsampled_ordinal_valid,tree_prediction_balanced)
print("Accuracy:",tree_accuracy_balanced)
print('ROC-AUC: ',roc_auc_dtb)
print('Diferencia accuracy: ',tree_accuracy-tree_accuracy_balanced)
print('Diferencia ROC-AUC: ',roc_auc_dt-roc_auc_dtb)

# %% [markdown]
# Al sobreajustar el arbol de decisión, podemos ver que la exactitud disminuyó en un 13% y el ROC-AUC bajo un 1.9%, lo que nos indica que la calidad del modelo disminuyó.

# %%
#Mostramos la matriz de confusión y el reporte de métricas de clasificación
print(classification_report(target_upsampled_ordinal_valid,tree_prediction_balanced))
confusion_matrix = metrics.confusion_matrix(target_upsampled_ordinal_valid,tree_prediction_balanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# En cuanto a nuestras métricas de recall y f1 score, pasamos de un 0.47 y 0.53 respectivamente, a 0.50 y 0.59, lo que nos indica que el sobremuestreo de nuestros datos mejoró estas métricas.

# %% [markdown]
# ### Logistic Regression con sobremuestreo

# %%
log_random.fit(features_upsampled_oh_train,target_upsampled_oh_train)
print(log_random.best_params_)

# %%
#Medimos la exactitud del modelo
best_log_balanced = log_random.best_estimator_
log_prediction_balanced = best_log_balanced.predict(features_upsampled_oh_valid)
log_accuracy_balanced=metrics.accuracy_score(target_upsampled_oh_valid,log_prediction_balanced)
roc_auc_lrb=roc_auc_score(target_upsampled_oh_valid,log_prediction_balanced)
print("Accuracy:",log_accuracy_balanced)
print('ROC-AUC: ',roc_auc_lr)
print('Diferencia accuracy: ',log_accuracy-log_accuracy_balanced)
print('Diferencia ROC-AUC: ',roc_auc_lr-roc_auc_lrb)

# %% [markdown]
# Al ver el cambio de la exactitud, podemos ver que el sobreajuste disminuye en un 11%, sin embargo, la calidad de nuestro modelo aumento con la ROC-AUC en un 11%.

# %%
#Mostramos la matriz de confusión y el reporte de métricas de clasificación
print(classification_report(target_upsampled_oh_valid,log_prediction_balanced))
confusion_matrix = metrics.confusion_matrix(target_upsampled_oh_valid,log_prediction_balanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Al ver nuestras otras métricas, el sobre ajuste cambió el recall y el f1 score respectivamente de 0.17 y 0.27 a 0.58 y 0.62, lo que es un cambio gigantezco en mejora de éstas métricas, lo que nos indica que el sobreajuste mejoro mucho muestro modelo de regresión logística.

# %% [markdown]
# ### Submuestreo

# %%
#Definimos la función para reducir el tamaño de la clase predominante y submuestrear los datos
def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled

# %%
#Desbalanceamos todos los conjuntos de datos
features_downsampled_ordinal_train, target_downsampled_ordinal_train = downsample(
    features_upsampled_ordinal_train, target_upsampled_ordinal_train, 0.75
)
features_downsampled_ordinal_valid, target_downsampled_ordinal_valid = downsample(
    features_upsampled_ordinal_valid, target_upsampled_ordinal_valid, 0.75
)
features_downsampled_oh_train, target_downsampled_oh_train = downsample(
    features_upsampled_oh_train, target_upsampled_oh_train, 0.75
)
features_downsampled_oh_valid, target_downsampled_oh_valid = downsample(
    features_upsampled_oh_valid, target_upsampled_oh_valid, 0.75
)

# %%
#Gráficamos las frecuencias relativas de cada clase
balance=target_downsampled_ordinal_train.value_counts(normalize=True)
plt=balance.plot(kind='bar')

# %%
balance=target_downsampled_ordinal_valid.value_counts(normalize=True)
plt=balance.plot(kind='bar')

# %% [markdown]
# ### Random Forest con submuestreo

# %%
# Entrenamos el modelo de nuevo con sobremuestreo
model_random.fit(features_downsampled_ordinal_train,target_downsampled_ordinal_train)
print(model_random.best_params_)

# %%
#Medimos la exactitud de nuestro modelo 
best_random = model_random.best_estimator_
random_prediction_subbalanced = best_random.predict(features_downsampled_ordinal_valid)
random_accuracy_subbalanced=metrics.accuracy_score(target_downsampled_ordinal_valid,random_prediction_subbalanced)
roc_auc_rfs=roc_auc_score(target_downsampled_ordinal_valid,random_prediction_subbalanced)
print("Accuracy:",random_accuracy_subbalanced)
print('ROC-AUC: ',roc_auc_rfs)
print('Diferencia accuracy: ',random_accuracy_balanced-random_accuracy_subbalanced)
print('Diferencia ROC-AUC: ',roc_auc_rfb-roc_auc_rfs)

# %% [markdown]
# La exactitud del modelo submuestreado al igual que en el sobremuestreo disminuye, teniendo en cuenta el modelos sobremuestreado, nuestra exactitud disminuye de nuevo en un 2.5%, sin embargo, nuestra curva ROC-AUC aumenta su calidad en un 0.9%.

# %%
#Mostramos la matriz de confusión y el reporte con nuestras métricas de clasificación
print(classification_report(target_downsampled_ordinal_valid,random_prediction_subbalanced))
confusion_matrix = metrics.confusion_matrix(target_downsampled_ordinal_valid,random_prediction_subbalanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Nuevamente, aumentamos nuestras métricas de recall y f1 score, de 0.48, 0.62 a 0.52 y 0.66, lo que nos indica que nuestro submuestreo también mejoró estas métricas con respecto a nuestro dataset sobremuestreado.

# %% [markdown]
# ### Decision Tree con submuestreo

# %%
random_tree.fit(features_downsampled_ordinal_train,target_downsampled_ordinal_train)
print(random_tree.best_params_)

# %%
#Medimos la exactitud del modelo
best_tree_subbalanced = random_tree.best_estimator_
tree_prediction_subbalanced = best_tree_subbalanced.predict(features_downsampled_ordinal_valid)
tree_accuracy_subbalanced=metrics.accuracy_score(target_downsampled_ordinal_valid,tree_prediction_subbalanced)
roc_auc_dts=roc_auc_score(target_downsampled_ordinal_valid,tree_prediction_subbalanced)
print("Accuracy:",tree_accuracy_subbalanced)
print('ROC-AUC: ',roc_auc_dts)
print('Diferencia accuracy: ',tree_accuracy_balanced-tree_accuracy_subbalanced)
print('Diferencia ROC-AUC: ',roc_auc_dtb-roc_auc_dts)

# %% [markdown]
# La exactitud aumenta en este caso un porcentaje muy mínimo, y nuestra calidad del modelo aumenta en un 2.7% la curva ROC-AUC, lo que indica que el submuestreo mejoró nuestro modelo con el dataset ya sobremuestreado.

# %%
#Mostramos la matriz de confusión y el reporte de métricas de clasificación
print(classification_report(target_downsampled_ordinal_valid,tree_prediction_subbalanced))
confusion_matrix = metrics.confusion_matrix(target_downsampled_ordinal_valid,tree_prediction_subbalanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Nuesto submuestreo, aumento el recall y el f1 score de nuestro modelo de 0.50 y 0.59 a 0.56 y 0.65, lo que mejora la calidad de nuestro arbol de decición.

# %% [markdown]
# ### Logistic Regression con submuestreo

# %%
log_random.fit(features_downsampled_oh_train,target_downsampled_oh_train)
print(log_random.best_params_)

# %%
#Medimos la exactitud del modelo
best_log_subbalanced = log_random.best_estimator_
log_prediction_subbalanced = best_log_subbalanced.predict(features_downsampled_oh_valid)
log_accuracy_subbalanced=metrics.accuracy_score(target_downsampled_oh_valid,log_prediction_subbalanced)
roc_auc_lrs=roc_auc_score(target_downsampled_oh_valid,log_prediction_subbalanced)
print("Accuracy:",log_accuracy_subbalanced)
print('ROC-AUC: ',roc_auc_lrs)
print('Diferencia accuracy: ',log_accuracy_balanced-log_accuracy_subbalanced)
print('Diferencia ROC-AUC: ',roc_auc_lrb-roc_auc_lrs)

# %% [markdown]
# Aunque la exactitud disminuye un porcentaje muy mínimo, casi descartable, nuestra ROC-AUC, aumenta en un 1.1%, lo que nos muestra que el submuestreo mejoró nuestro modelo ligeramente.

# %%
#Mostramos la matriz de confusión y el reporte de métricas de clasificación
print(classification_report(target_downsampled_oh_valid,log_prediction_subbalanced))
confusion_matrix = metrics.confusion_matrix(target_downsampled_oh_valid,log_prediction_subbalanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# El recall y el f1 score aumentaron de 0.58 y 0.62 respectivamente a 0.69 y 0.69, lo que nos muestra que el submuestreo mejoró significativamente nuestra regresión logística.

# %% [markdown]
# ## Prueba Final

# %%
features_test_ordinal= df_test_ordinal.drop(['exited'],axis=1)
target_test_ordinal=df_test_ordinal['exited']
features_test_oh= df_test_oh.drop(['exited'],axis=1)
target_test_oh=df_test_oh['exited']

# %%
features_upsampled_ordinal_test, target_upsampled_ordinal_test = upsample(
    features_test_ordinal, target_test_ordinal, 3
)
features_upsampled_oh_test, target_upsampled_oh_test = upsample(
    features_test_oh, target_test_oh, 3
)
features_downsampled_ordinal_test, target_downsampled_ordinal_test = downsample(
    features_upsampled_ordinal_test, target_upsampled_ordinal_test, 0.75
)
features_downsampled_oh_test, target_downsampled_oh_test = downsample(
    features_upsampled_oh_test, target_upsampled_oh_test, 0.75
)

# %%
#Gráficamos las frecuencias relativas de cada clase
balance=target_downsampled_ordinal_test.value_counts(normalize=True)
plt=balance.plot(kind='bar')

# %%
#Gráficamos las frecuencias relativas de cada clase
balance=target_downsampled_oh_test.value_counts(normalize=True)
plt=balance.plot(kind='bar')

# %% [markdown]
# ### Random Forest prueba final

# %%
#Medimos la exactitud de nuestro modelo 
best_random = model_random.best_estimator_
random_prediction_finalbalanced = best_random.predict(features_downsampled_ordinal_test)
random_accuracy_finalbalanced=metrics.accuracy_score(target_downsampled_ordinal_test,random_prediction_finalbalanced)
print("Accuracy:",random_accuracy_finalbalanced)
print('ROC-AUC: ',roc_auc_score(target_downsampled_ordinal_test,random_prediction_finalbalanced))

# %%
#Mostramos la matriz de confusión y el reporte con nuestras métricas de clasificación
print(classification_report(target_downsampled_ordinal_test,random_prediction_finalbalanced))
confusion_matrix = metrics.confusion_matrix(target_downsampled_ordinal_test,random_prediction_finalbalanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Al probar el mejor random forest y balanceando los datos del grupo test con nuestras dos técnicas de sobremuestreo y submuestreo, obtenemos lo siguiente: La exactitud del modelo fue del 72% y la curva ROC-AUC del 73%, Adicionalmente nuestro recall fue de 0.53 y el f1 score de 0.66. 
# 
# Si hacemos un recuento de estos resultados, el modelo pasa con una exactitud un poco baja debido a las técnicas de balanceo, sin embargo, tiene una mejor calidad que el dataset desbalanceado que era de ROC-AUC 70%. Esto quiere decir que en general nuestro modelo mejoró al aplicar las técnicas de balanceo.

# %% [markdown]
# ### Decision Tree prueba final

# %%
#Medimos la exactitud del modelo
best_tree_finalbalanced = random_tree.best_estimator_
tree_prediction_finalbalanced = best_tree_finalbalanced.predict(features_downsampled_ordinal_test)
tree_accuracy_finalbalanced=metrics.accuracy_score(target_downsampled_ordinal_test,tree_prediction_finalbalanced)
print("Accuracy:",tree_accuracy_finalbalanced)
print('ROC-AUC: ',roc_auc_score(target_downsampled_ordinal_test,tree_prediction_finalbalanced))

# %%
#Mostramos la matriz de confusión y el reporte de métricas de clasificación
print(classification_report(target_downsampled_ordinal_test,tree_prediction_finalbalanced))
confusion_matrix = metrics.confusion_matrix(target_downsampled_ordinal_test,tree_prediction_finalbalanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Al probar el arbol de decisión y balanceando los datos del grupo test con nuestras dos técnicas de sobremuestreo y submuestreo, obtenemos lo siguiente: La exactitud del modelo fue del 69% y la curva ROC-AUC del 69%, Adicionalmente nuestro recall fue de 0.54 y el f1 score de 0.64. 
# 
# Si hacemos un recuento de estos resultados, el modelo pasa con una exactitud un poco baja debido a las técnicas de balanceo, además bajó la calidad con respecto al dataset desbalanceado que era de ROC-AUC 70%. Esto quiere decir que en general nuestro modelo empeoró al aplicar las técnicas de balanceo, quizas porque el modelo es muy sensible al modificar los datos con las técnicas de balanceo. De todas maneras se cumplió dejando el f1 score alto.

# %% [markdown]
# ### Logistic Regression prueba final

# %%
#Medimos la exactitud del modelo
best_log_finalbalanced = log_random.best_estimator_
log_prediction_finalbalanced = best_log_finalbalanced.predict(features_downsampled_oh_test)
log_accuracy_finalbalanced=metrics.accuracy_score(target_downsampled_oh_test,log_prediction_finalbalanced)
print("Accuracy:",log_accuracy_finalbalanced)
print('ROC-AUC: ',roc_auc_score(target_downsampled_oh_test,log_prediction_finalbalanced))

# %%
#Mostramos la matriz de confusión y el reporte de métricas de clasificación
print(classification_report(target_downsampled_oh_test,log_prediction_finalbalanced))
confusion_matrix = metrics.confusion_matrix(target_downsampled_oh_test,log_prediction_finalbalanced)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

plt=cm_display.plot()

# %% [markdown]
# Por último, la regresión logística y balanceando los datos del grupo test con nuestras dos técnicas de sobremuestreo y submuestreo, 0.70 y el f1 score de 0.71. 
# 
# Si hacemos un recuento de estos resultados, el modelo pasa con una exactitud un poco baja debido a las técnicas de balanceo, además bajó la calidad con respecto al dataset desbalanceado que era de ROC-AUC 57%. Esto quiere decir que este modelo fue el más beneficiado por el balanceo, debido a que su calidad aumento mucho.

# %% [markdown]
# ## Conclusiones
# 
# 1. El balanceo mejora la calidad de los modelos de clasificación teniendo en cuenta la curva ROC-AUC, su f1-score y su recall, sin embargo, al aplicar las técicas de sobremuestreo y submuestreo, la exactitud de nuestros modelos baja significativamente.
# 
# 2. El mejor modelo evaluado fue el random forest que tuvo una métrica de ROC-AUC del 72%, seguido de la regresión logística con 70% y el arbol de decisión con un 69%.
# 
# 3. El arbol de decisión fue sensible a las técnicas de balanceo, por lo tanto, se vio afectada su métrica ROC-AUC y empeoró con respecto al dataset sin balancear, aunque sus métricas de recall y f1-score aumentaron.


