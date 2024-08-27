import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay
from src.models.hyper_parameters import all_models
import joblib
from sklearn.utils import shuffle

#Definimos la función para arreglar el sobremuestreo
def upsample(features, target, repeat):
    # Primero dividimos el conjunto de datos de entrenamiento en positivos y negativos
    seed =12345
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    # Posteriormente multiplicamos los datos de la clase que tiene menos datos, en este caso la clase 1 y unimos todos los datos
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    # Por último, mesclamos todos los datos con la función shuffle y devolvemos los datos desbalanceados
    features_upsampled, target_upsampled = shuffle(
       features_upsampled, target_upsampled, random_state=seed
    )
    return features_upsampled, target_upsampled

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

def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''
    
    models = all_models() 
    
    output_path = './files/modeling_output/model_fit/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    results = []
    # Iterating the models
    for model in models:
        best_estimator, best_score, acc_val,f1_val,roc_auc_val= model_structure(data, model[1], model[2])[0]
        random_predict = model_structure(data, model[1], model[2])[1]
        target_valid = model_structure(data, model[1], model[2])[2]
        results.append([model[0],best_estimator,best_score, acc_val,f1_val,roc_auc_val])
        # Confusion matrix
        cm = confusion_matrix(target_valid,random_predict)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
        fig, ax = plt.subplots()
        cm_display.plot(ax=ax)
        ax.set_title('Confusion_matrix')
        fig.savefig(f'./files/modeling_output/figures/fig_{model[0]}')
        
        # Guardamos el modelo
        joblib.dump(best_estimator,output_path +f'best_random_{model[0]}.joblib')
    results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','acc_val','f1_val','roc_auc_val'])
    results_df.to_csv('./files/modeling_output/reports/model_report.csv',index=False)
    
    results_upsampled = []
    for model in models:
        best_estimator, best_score, acc_val,f1_val,roc_auc_val= model_structure_up(data, model[1], model[2])[0]
        random_predict = model_structure_up(data, model[1], model[2])[1]
        target_valid = model_structure_up(data, model[1], model[2])[2]
        results_upsampled.append([model[0],best_estimator,best_score, acc_val,f1_val,roc_auc_val])
        # Confusion matrix
        cm = confusion_matrix(target_valid,random_predict)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
        fig, ax = plt.subplots()
        cm_display.plot(ax=ax)
        ax.set_title('Confusion_matrix')
        fig.savefig(f'./files/modeling_output/figures/fig_up_{model[0]}')
        # Guardamos el modelo
        joblib.dump(best_estimator,output_path +f'best_random_up_{model[0]}.joblib')
    results_df_up = pd.DataFrame(results_upsampled, columns=['model','best_estimator','best_train_score','acc_val','f1_val','roc_auc_val'])
    results_df_up.to_csv('./files/modeling_output/reports/model_report_up.csv',index=False)
    
    results_downsampled = []
    for model in models:
        best_estimator, best_score, acc_val,f1_val,roc_auc_val= model_structure_up_down(data, model[1], model[2])[0]
        random_predict = model_structure_up_down(data, model[1], model[2])[1]
        target_valid = model_structure_up_down(data, model[1], model[2])[2]
        results_downsampled.append([model[0],best_estimator,best_score, acc_val,f1_val,roc_auc_val])
        # Confusion matrix
        cm = confusion_matrix(target_valid,random_predict)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
        fig, ax = plt.subplots()
        # Dibujar la matriz de confusión en el eje
        cm_display.plot(ax=ax)
        ax.set_title('Confusion_matrix')
        fig.savefig(f'./files/modeling_output/figures/fig_up_{model[0]}.png')
        # Guardamos el modelo
        joblib.dump(best_estimator,output_path +f'best_random_down_{model[0]}.joblib')
    results_df_up_down = pd.DataFrame(results_downsampled, columns=['model','best_estimator','best_train_score','acc_val','f1_val','roc_auc_val'])
    results_df_up_down.to_csv('./files/modeling_output/reports/model_report_up_down.csv',index=False)
    
    return results_df,results_df_up,results_df_up_down


def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    features_train = data[0]
    features_valid = data[1]
    target_train = data[2]
    target_valid = data[3]
    
    #Gráficamos las frecuencias relativas de cada clase
    fig,ax=plt.subplots()
    balance=target_train.value_counts(normalize=True)
    ax.bar(balance.index.astype(str),balance)
    ax.set_title('Balance of clases')
    fig.savefig('./files/modeling_output/figures/fig_balance.png')
    
    # Training the model
    gs = RandomizedSearchCV(pipeline, param_grid, cv=2, scoring='f1', n_jobs=-1, verbose=2)
    gs.fit(features_train,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    best_prediction = best_estimator.predict(features_valid)
    acc_val,f1_val,roc_auc_val = eval_model(best_estimator,features_valid,target_valid)
    
    results = best_estimator, best_score,acc_val,f1_val,roc_auc_val
    return results, best_prediction,target_valid

def model_structure_up(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    features_train = data[0]
    features_valid = data[1]
    target_train = data[2]
    target_valid = data[3]
    
    features_upsampled_train, target_upsampled_train = upsample(
    features_train, target_train, 3
    )
    features_upsampled_valid, target_upsampled_valid = upsample(
        features_valid, target_valid, 3
    )
    
    #Gráficamos las frecuencias relativas de cada clase
    fig,ax=plt.subplots()
    balance=target_upsampled_valid.value_counts(normalize=True)
    ax.bar(balance.index.astype(str),balance)
    ax.set_title('Balance of clases')
    fig.savefig('./files/modeling_output/figures/fig_up_balance.png')
    # Training the model
    gs = RandomizedSearchCV(pipeline, param_grid, cv=2, scoring='f1', n_jobs=-1, verbose=2)
    gs.fit(features_upsampled_train,target_upsampled_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    best_prediction = best_estimator.predict(features_upsampled_valid)
    acc_val,f1_val,roc_auc_val = eval_model(best_estimator,features_upsampled_valid,target_upsampled_valid)
    
    results = best_estimator, best_score,acc_val,f1_val,roc_auc_val
    return results, best_prediction,target_upsampled_valid

def model_structure_up_down(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    features_train = data[0]
    features_valid = data[1]
    target_train = data[2]
    target_valid = data[3]
    
    features_upsampled_train, target_upsampled_train = upsample(
    features_train, target_train, 3
    )
    features_upsampled_valid, target_upsampled_valid = upsample(
        features_valid, target_valid, 3
    )
    features_downsampled_train, target_downsampled_train = downsample(
    features_upsampled_train, target_upsampled_train, 0.75
    )
    features_downsampled_valid, target_downsampled_valid = downsample(
        features_upsampled_valid, target_upsampled_valid, 0.75
    )
    
    #Gráficamos las frecuencias relativas de cada clase
    fig,ax=plt.subplots()
    balance=target_downsampled_valid.value_counts(normalize=True)
    ax.bar(balance.index.astype(str),balance)
    ax.set_title('Balance of clases')
    fig.savefig('./files/modeling_output/figures/fig_up_down_balance.png')
    
    # Training the model
    gs = RandomizedSearchCV(pipeline, param_grid, cv=2, scoring='f1', n_jobs=-1, verbose=2)
    gs.fit(features_downsampled_train,target_downsampled_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    best_prediction = best_estimator.predict(features_downsampled_valid)
    acc_val,f1_val,roc_auc_val = eval_model(best_estimator,features_downsampled_valid,target_downsampled_valid)
    
    results = best_estimator, best_score,acc_val,f1_val,roc_auc_val
    return results, best_prediction,target_downsampled_valid

    
def eval_model(best,features_valid,target_valid):
    random_prediction = best.predict(features_valid)
    random_prob=best.predict_proba(features_valid)[:, 1]
    accuracy_val=accuracy_score(target_valid,random_prediction)
    f1_val=f1_score(target_valid,random_prediction)
    roc_auc_val= roc_auc_score(target_valid,random_prob)
    return accuracy_val,f1_val,roc_auc_val
    