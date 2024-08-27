from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

## Logistic Regression Model
def all_models():
    '''This function will host all the model parameters, can be used to iterate the
    grid search '''

    seed = 12345
    lr_pipeline = Pipeline([
        ('scaler',StandardScaler()),
        ('lr', LogisticRegression(random_state=seed))
    ])

    lr_param_grid = {
    'lr__max_iter': range(100, 500),
    'lr__solver' : ['lbfgs', 'newton-cg', 'liblinear'],
    'lr__warm_start' : [True, False],
    'lr__C': np.arange(0.01, 1, 0.01)
    }
    
    
    lr = ['Logreg',lr_pipeline,lr_param_grid]

    
    rf_pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('rf', RandomForestClassifier(random_state=seed))])

    # Creación de la malla aleatoria
    rf_param_grid = {'rf__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                'rf__max_features': np.arange(1, 11),
                'rf__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                'rf__min_samples_split': [2, 5, 10],
                'rf__min_samples_leaf': [1, 2, 4],
                'rf__bootstrap': [True, False]}

    # Evaluar el modelo con la función model_evaluation
    rf = ['Random_Forest',rf_pipeline,rf_param_grid]
    
    
    dt_pipeline=Pipeline([
    ('scaler',StandardScaler()),
    ('dt',DecisionTreeClassifier(random_state=seed))])
    
    dt_params={"dt__max_depth": [3, None],
            "dt__max_features": np.arange(1, 10),
            "dt__min_samples_leaf": np.arange(1, 10),
            "dt__criterion": ["gini", "entropy"]}
    
    dt=['dt',dt_pipeline,dt_params]
    
    models = [rf,dt,lr] #Activate to run all the models
    return models
