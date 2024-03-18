# train_model.py
import pathlib, sys, yaml, joblib, pickle
from datetime import datetime
from hyperopt import hp 
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def find_best_model_with_params(X_train, y_train, X_test, y_test):
    
    hyperparameters = {
        'RandomForestRegressor':{
            'n_estimators': hp.choice('n_estimators', [10, 15, 20]),
            'max_depth': hp.choice('max_depth', [6, 8, 10]),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
        },
        'XGBRegressor':{
            'n_estimators': hp.choice('n_estimators', [10, 15, 20]),
            'max_depth': hp.choice('max_depth', [6, 8, 10]),
            'learning_rate': hp.uniform('learning_rate', 0.03, 0.3)
        }
    }
    
    def evaluate_model(hyperopt_params: dict) -> dict:
        
        params = hyperopt_params
        if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
        if 'min_child_weight' in params: params['min_child_weight'] = int(params['min_child_weight'])
        if 'max_delta_step' in params: params['max_delta_step'] = int(params['max_delta_step'])
        
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        model_rmse = mean_squared_error(y_test, y_pred)
        return {'loss': model_rmse, 'status': STATUS_OK}
    
    space = hyperparameters['XGBRegressor']
    
    time_field_mlflow = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
    
    with mlflow.start_run(run_name=f'XGBRegressor {time_field_mlflow}'):
        argmin = fmin(
            fn=evaluate_model,
            space=space,
            algo=tpe.suggest,
            max_evals=5,
            trials=Trials(),
            verbose=True
        )
    
    run_ids = []
    with mlflow.start_run(run_name=f'XGBRegressor Final {time_field_mlflow}') as run:
        run_id = run.info.run_id
        run_name = run.data.tags['mlflow.runName']
        run_ids += [(run_name, run_id)]
        
        #configure params
        params = space_eval(space=space, hp_assignment=argmin)
        if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
        if 'min_child_weight' in params: params['min_child_weight'] = int(params['min_child_weight'])
        if 'max_delta_step' in params: params['max_delta_step'] = int(params['max_delta_step'])
        mlflow.log_params(params)
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        final_model_rmse = mean_squared_error(y_test, y_pred)
        mlflow.sklearn.log_model(model, 'model')
        mlflow.log_metric('RMSE', final_model_rmse)
    
    return model
                        
        
def train_model(train_features, target, n_estimators, max_depth, seed):
    # Train your machine learning model
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(train_features, target)
    return model

def save_model(model, output_path):
    # Save the trained model to the specified output path
    #joblib.dump(model, output_path + '/model.joblib')
    with open(output_path + '/model.pkl', 'wb') as file: 
        pickle.dump(model, file)

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    '''params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]'''

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    TARGET = 'trip_duration'
    train_features = pd.read_csv(data_path + '/train.csv')
    X = train_features.drop(columns=[TARGET], axis=1)
    y = train_features[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #trained_model = train_model(X, y, params['n_estimators'], params['max_depth'], params['seed'])
    trained_model = find_best_model_with_params(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    save_model(trained_model, output_path)
    

if __name__ == "__main__":
    
    main()