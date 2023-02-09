from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
from copy import deepcopy
import yaml
from yaml.loader import SafeLoader
import os
os.chdir(r'C:\Users\cason\repos\book-of-mormon-authorship\book_of_mormon\experiments')


class DateExperiment:
    def __init__(self, params, **kwargs):
        self.params = params
        self.random_state = 42
        self.train_size = .8
        
        # Read in Book of Mormon data
        self.results = pd.DataFrame(columns=['feature', 'model', 'MSE', 'R2', 'important_vars', 'importance'])
        self.bom_df = pd.read_pickle('../data/bom_verses.pkl')

        # Get train and validation indices
        bom_df_train, bom_df_val = train_test_split(
            self.bom_df, 
            random_state=self.random_state, 
            train_size=self.train_size
        )
        self.train_index = bom_df_train.index
        self.val_index = bom_df_val.index

        # Define models
        self.models = {
            'regression': ElasticNet(),
            'random-forest': RandomForestRegressor(),
            'xgboost': XGBRegressor(),

        }
        self.__dict__.update(kwargs)

    def train_and_validate(self, model_name, feature_set, feature_names, 
                           X_train, Y_train, X_val, Y_val, num_important=20):
        print(f"\n\nTraining a {model_name} model on {feature_set}")

        grid_cv_model = GridSearchCV(
            estimator=deepcopy(self.models[model_name]),
            param_grid=self.params['experiments']['date_prediction'][model_name]['params'],
            n_jobs=-1,
            verbose=3,
            cv=4,
            scoring='neg_mean_squared_error'
        )

        grid_cv_model.fit(X=X_train, y=Y_train)

        Yhat_val = grid_cv_model.predict(X_val)
        
        results = {
            'feature': feature_set,
            'model': model_name,
            'MSE': mean_squared_error(Y_val, Yhat_val),
            'R2': r2_score(Y_val, Yhat_val)
        }
        
        best_model = grid_cv_model.best_estimator_
        coeff_name = self.params['experiments']['date_prediction'][model_name]['coefs']

        coefs_df = pd.DataFrame({
            'var': feature_names,
            'coef': getattr(best_model, coeff_name)
        })

        important_variables = (
            coefs_df
            .assign(coef_abs = lambda x: np.abs(x['coef']))
            .sort_values('coef_abs', ascending=False)
            .reset_index(drop=True)
        )

        results['important_vars'] = important_variables.loc[:num_important, 'var'].tolist()
        results['importance'] = important_variables.loc[:num_important, 'coef'].tolist()

        return results


    def run(self, path=None):
        for feature_set in self.params['data'].keys():

            scaler = StandardScaler()
            features_df = pd.read_pickle(f'../data/{feature_set}.pkl')
            feature_names = features_df.columns

            X_train = scaler.fit_transform(features_df.loc[self.train_index, :])
            Y_train = self.bom_df.loc[self.train_index, 'date_num']

            X_val = scaler.transform(features_df.loc[self.val_index, :])
            Y_val = self.bom_df.loc[self.val_index, 'date_num']

            for model_name in self.models.keys():
                model_results = self.train_and_validate(
                    model_name, 
                    feature_set, feature_names,
                    X_train, Y_train, X_val, Y_val
                )
                self.results = self.results.append(model_results, ignore_index=True)

        if path is not None:
            self.results.to_pickle(path)
        return self.results


if __name__ == "__main__":
    # Open the file and load the file
    with open('../params.yml') as f:
        params = yaml.load(f, Loader=SafeLoader)

    experiment = DateExperiment(params)
    results = experiment.run("date_results.pkl")
    print(results)