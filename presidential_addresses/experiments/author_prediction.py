from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
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
        self.results = pd.DataFrame(columns=['feature', 'model', 'f1', 'precision', 'recall', 'auc', 'important_vars', 'importance'])
        self.bom_df = pd.read_pickle('../data/bom_verses.pkl')

        # Get train and validation indices
        bom_df_train, bom_df_val = train_test_split(
            self.bom_df, 
            random_state=self.random_state, 
            train_size=self.train_size,
            stratify=self.bom_df['book']
        )
        self.train_index = bom_df_train.index
        self.val_index = bom_df_val.index

        # Define models
        self.models = {
            'logistic-regression': LogisticRegression(),
            'naive-bayes': GaussianNB(),
            'random-forest': RandomForestClassifier(),
            'xgboost': XGBClassifier(),
            'k-nearest-neighbors': KNeighborsClassifier(),
            'linear-discriminant-analysis': LinearDiscriminantAnalysis(),
            'quadratic-discriminant-analysis': QuadraticDiscriminantAnalysis()
        }
        self.__dict__.update(kwargs)

    def train_and_validate(self, model_name, feature_set, feature_names, 
                           X_train, Y_train, X_val, Y_val, num_important=20):
        print(f"\n\nTraining a {model_name} model on {feature_set}")
        
        grid_cv_model = GridSearchCV(
            estimator=deepcopy(self.models[model_name]),
            param_grid=self.params['experiments']['author_prediction'][model_name]['params'],
            n_jobs=-1,
            verbose=3,
            cv=4,
            scoring='roc_auc_ovo_weighted'
        )

        grid_cv_model.fit(X=X_train, y=Y_train)

        Yhat_val = grid_cv_model.predict(X_val)

        results = {
            'feature': feature_set,
            'model': model_name,
            'f1': f1_score(Y_val, Yhat_val), 
            'precision': precision_score(Y_val, Yhat_val), 
            'recall': recall_score(Y_val, Yhat_val), 
            'auc': roc_auc_score(Y_val, Yhat_val)
        }
        
        return results


    def run(self, path=None):
        for feature_set in self.params['data'].keys():

            scaler = StandardScaler()
            features_df = pd.read_pickle(f'../data/{feature_set}.pkl')
            feature_names = features_df.columns

            X_train = scaler.fit_transform(features_df.loc[self.train_index, :])
            Y_train = self.bom_df.loc[self.train_index, 'book']

            X_val = scaler.transform(features_df.loc[self.val_index, :])
            Y_val = self.bom_df.loc[self.val_index, 'book']

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
    results = experiment.run("author_results.pkl")
    print(results)