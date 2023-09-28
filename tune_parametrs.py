import optuna
import numpy as np
import pandas as pd
import sns
from imblearn.under_sampling import EditedNearestNeighbours
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_wine
import optuna
from optuna.samplers import TPESampler
import catboost
import pickle

df_train = pd.read_csv('train_AIC_processed_v6.csv', index_col=0)
# df_train.rename(columns = {'Поставщик':'id'}, inplace = True )
df_test = pd.read_csv('test_AIC_processed_v6.csv', index_col=0)

df_test.head()

features = list(set(df_train.columns))

_ = df_train[features].hist(figsize=(40,24))

plt.rcParams['figure.figsize']=(55,55)

# corr = df_train.corr()
# g = sns.heatmap(corr, square = True, annot=True)

df_train.head()

X = df_train.drop(["y"], axis = 1)
# df_train.drop(["Количество позиций"], axis = 1)
y = df_train["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00001, random_state=1)

# undersample = EditedNearestNeighbours(n_neighbors=5)
# X_train_tl, y_train_tl = undersample.fit_resample(X_train, y_train)

class ModelTuner:
    def __init__(self, X, y, n_splits=4, random_seed=42, n_trials=5, scoring='roc_auc', direction='maximize'):
        # dataset
        self.X = X
        self.y = y
        # KFold parameters
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.kfolds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        # scoring parameter for cross_val_score
        self.scoring = scoring
        # number of trials for optuna study
        self.n_trials = n_trials
        # direction for the objective function
        self.direction = direction

    model_tuner = ModelTuner(X_train, y_train, n_splits=3, random_seed=42, n_trials=20, scoring='recall',
                              direction='maximize')

def params(self, trial):
    # params = {
    #         'learning_rate': trial.suggest_float('learning_rate', 0.05, 1),
    #         'n_estimators': trial.suggest_int('n_estimators', 50, 200),
    #         'max_depth': trial.suggest_int('max_depth', 4, 6),
    #         'min_child_weight': trial.suggest_int('min_child_weight', 2, 5),
    #         'subsample': trial.suggest_float('subsample', 0.01, 1.0),
    #         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
    #         'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
    #         'gamma': trial.suggest_float('gamma', 0.1, 1.0),
    #         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
    #     }
    # return params

    def params(self, trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 1),
            'iterations': trial.suggest_int('iterations', 10000, 30000),
            'depth': trial.suggest_int('depth', 1, 15),
            'random_strength': trial.suggest_int('random_strength', 1, 500),
            'rsm': trial.suggest_int('rsm', 0.01, 200),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 0, 500)
        }
        return params

    # принимает входную функцию и оптимизирует ее
    def tune(self, objective):
        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_score = study.best_value
        print(f"Best score: {self.best_score} \nOptimized parameters: {self.best_params}")
    
    ## выберите целевую функцию, которую хотите изучить

    def catboost_CatBoostClassifier(self, trial):
        # model
        ct_cls = catboost.CatBoostClassifier(**self.params(trial))
        # cross validation
        score = cross_val_score(ct_cls, self.X, self.y, cv=self.kfolds, scoring=self.scoring).mean()
        return score

model_tuner = ModelTuner(X, y, n_splits=3, random_seed=42,  n_trials = 2, scoring= 'roc_auc', direction ='maximize')

model_tuner.tune(model_tuner.catboost_CatBoostClassifier)



