import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import optuna

df_train = pd.read_csv('обучение/train_AIC_processed_v6.csv', index_col=0)
df_test = pd.read_csv('обучение/test_AIC_processed_v6.csv', index_col=0)
#
# X = df_train.drop(["y"], axis=1)
# df_train.drop(["Количество позиций"], axis=1)
# df_test.drop(["Количество позиций"], axis=1)
# y = df_train["y"]
# df_train.head()
# df_test.head()
#
# features = list(set(df_train.columns))
#
# _ = df_train[features].hist(figsize=(40, 24))
#
# plt.rcParams['figure.figsize'] = (55, 55)
#
# corr = df_train.corr()
# g = sns.heatmap(corr, square = True, annot=True)


#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00001, random_state=1)
#
# undersample = EditedNearestNeighbours(n_neighbors=2)
# X_train_tl, y_train_tl = undersample.fit_resample(X_train, y_train)

X = df_train.drop(["y"], axis = 1)
y = df_train["y"]

X_train, y_train = X, y


class ModelTuner:
    def __init__(self, X, y, n_splits=4, random_seed=42, n_trials=5, scoring='roc_auc', direction='maximize'):
        # dataset
        self.X = X_train
        self.y = y_train
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


 # hyperparameters for both xgb_classifier and xgb_regressor
 #    def params(self, trial):
 #        params = {
 #            'iterations': trial.suggest_int('iterations', 100, 30000),
 #            'depth': trial.suggest_int('depth', 1, 30),
 #            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 20),
 #            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation _iterations', 1, 30),
 #            'random_strength': trial.suggest_int('random_strength', 1, 200),
 #            'random_state': trial.suggest_int('random_state', 1, 100000),
 #            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
 #            'rsm': trial.suggest_float('rsm', 0.001, 1.0),
 #          }
 #
 #        return params

    def params(self, trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 35000),
            'learning_rate': trial.suggest_int('learning_rate', 0.01, 100),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 0, 500),
            'random_state': trial.suggest_int('random_state', 1, 1500),
            'depth': trial.suggest_int('depth', 1, 200),
            'rsm': trial.suggest_int('rsm', 0.01, 200),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 400),
            'random_strength': trial.suggest_int('random_strength', 1, 500)
          }

        return params

    # takes input function and optimize it
    def tune(self, objective):
        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_score = study.best_value
        print(f"Best score: {self.best_score} \nOptimized parameters: {self.best_params}")

    ## choose the objective function you want to study

    # xgb classifier
    def CatBoostClassifier(self, trial):
        # model
        xgb_cls = CatBoostClassifier(**self.params(trial))
        # cross validation
        score = cross_val_score(xgb_cls, self.X, self.y, cv=self.kfolds, scoring=self.scoring).mean()
        return score

### How to use ModelTuner:

# Classification

# create an object
model_tuner = ModelTuner(X, y, n_splits=3, random_seed=42,  n_trials = 2, scoring= 'roc_auc', direction ='maximize')

# to tune the model, we use the tune method and pass the objective function we wish to optimize, as shown below:
model_tuner.tune(model_tuner.CatBoostClassifier)

# ctt_model = CatBoostClassifier(iterations=12907,
#                               depth=9,
#                               l2_leaf_reg=6,
#                               leaf_estimation_iterations=11,
#                               random_strength=10,
#                               random_state=8247,
#                               learning_rate=0.15425511458247868,
#                               rsm=0.1617041365198415)
# ctt_model.fit(X_train_tl, y_train_tl)

# modelx = XGBClassifier(learning_rate=0.415569315124448,
#                        n_estimators=167,
#                        max_depth=5,
#                        min_child_weight=5,
#                        subsample=0.9351530084184804,
#                        reg_alpha=0.6272616547652957,
#                        reg_lambda=0.49945111631818806,
#                        gamma=0.48508662520673407,
#                        colsample_bytree=0.8743725137983754)
# modelx.fit(X_train_tl, y_train_tl)
# print(modelx.fit)
#
# model = (modelx and ctt_model)

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred)
#     return accuracy, precision, recall, f1, roc_auc

#
# rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(model, X_test, y_test)
#
# print("Результаты оценки производительности модели на тестовой выборке:")
# print(f"Точность: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")
#
# X = df_test.drop([], axis=1)
# df_test["value"] = model.predict(X)
# df_test.head()
#
# list(df_test)
#
# df_test["value"].to_csv("test.csv")
#
# df_submit = pd.read_csv('test.csv', index_col=0)
# df_submit.head()
