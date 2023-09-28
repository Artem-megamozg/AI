import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC

df_train = pd.read_csv('обучение/train_AIC_processed_v19.csv', index_col=0)
df_test = pd.read_csv('обучение/test_AIC_processed_v19.csv', index_col=0)

X = df_train.drop(["y"], axis = 1)
y = df_train["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=1)

clf1 = BernoulliNB()
clf1.fit(X_train, y_train)
clf1.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf1, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность BernoulliNB: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


clf2 = LogisticRegression()
clf2.fit(X_train, y_train)
clf2.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf2, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность LogisticRegression: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


clf3 = DecisionTreeClassifier(max_depth=2, random_state=42)
clf3.fit(X_train, y_train)
clf3.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf3, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность DecisionTreeClassifier: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


clf4 = CatBoostClassifier()
clf4.fit(X_train, y_train)
clf4.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf4, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность CatBoostClassifier: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


clf5 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
clf5.fit(X_train, y_train)
clf5.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf5, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность XGBClassifier: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


clf6 = RandomForestClassifier()
clf6.fit(X_train, y_train)
clf6.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf6, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность RandomForestClassifier: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


# clf7 = lgb.LGBMClassifier()
# clf7.fit(X_train, y_train)
# y_pred = clf7.predict(X_test)

clf8 = GradientBoostingClassifier()
clf8.fit(X_train, y_train)
clf8.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf8, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность GradientBoostingClassifier: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")



clf10 = BaggingClassifier()
clf10.fit(X_train, y_train)
clf10.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf10, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность BaggingClassifier: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


clf11 = DummyClassifier(strategy="most_frequent")
clf11.fit(X_train, y_train)
clf11.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf11, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность DummyClassifier: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


clf12 = LinearSVC(C=0.01, random_state=42)
clf12.fit(X_train, y_train)
clf12.predict(X_test)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(clf12, X_test, y_test)
print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность LinearSVC: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")







