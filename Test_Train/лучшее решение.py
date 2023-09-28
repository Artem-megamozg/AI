import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

df_train = pd.read_csv('train_AIC_processed_v19.csv', index_col=0)
df_test = pd.read_csv('test_AIC_processed_v19.csv', index_col=0)

X = df_train.drop(["y"], axis=1)
y = df_train["y"]

X_train, y_train = X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, shuffle=True, stratify=y)

ct_model = CatBoostClassifier(iterations=12907,
                              depth=9,
                              l2_leaf_reg=6,
                              leaf_estimation_iterations=11,
                              random_strength=10,
                              random_state=8247,
                              learning_rate=0.15425511458247868,
                              rsm=0.1617041365198415,
                              one_hot_max_size=50
                              )

ct_model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc


rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(ct_model, X_test, y_test)

print("Результаты оценки производительности модели на тестовой выборке:")
print(f"Точность: {rf_accuracy:.4f}, Полнота: {rf_recall:.4f}, F1-мера: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")


X = df_test
df_test["value"] = ct_model.predict(X)
df_test.head()
list(df_test)


df_test["value"].to_csv("test1.csv")

df_submit = pd.read_csv('test1.csv', index_col=0)
df_submit.head()
