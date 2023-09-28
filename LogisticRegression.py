import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Загрузка данных
data = pd.read_csv('train_AIC.csv')
df_test = pd.read_csv('test_AIC.csv', index_col=0)

# Разделение на признаки и метки классов
X = data.drop('y', axis=1)
y = data['y']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Предобработка данных: масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Предобработка данных: выбор наиболее информативных признаков
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Обучение модели
model = LogisticRegression(max_iter=17000,
                           tol=0.0005,
                           multi_class='ovr',
                           random_state=175,
                           C=0.22498757747329218,
                           solver='lbfgs')
model.fit(X_train_selected, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test_selected)

# Оценка модели
report = classification_report(y_test, y_pred)
print(report)
