import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split


# df = pd.read_csv('diabetes_prepared.csv')
df = pd.read_csv('diabetes_prepared.csv')

# a
target_column = ['class']
predictors = list(set(list(df.columns))-set(target_column))
# df[predictors] = df[predictors]/df[predictors].max()

X = df[predictors].values
y = df[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

# b
mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation='relu', max_iter=900)
# c
mlp.fit(X_train, y_train.ravel())

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

# d
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))