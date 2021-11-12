"""
Wykorzystując wiedzę z samouczków wykonaj następujące polecenia.
a) Podziel w losowy sposób bazę danych irysów na zbiór treningowy i zbiór
testowy w proporcjach 70%/30%. Wyświetl oba zbiory.
b) Wytrenuj drzewo decyzyjne na zbiorze treningowym.
c) Wyświetl drzewo w formie tekstowej i w formie graficznej.
d) Dokonaj ewaluacji klasyfikatora: sprawdź jak drzewo poradzi sobie z rekordami
ze zbioru testowego. Wyświetl procent poprawnych odpowiedzi.
e) Wyświetl macierz błędu(confusionmatrix) dla tej ewaluacji. Wyjaśnij jakie błędy
popełniał klasyfikator wskazując na liczby w macierzy błędu.
"""
import matplotlib.pyplot as plt
from sklearn import tree
import csv
import random
from sklearn.metrics import confusion_matrix

classes = ['setosa', 'versicolor', 'virginica']
full_set = []

training_set = []
training_target = []

test_set = []
test_target = []

with open('iris.csv') as iris_csv:
    csv_reader = csv.reader(iris_csv)
    column_names = next(csv_reader)
    for row in csv_reader:
        full_set.append(row)

random.shuffle(full_set)
full_set_length = len(full_set)
training_set_length = round(full_set_length * 0.7)
test_set_length = full_set_length - training_set_length

for index in range(0, training_set_length):
    row = full_set[index]
    training_set.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    training_target.append(classes.index(row[4]))

for index in range(training_set_length, full_set_length):
    row = full_set[index]
    test_set.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    test_target.append(classes.index(row[4]))


# ad. A
print('Training set: \n{}\n'.format(training_set))
print('Test set: \n{}\n'.format(test_set))

# ad. B
clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_set, training_target)

# ad. C
print("Text tree:")
print(tree.export_text(clf))
fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 20), dpi=80)
tree.plot_tree(clf, feature_names=fn, class_names=classes, filled=True)
fig.savefig('decisionTree.png')

# ad. D
predicted_classes = clf.predict(test_set)
correct = 0
for index in range(0, test_set_length):
    if predicted_classes[index] == test_target[index]:
        correct = correct + 1
print()
print("Correct predictions: {}%".format(round(correct/test_set_length * 100, 2)))


# ad. E
cm = confusion_matrix(test_target, predicted_classes)

print("                              true")
print("                     setosa versicolor virginica")
print("        setosa         {}       {}         {}".format(cm[0][0], cm[0][1], cm[0][2]))
print("false   versicolor     {}       {}         {}".format(cm[1][0], cm[1][1], cm[1][2]))
print("        virginica      {}       {}         {}".format(cm[2][0], cm[2][1], cm[2][2]))




