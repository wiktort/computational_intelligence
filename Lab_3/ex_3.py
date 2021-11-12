import csv
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
training_set_length = round(full_set_length * 0.67)
test_set_length = full_set_length - training_set_length

for index in range(0, training_set_length):
    row = full_set[index]
    training_set.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    training_target.append(classes.index(row[4]))

for index in range(training_set_length, full_set_length):
    row = full_set[index]
    test_set.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    test_target.append(classes.index(row[4]))

scaler = StandardScaler()
scaler.fit(training_set)
X_train = scaler.transform(training_set)
X_test = scaler.transform(test_set)

# Ad. b

def print_output(confusion_matrix: [[int]], test_set_length: int, k_neighbours: int):
    correct = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]
    print("Output for k = {}".format(k_neighbours))
    print("Correct predictions: {}%".format(round(correct/test_set_length * 100, 2)))
    cm = confusion_matrix
    print("                              true")
    print("                     setosa versicolor virginica")
    print("        setosa         {}       {}         {}".format(cm[0][0], cm[0][1], cm[0][2]))
    print("false   versicolor     {}       {}         {}".format(cm[1][0], cm[1][1], cm[1][2]))
    print("        virginica      {}       {}         {}".format(cm[2][0], cm[2][1], cm[2][2]))
    print()


# 2 neighbours
classifier_2 = KNeighborsClassifier(n_neighbors=2)
classifier_2.fit(X_train, training_target)
pred_target = classifier_2.predict(X_test)
cm_2 = confusion_matrix(test_target, pred_target)

print_output(cm_2, test_set_length, 2)

# 3 neighbours
classifier_3 = KNeighborsClassifier(n_neighbors=3)
classifier_3.fit(X_train, training_target)
pred_target = classifier_3.predict(X_test)
cm_3 = confusion_matrix(test_target, pred_target)

print_output(cm_3, test_set_length, 3)

# 4 neighbours
classifier_4 = KNeighborsClassifier(n_neighbors=4)
classifier_4.fit(X_train, training_target)
pred_target = classifier_4.predict(X_test)
cm_4 = confusion_matrix(test_target, pred_target)

print_output(cm_4, test_set_length, 4)

# 5 neighbours
classifier_5 = KNeighborsClassifier(n_neighbors=5)
classifier_5.fit(X_train, training_target)
pred_target = classifier_5.predict(X_test)
cm_5 = confusion_matrix(test_target, pred_target)

print_output(cm_5, test_set_length, 3)





