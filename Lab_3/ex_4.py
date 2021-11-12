"""
Powtórz eksperyment z zadania 2 i 3 (klasyfikatory: drzewo, 1NN, 3NN, 5NN, 7NN, 11 NN) dla innego zbioru danych: diabetes.csv (załączony plik). Tutaj klasyfikacja polega na diagnozowaniu cukrzycy (u kobiet pochodzących z rdzennych plemion w Ameryce). Zgadujemy czy osoba jest chora, czy zdrowia, patrząc na parametry jej organizmu (wagę, parametry krwi, liczbę ciąż, itp.).
Dokładności wszystkich klasyfikatorów zestaw na wykresie słupkowym

Pytanie dodatkowe: chcemy zminimalizować błędy, gdy klasyfikator chore osoby klasyfikuje jako zdrowe ( i odsyła do domu bez leków). Który z klasyfikatorów najbardziej się do tego nadaje?

"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


def print_output(confusion_matrix: [[int]], test_set_length: int, output_for: str):
    correct = confusion_matrix[0][0] + confusion_matrix[1][1]
    correct_ratio = round(correct/test_set_length * 100, 2)
    print("Output for: {}".format(output_for))
    print("Correct predictions: {}%".format(correct_ratio))
    cm = confusion_matrix
    print("                     true-positive  true-negative ")
    print("false-positive            {}              {}".format(cm[0][0], cm[0][1]))
    print("false-negative            {}              {}".format(cm[1][0], cm[1][1]))
    print()
    return correct_ratio


classes = ['tested_positive', 'tested_negative']
full_set = pd.read_csv('diabetes.csv')


X = full_set.iloc[:, :-1].values
Y = full_set.iloc[:, 8].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
test_set_length = round(len(X_test))

scaler = StandardScaler()
scaler.fit(X_train)
X_train_NN = scaler.transform(X_train)
X_test_NN = scaler.transform(X_test)


# decision tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train,)
Y_pred_tree = clf.predict(X_test)
cm_tree = confusion_matrix(Y_test, Y_pred_tree)

correct_ratio_tree = print_output(cm_tree, test_set_length, "decision tree")



# 1NN
classifier_1 = KNeighborsClassifier(n_neighbors=1)
classifier_1.fit(X_train_NN, Y_train)
Y_pred_1 = classifier_1.predict(X_test_NN)
cm_1 = confusion_matrix(Y_test, Y_pred_1)

correct_ratio_1 = print_output(cm_1, test_set_length, "k = 1")

# 3NN
classifier_3 = KNeighborsClassifier(n_neighbors=3)
classifier_3.fit(X_train_NN, Y_train)
Y_pred_3 = classifier_3.predict(X_test_NN)
cm_3 = confusion_matrix(Y_test, Y_pred_3)

correct_ratio_3 = print_output(cm_3, test_set_length, "k = 3")

# 5NN
classifier_5 = KNeighborsClassifier(n_neighbors=5)
classifier_5.fit(X_train_NN, Y_train)
Y_pred_5 = classifier_5.predict(X_test_NN)
cm_5 = confusion_matrix(Y_test, Y_pred_5)

correct_ratio_5 = print_output(cm_5, test_set_length, "k = 5")

# 7NN
classifier_7 = KNeighborsClassifier(n_neighbors=7)
classifier_7.fit(X_train_NN, Y_train)
Y_pred_7 = classifier_7.predict(X_test_NN)
cm_7 = confusion_matrix(Y_test, Y_pred_7)

correct_ratio_7 = print_output(cm_7, test_set_length, "k = 7")

# 11NN
classifier_11 = KNeighborsClassifier(n_neighbors=11)
classifier_11.fit(X_train_NN, Y_train)
Y_pred_11 = classifier_11.predict(X_test_NN)
cm_11 = confusion_matrix(Y_test, Y_pred_11)

correct_ratio_11 = print_output(cm_11, test_set_length, "k = 11")


# print bar chart
s = pd.Series(
    [correct_ratio_tree, correct_ratio_1, correct_ratio_3, correct_ratio_5, correct_ratio_7, correct_ratio_11],
    index=['Decision tree', '1NN', '3NN', '5NN', '7NN', '11NN']
)
my_colors = list('rgbkymc')
plt.ylabel('%')
s.plot(
    kind='bar',
    color=my_colors,
    rot=0
)
plt.show()
