import csv


def classify_iris(sl, sw, pl, pw):
    if sl <= 4.9 and pw <= 0.6 or sw >= 3.0 and pl <= 2.0 and pw <= 0.6:
        return 'setosa'
    else:
        if 2.0 < pl <= 5.1 and 1.8 > pw >= 1.0:
            return 'versicolor'
        else:
            return 'virginica'


with open('iris.csv') as iris_csv:
    csv_reader = csv.reader(iris_csv)
    column_names = next(csv_reader)
    correct = 0
    iris_length = 0
    for row in csv_reader:
        iris_length = iris_length + 1
        predicted_row = classify_iris(float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        if predicted_row == row[4]:
            correct = correct + 1
    print("Correct predictions: {}%".format(round(correct/iris_length * 100, 2)))




