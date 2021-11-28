import numpy as np
import pandas as pd


full_set = pd.read_csv('diabetes.csv')

# Ad. a
new_class = {'tested_positive': 1, 'tested_negative': 0}
full_set['class'] = full_set['class'].map(new_class)


# Ad. b
def get_interquartile_range(arr):
    q1 = np.quantile(arr, 0.25)
    median = np.quantile(arr, 0.5)
    q3 = np.quantile(arr, 0.75)
    IQR = q3 - q1
    return [median - 1.5 * IQR, median + 1.5 * IQR]


def remove_index(_index):
    global full_set
    full_set = full_set.drop(axis=0, index=_index)


indices_to_remove = []
for column in full_set.columns:
    if column == 'class':
        break
    interquartile_range = get_interquartile_range(full_set[column])
    for index in range(0, len(full_set[column])):
        if full_set[column][index] < interquartile_range[0] or full_set[column][index] > interquartile_range[1]:
            indices_to_remove.append(index)


for index in set(indices_to_remove):
    remove_index(index)


# Ad. c
for column in full_set.columns:
    if column == 'class':
        break
    min_value = min(full_set[column])
    max_value = max(full_set[column])
    full_set[column] = full_set[column].apply(lambda value: (value-min_value)/(max_value-min_value))

full_set.to_csv('diabetes_prepared.csv', index=False)







