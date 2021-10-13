import pandas as pd
import csv
from matplotlib import pyplot as plt

year_to_append = '2010'
row_to_append = {'Rok': year_to_append, 'Gdansk': '460', 'Poznan': '555', 'Szczecin': '405'}
should_append: bool = True

"""
a) print data
"""
with open('miasta.csv') as cities_csv:
    csv_reader = csv.reader(cities_csv)
    column_names = next(csv_reader)
    print(column_names)
    for row in csv_reader:
        print(row)

with open('miasta.csv') as cities_csv:
    csv_reader = csv.DictReader(cities_csv)
    for row in csv_reader:
        if row['Rok'] == year_to_append:
            should_append = False
"""
b) add data
"""
with open('miasta.csv', 'a', newline='') as cities_csv:
    if should_append:
        csv_writter = csv.DictWriter(cities_csv, fieldnames=column_names)
        csv_writter.writerows([row_to_append])


with open('miasta.csv') as cities_csv:
    cities = pd.read_csv(cities_csv, index_col=0)
    print(cities)

    """
    c) Gdansk
    """

    plt.figure(1)
    plt.plot(cities['Gdansk'], linestyle='-', marker='o', color='r')
    plt.title('Ludnosc w Gdansku')
    plt.xlabel('Lata')
    plt.ylabel('Liczba ludnosci [w tys.]')
    plt.legend(['Gdansk'])

    """
    d) All cities
    """
    plt.figure(2)
    plt.plot(cities, linestyle='-', marker='o')
    plt.title('Ludnosc w miastach Polski')
    plt.xlabel('Lata')
    plt.ylabel('Liczba ludnosci [w tys.]')
    plt.legend(cities.axes[1])

    plt.show()
