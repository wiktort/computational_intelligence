"""
Zadanie 1
W tym zadaniu wykonamy kilka komend matematyczno-statystycznych:
a) Zapisz pod zmienną a liczbę 123, a pod zmienną b liczbę 321. Następnie prostą komendą policz i zwróć wynik mnożenia liczb.
b) Wczytaj dwa wektory z liczbami [3, 8, 9, 10, 12] oraz [8, 7, 7, 5, 6]. Następnie zwróć sumę tych wektorów oraz iloczyn (po współrzędnych) tych wektorów.
c) Dla powyższych wektorów podaj iloczyn skalarny i długości euklidesowe.
d) Stwórz dwie dowolne macierze 3 x 3 pomnóż je po współrzędnych i wyświetl wynik,
a następnie pomnóż je macierzowo i wyświetl wynik.
e) Stwórz wektor 50 losowych liczb z zakresu od 1 do 100.
f) Dla wektora z punktu (e) policz średnią z wszystkich jego liczb, min, max oraz
odchylenie standardowe.
g) Dokonaj normalizacji wektora z podpunktu (e) (ściskamy wszystkie liczby do
przedziału [0,1]) za pomocą poniższego wzoru (xi to liczba w starym wektorze na pozycji i, a zi to liczba w nowym wektorze na pozycji i)

 W oryginalnym wektorze jakie było max? Na której pozycji stało? Jaka liczba stoi na tej pozycji w nowym wektorze?
"""

import numpy as np
import statistics
# a
a = 123
b = 321
product = a * b

# b
vector_a = np.array([3, 8, 9, 10, 12])
vector_b = np.array([8, 7, 7, 5, 6])
sum_of_vectors = np.add.reduce([vector_a, vector_b]);
product_of_vectors = vector_a * vector_b
# print(sum_of_vectors, product_of_vectors)

# c
dot_product = np.dot(vector_a, vector_b)
# print(dot_product)

# d
matrix_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
product_of_matrixes = matrix_1 * matrix_2
matrix_product = np.matmul(matrix_1, matrix_2)

# print(product_of_matrixes, matrix_product)

# e
vector_with_randoms = np.random.randint(1, 100, 50)
# print(vector_with_randoms)

# f
average = sum(vector_with_randoms)/len(vector_with_randoms)
min_value = min(vector_with_randoms)
max_value = max(vector_with_randoms)
standard_deviation = statistics.stdev(vector_with_randoms)
# print(average, min_value, max_value, standard_deviation)

#g
normalised_vector = vector_with_randoms / np.sqrt(np.sum(vector_with_randoms**2))
index_of_max_in_base_vector = vector_with_randoms.argmax(0)
max_value_in_normalised_vector = normalised_vector[index_of_max_in_base_vector]
# print(normalised_vector, index_of_max_in_base_vector, max_value_in_normalised_vector)


