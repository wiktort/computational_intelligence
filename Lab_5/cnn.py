import numpy as np
from matplotlib import pyplot as plt

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128, 3), dtype=np.uint8)


# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = [color, color, color]


# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)


# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)
# save prepared data to file
# np.save('cnn_data.npy', data)

# b -  nakładanie filtra na obrazek
base_image = data
image_with_filter = np.zeros((126, 126, 3), dtype=np.uint8)
for i in range(126):
    for j in range(126):
        color = base_image[i][j][0] * 1 + base_image[i + 1][j][0] * 0 + base_image[i + 2][j][0] * -1 \
                + base_image[i][j+1][0] * 1 + base_image[i+1][j+1][0] * 0 + base_image[i+2][j+1][0] * -1 \
                + base_image[i][j+2][0] * 1 + base_image[i+1][j+2][0] * 0 + base_image[i+2][j+2][0] * -1

# konwersja macierzy na obrazek i wyświetlenie
plt.imshow(image_with_filter_2, interpolation='nearest')
plt.show()
