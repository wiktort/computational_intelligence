import numpy as np
from matplotlib import pyplot as plt

with open('cnn_data.npy', 'rb') as cnn_data:
    base_image = np.load(cnn_data)

def draw(img, x, y, color):
    img[x, y] = [color, color, color]

# b -  nakładanie filtra na obrazek
# filter:
# |1 0 -1|
# |1 0 -1|
# |1 0 -1|


image_with_filter = np.zeros((126, 126, 3), dtype=np.uint8)
image_with_relu = np.zeros((126, 126, 3), dtype=np.uint8)
image_with_rec = np.zeros((126, 126, 3), dtype=np.uint8)
for i in range(126):
    for j in range(126):
        color = base_image[i][j][0] * 1 + base_image[i + 1][j][0] * 0 + base_image[i + 2][j][0] * -1 \
                + base_image[i][j+1][0] * 1 + base_image[i+1][j+1][0] * 0 + base_image[i+2][j+1][0] * -1 \
                + base_image[i][j+2][0] * 1 + base_image[i+1][j+2][0] * 0 + base_image[i+2][j+2][0] * -1
        # c
        color_relu = 0;
        if color < 0:
            color_relu = 0
        else:
            color_relu = color
        # d
        color_f = 0
        if color < 0:
            color_f = 0
        elif 0 <= color <= 255:
            color_f = color
        else:
            color_f = 255

        draw(image_with_filter, i, j, color)
        draw(image_with_relu, i, j, color_relu)
        draw(image_with_rec, i, j, color_f)


# e
# filter:
# | 1  1  1|
# | 0  0  0|
# |-1 -1 -1|
image_with_filter_2 = np.zeros((126, 126, 3), dtype=np.uint8)
image_with_relu_2 = np.zeros((126, 126, 3), dtype=np.uint8)
image_with_rec_2 = np.zeros((126, 126, 3), dtype=np.uint8)
for i in range(126):
    for j in range(126):
        color = base_image[i][j][0] * 1 + base_image[i + 1][j][0] * 1 + base_image[i + 2][j][0] * 1 \
                + base_image[i][j + 1][0] * 0 + base_image[i + 1][j + 1][0] * 0 + base_image[i + 2][j + 1][0] * 0 \
                + base_image[i][j + 2][0] * -1 + base_image[i + 1][j + 2][0] * -1 + base_image[i + 2][j + 2][0] * -1
        # e (c)
        color_relu = 0
        if color < 0:
            color_relu = 0
        else:
            color_relu = color
        # e (d)
        color_f = 0
        if color < 0:
            color_f = 0
        elif 0 <= color <= 255:
            color_f = color
        else:
            color_f = 255

        draw(image_with_filter_2, i, j, color)
        draw(image_with_relu_2, i, j, color_relu)
        draw(image_with_rec_2, i, j, color_f)


# d
# Filter
# | 0  1  2|
# |-1  0  1|
# |-2 -1  0|
image_with_sobel = np.zeros((126, 126, 3), dtype=np.uint8)
image_with_relu_3 = np.zeros((126, 126, 3), dtype=np.uint8)
image_with_rec_3 = np.zeros((126, 126, 3), dtype=np.uint8)
for i in range(126):
    for j in range(126):
        color = base_image[i][j][0] * 0 + base_image[i + 1][j][0] * 1 + base_image[i + 2][j][0] * 2 \
                + base_image[i][j + 1][0] * -1 + base_image[i + 1][j + 1][0] * 0 + base_image[i + 2][j + 1][0] * 1 \
                + base_image[i][j + 2][0] * -2 + base_image[i + 1][j + 2][0] * -1 + base_image[i + 2][j + 2][0] * 0
        # e (c)
        color_relu = 0
        if color < 0:
            color_relu = 0
        else:
            color_relu = color
        # e (d)
        color_f = 0
        if color < 0:
            color_f = 0
        elif 0 <= color <= 255:
            color_f = color
        else:
            color_f = 255

        draw(image_with_sobel, i, j, color)
        draw(image_with_relu_3, i, j, color_relu)
        draw(image_with_rec_3, i, j, color_f)


# konwersja macierzy na obrazek i wyświetlenie
plt.imshow(image_with_rec_3, interpolation='nearest')
plt.show()


