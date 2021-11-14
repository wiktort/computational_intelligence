import math

columns = ['wiek', 'waga', 'wzrost', 'gra']
row = [
    [23, 75, 176, True],
    [25, 67, 180, True],
    [28, 120, 175, False],
    [22, 65, 165, True],
    [46, 70, 187, True],
    [50, 68, 180, False],
    [48, 97, 178, False]
]


def f_act(x):
    return 1 / (1 + math.exp(-x))


def forward_pass(age, weight, height):
    a = float(age)
    w = float(weight)
    h = float(height)
    h1 = f_act(a * -0.46122 + w * 0.97314 + h * -0.39203 + 0.80109)
    h2 = f_act(a * 0.78548 + w * 2.10584 + h * -0.57847 + 0.43529)
    return h1 * -0.81546 + h2 * 1.03775 - 0.2368


print("Output for (23, 75, 176): {}".format(forward_pass(row[0][0], row[0][1], row[0][2])))
print("Output for (25, 67, 180): {}".format(forward_pass(row[1][0], row[1][1], row[1][2])))
print("Output for (28, 120, 175): {}".format(forward_pass(row[2][0], row[2][1], row[2][2])))

