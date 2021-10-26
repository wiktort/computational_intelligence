import numpy as np
import pygad
import math

#0 - ściana, 1 - droga, 2 - start, 3 - drzwi
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

#wyjście
door = [10, 10]

#Pytanie bazowe
# Czy istnieje droga o maksymalnie 30 krokach, ze startu do exitu, w labiryncie 12x12 przedstawionym na obrazku?

#definiujemy parametry chromosomu
#geny to liczby: 0, 1, 2 lub 3 gdzie:
#0 - ruch do góry, 1 - ruch w prawo, 2 - ruch w dół, 3 - ruch w lewo
gene_space = [0, 1, 2, 3]

def check_move(row_index: int, column_index: int):
    spot = maze[row_index][column_index]
    if spot == 0:
        return [0, False]
    elif spot == 3:
        return [1, True]
    else:
        return [1, False]


#definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    row_index = 1
    column_index = 1
    is_finished = False
    moves = 0
    bonus = 0

    for index in range(len(solution)):
        move = solution[index]
        if move == 0:
            move_result, is_finished = check_move(row_index - 1, column_index)
            row_index = row_index - move_result
        elif move == 1:
            move_result, is_finished = check_move(row_index, column_index + 1)
            column_index = column_index + move_result
        elif move == 2:
            move_result, is_finished = check_move(row_index + 1, column_index)
            row_index = row_index + move_result
        else:
            move_result, is_finished = check_move(row_index, column_index - 1)
            column_index = column_index - move_result
        bonus = bonus + np.abs(move_result/1000)
        if is_finished:
            moves = index + 1
            break

    # if is_finished:
    #     return fitness + moves

    distance_to_exit = math.sqrt(math.pow(door[1] - row_index, 2) + math.pow(door[0] - column_index, 2))
    fitness = 1.0 / (1.0 + distance_to_exit)
    if is_finished:
        return fitness + bonus + 5
    return fitness + bonus

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 100
num_genes = 30

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = int(sol_per_pop * 0.65)
num_generations = 35
keep_parents = int(sol_per_pop * 0.25)

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 8

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       )
# uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()

