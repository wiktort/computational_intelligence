import pygad
import time

S = [
    {"name": "watch", "value": 100, "weight": 7},
    {"name": "painting-landscape", "value": 300, "weight": 7},
    {"name": "painting-portrait", "value": 200, "weight": 6},
    {"name": "radio", "value": 40, "weight": 2},
    {"name": "laptop", "value": 500, "weight": 5},
    {"name": "lamp", "value": 70, "weight": 6},
    {"name": "porcelain", "value": 250, "weight": 3},
    {"name": "silver-cutlery", "value": 100, "weight": 1},
    {"name": "bronze-figure", "value": 300, "weight": 10},
    {"name": "leather-purse", "value": 280, "weight": 3},
    {"name": "vacuum-cleaner", "value": 300, "weight": 15},
]

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcjęfitness
def fitness_func(solution, solution_idx):
    total_weight = 0
    total_value = 0

    for index in range(len(solution)):
        if solution[index] == 1:
            total_weight = total_weight + S[index]['weight']
            total_value = total_value + S[index]['value']
    if total_weight > 25:
        return 0
    return total_value

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 11
num_genes = len(S)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = int(sol_per_pop * 0.5)
num_generations = 30
keep_parents = int(sol_per_pop * 0.2)

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 12

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
                       mutation_percent_genes=mutation_percent_genes)

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()