################
# Autor: Thalles Stanziola
# Descrição: Simulador de IA genética em python com Hill Climbing híbrido
# Data: 16/05/2025
# Versão: 2.0
################

import random  # Para gerar números aleatórios (seleção, mutação e crossover)
import matplotlib.pyplot as plt  # Para plotar gráficos da evolução dos coeficientes
from prettytable import PrettyTable  # Para exibir tabelas de resultados

print("Bibliotecas importadas com sucesso!")

# Função de aptidão: soma dos quadrados (quanto menor, melhor)
def fitness_function(individual):
    a, b, c = individual
    return - (a**2 + b**2 + c**2)

# Cria população inicial
def create_initial_population(pop_size, lb, ub):
    return [(
        random.uniform(lb, ub),
        random.uniform(lb, ub),
        random.uniform(lb, ub)
    ) for _ in range(pop_size)]

# Seleção por roleta
def selection(population, fitnesses):
    total = sum(fitnesses)
    if total == 0:
        return random.choices(population, k=len(population))
    weights = [f/total for f in fitnesses]
    return random.choices(population, weights=weights, k=len(population))

# Crossover de um ponto
def crossover(p1, p2):
    pt = random.randint(1, 2)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

# Mutação simples
def mutation(individual, rate, lb, ub):
    genes = []
    for g in individual:
        genes.append(random.uniform(lb, ub) if random.random() < rate else g)
    return tuple(genes)

# Hill Climbing (Steepest-Ascent)
def hill_climb(individual, fitness_fn, step=0.5, max_iters=10):
    current = list(individual)
    current_fit = fitness_fn(tuple(current))
    for _ in range(max_iters):
        neighborhood = []
        for i in range(len(current)):
            for delta in (-step, step):
                nb = current.copy()
                nb[i] += delta
                neighborhood.append(nb)
        best_nb, best_fit = current, current_fit
        for nb in neighborhood:
            f = fitness_fn(tuple(nb))
            if f > best_fit:
                best_nb, best_fit = nb, f
        if best_fit <= current_fit:
            break
        current, current_fit = best_nb, best_fit
    return tuple(current)

# Algoritmo Genético híbrido com HC seletivo
def genetic_algorithm(pop_size, lower_bound, upper_bound, generations, mutation_rate,
                      hc_step=0.5, hc_iters=10, hc_frac=0.1):
    population = create_initial_population(pop_size, lower_bound, upper_bound)
    best_performers = []
    table = PrettyTable(["Geração", "a", "b", "c", "Aptidão"])

    for gen in range(1, generations+1):
        fitnesses = [fitness_function(ind) for ind in population]
        selected = selection(population, fitnesses)
        offspring = []
        # Crossover e mutação
        for i in range(0, len(selected), 2):
            p1 = selected[i]
            p2 = selected[i+1] if i+1 < len(selected) else selected[0]
            for child in crossover(p1, p2):
                mchild = mutation(child, mutation_rate, lower_bound, upper_bound)
                offspring.append(mchild)
        # Hill Climbing em hc_frac dos filhos
        hc_count = max(1, int(hc_frac * len(offspring)))
        for idx in random.sample(range(len(offspring)), k=hc_count):
            offspring[idx] = hill_climb(offspring[idx], fitness_function, hc_step, hc_iters)
        population = offspring
        # Registro do melhor
        best = max(population, key=fitness_function)
        bfit = fitness_function(best)
        best_performers.append((best, bfit))
        table.add_row([gen, best[0], best[1], best[2], bfit])

    print(table)

    # Gráfico da evolução dos coeficientes
    gens = list(range(1, generations+1))
    a_vals = [ind[0][0] for ind in best_performers]
    b_vals = [ind[0][1] for ind in best_performers]
    c_vals = [ind[0][2] for ind in best_performers]

    plt.figure(figsize=(12, 6))
    plt.plot(gens, a_vals, label='a')
    plt.plot(gens, b_vals, label='b')
    plt.plot(gens, c_vals, label='c')
    plt.xlabel('Geração')
    plt.ylabel('Valores dos Coeficientes')
    plt.title('Evolução dos Coeficientes com GA + HC')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Retorna o melhor indivíduo que apareceu em qualquer geração
    global_best = max(best_performers, key=lambda x: x[1])[0]
    return global_best


# Execução com parâmetros recomendados
if __name__ == '__main__':
    top = genetic_algorithm(
        pop_size=60,
        lower_bound=-20,
        upper_bound=20,
        generations=60,
        mutation_rate=0.6,  # Mutação alta
        hc_step=0.3,  # Passos médios
        hc_iters=10,
        hc_frac=0.2
    )
    print(f"\nMelhor solução encontrada: {top}")
    print(f"Aptidão da melhor solução: {fitness_function(top)}")
