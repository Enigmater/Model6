import numpy as np
import matplotlib.pyplot as plt

GRAPH = [
 [0, 0, 3, 3, 0],
 [5, 0, 4, 0, 0],
 [9, 3, 0, 2, 5],
 [0, 0, 0, 0, 0],
 [5, 8, 6, 0, 0]
]
N = len(GRAPH)
L = np.array(GRAPH, dtype=float) # преобраузем граф

# уравнения колмагорова (матрица А)
def createKolmagorovSystem(L):
    A = np.zeros_like(L) # заполняем А нулями
    for i in range(N):
        for j in range(N):
            if i == j: continue #пропуск диагональных эл-ов
            A[i, j] = L[j, i]   # Заполняем входящие переходы
            A[i, i] -= L[i, j]  # Вычитаем исходящие переходы из диагонального элемента
    return A

# решение системы ДУ методом Эйлера
# L - матрица интенсивностей
# P0 - начальное состояние
# delta - шаг времени
# max - время эволюции системы
def euler(L, P0, delta, time):
    steps = int(time / delta)
    result = []              # Список для хранения значений вероятностей на каждом шаге
    P = P0.copy() 
    result.append(P.copy()) 

    for _ in range(steps):   # Итерации по времени
        deltaP = np.zeros(N) # Изменение вероятностей на текущем шаге
        for i in range(N):
            for j in range(N):
                if i == j: continue
                deltaP[i] -= L[i, j] * P[i] # вычитаем исходящие переходы
                deltaP[i] += L[j, i] * P[j] # добавляем входящие переходы
        P += deltaP * delta 
        P = np.clip(P, 0, 1) # ограничиваем P в (0, 1)
        result.append(P.copy())
    return result

# Функция проверки эргодичности системы
def isErgodic(L):
    # поиск в глубину по графу
    def dfs(vertex, graph):
        visited = set()
        stack = [vertex]
        while stack:
            edge = stack.pop()
            # не проверяем посещенные вершины
            if edge not in visited:
                visited.add(edge)
                # если переход edge - newEdge возможен
                for newEdge, cost in enumerate(graph[edge]):
                    if cost > 0: stack.append(newEdge)
        return visited
    
    # можем ли мы попасть из пункта i до всех пунктов
    LT = L.T # транспонированный граф
    for i in range(N):
        if len(dfs(i, L)) < N or len(dfs(i, LT)) < N: return False
    return True

# нахождение предельных вероятностей
def getLimitProbabilities(A):
    startA = A.copy() 
    print("Начальная матрица А:")
    print(startA) 
    b = np.zeros(N) # правая часть (заполняем нулями)
    b[-1] = 1       # условие нормировки
    print("\nСтолбец свободных членов:", b)
    A[-1] = np.ones(N) # заменяем последнюю строку на условие нормировки - сумма вероятностей = 1
    return np.linalg.solve(A, b) # решение СЛАУ

deltaTime = 0.01
timeSimulation = 2
P0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) # в 1ом состояние
A = createKolmagorovSystem(L)
solutions = euler(L, P0, deltaTime, timeSimulation)
ergodic = isErgodic(L)
limitProbabilities = getLimitProbabilities(A)

print("Система Колмагорова:")
print(A)

print("\nЭргодичность: ", "Есть" if ergodic else "Отсутствует")

print("\nПредельные вероятности:")
for i, p in enumerate(limitProbabilities):
    print(f"P{i+1} = {p:.4f}")

# построение графиков
solutions = np.array(solutions)
time = np.linspace(0, timeSimulation, len(solutions))
# размеры графика
plt.figure(figsize=(9, 5))
# рисуем график
for i in range(N):
    plt.plot(time, solutions[:, i], label=f'Состояние S{i+1}')

plt.xlabel('Время')
plt.ylabel('Вероятность') 
plt.legend()
plt.grid()
plt.show()