import pandas as pd
import numpy as np
from scipy import optimize

# Исходные данные, вариант 5
def f(x): return np.cos(x) - x
def df(x): return -np.sin(x) - 1
def phi(x): return np.cos(x)
def dphi(x): return -np.sin(x)

a, b = 0.0, 1.0
excel_roots = [
    0.7392578125, 0.7390781309, 0.7390851334,
    0.7390851334, 0.7387603199, 0.7390823364,
    0.7390847824, 0.7390851332, 0.7390851332,
    0.7390822985
]

# Табулирование
print("ТАБУЛИРОВАНИЕ")
xs = np.linspace(a, b, 6)
table = pd.DataFrame({"x": xs, "f(x)": [f(x) for x in xs]})
print(table.to_string(index=False))

# Поиск участка с корнем
for i in range(len(xs) - 1):
    if f(xs[i]) * f(xs[i + 1]) < 0:
        print("\nУчасток с корнем:", xs[i], xs[i + 1])
        print("Произведение =", f(xs[i]) * f(xs[i + 1]))
        break

# Условие сходимости МПИ
q = np.max(np.abs(dphi(np.linspace(a, b, 1000))))
print("\nmax |phi'(x)| =", q)
print("Условие сходимости выполнено:", q < 1)

# Метод бисекции
def bisection(f, a, b, eps):
    step, log = 0, []
    while True:
        step += 1
        mid = (a + b) / 2
        error = abs(b - a) / 2
        log.append([step, a, mid, error, f(mid)])
        if error < eps or f(mid) == 0: break
        if f(a) * f(mid) < 0: b = mid
        else: a = mid
    return mid, step, abs(f(mid)), log

# Метод хорд
def secant_chord(f, a, b, eps):
    step, log = 0, []
    x_curr, fixed = a, b
    while True:
        step += 1
        x_next = x_curr - f(x_curr)*(fixed-x_curr)/(f(fixed)-f(x_curr))
        error = abs(x_next - x_curr)
        log.append([step, x_curr, x_next, error, f(x_next)])
        if error < eps: break
        x_curr = x_next
    return x_next, step, abs(f(x_next)), log

# Метод Ньютона
def newton(f, df, x0, eps):
    step, log = 0, []
    x_curr = x0
    while True:
        step += 1
        x_next = x_curr - f(x_curr) / df(x_curr)
        error = abs(x_next - x_curr)
        log.append([step, x_curr, x_next, error, f(x_next)])
        if error < eps: break
        x_curr = x_next
    return x_next, step, abs(f(x_next)), log

# Метод простой итерации
def simple_iteration(phi, x0, eps):
    step, log = 0, []
    x_curr = x0
    while True:
        step += 1
        x_next = phi(x_curr)
        error = abs(x_next - x_curr)
        log.append([step, x_curr, x_next, error, f(x_next)])
        if error < eps: break
        x_curr = x_next
    return x_next, step, abs(f(x_next)), log

# Вывод итераций
def print_iterations(name, log):
    data = pd.DataFrame(log, columns=[
        "Итерация", "x текущее", "x следующее", "Ошибка", "f(x)"
    ])
    print("\n" + name)
    print(data.to_string(index=False))

results = []
excel_idx = 0

# Расчеты для двух точностей
for eps in [1e-3, 1e-5]:

    print("ТОЧНОСТЬ =", eps)


    root, n, res, log = bisection(f, a, b, eps)
    print_iterations("Бисекция", log)
    lib = optimize.bisect(f, a, b, xtol=eps)
    results.append(["Бисекция", excel_roots[excel_idx], lib, root, eps, n, res])
    excel_idx += 1

    root, n, res, log = secant_chord(f, a, b, eps)
    print_iterations("Метод хорд", log)
    lib = optimize.root_scalar(f, method="secant", x0=a, x1=b, xtol=eps).root
    results.append(["Хорд", excel_roots[excel_idx], lib, root, eps, n, res])
    excel_idx += 1

    for x0 in [1.0, 0.0]:
        root, n, res, log = newton(f, df, x0, eps)
        print_iterations("Ньютон x0=" + str(x0), log)
        lib = optimize.newton(f, x0, fprime=df, tol=eps)
        name = "Ньютона (x0=" + str(x0) + ")"
        results.append([name, excel_roots[excel_idx], lib, root, eps, n, res])
        excel_idx += 1

    x0 = (a + b) / 2
    root, n, res, log = simple_iteration(phi, x0, eps)
    print_iterations("Простая итерация", log)
    lib = optimize.root_scalar(f, bracket=[a, b], xtol=eps).root
    results.append([
        "Простой итерации", excel_roots[excel_idx], lib, root, eps, n, res
    ])
    excel_idx += 1

# Итоговая таблица
df_results = pd.DataFrame(results, columns=[
    "Метод", "Корень (Excel)", "Корень (SciPy)",
    "Найденный корень (код)", "Точность",
    "Итераций", "Невязка |f(x*)|"
])
print("\nИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
print(df_results.to_string(index=False))

# Вывод
print("\nВЫВОД")
print("Корень уравнения: x ≈ 0.7390851332")
print("Все методы сходятся к одному корню.")
print("При eps=10^-5 требуется больше итераций, чем при eps=10^-3.")
print("Метод Ньютона показывает наиболее быструю сходимость.")
print("Результаты Python, SciPy и Excel совпадают в пределах точности.")
