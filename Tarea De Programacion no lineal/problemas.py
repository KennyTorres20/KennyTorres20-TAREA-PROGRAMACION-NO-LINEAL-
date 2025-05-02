import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    print("=== PROBLEMA 1: OPTIMIZACIÓN DE INVERSIONES ===")
    problema_1()
    
    print("\n=== PROBLEMA 2: MINIMIZACIÓN DE COSTOS DE PRODUCCIÓN ===")
    problema_2()
    
    print("\n=== PROBLEMA 3: MÉTODO DEL GRADIENTE DESCENDENTE ===")
    problema_3()
    
    print("\n=== PROBLEMA 4: MINIMIZACIÓN CON RESTRICCIONES DE DESIGUALDAD ===")
    problema_4()

def problema_1():
    """Optimización de inversiones con restricciones de presupuesto y riesgo"""
    # Función objetivo (rendimientos)
    def objective(x):
        return - (0.1*x[0] + 0.08*x[1])  # Negativo porque queremos maximizar

    # Restricciones
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1},  # x + y = 1
        {'type': 'ineq', 'fun': lambda x: 0.05 - (0.02*x[0]**2 + 0.03*x[1]**2)},  # riesgo ≤ 0.05
        {'type': 'ineq', 'fun': lambda x: x[0]},  # x ≥ 0
        {'type': 'ineq', 'fun': lambda x: x[1]}   # y ≥ 0
    ]

    # Punto inicial
    x0 = np.array([0.5, 0.5])

    # Resolver
    sol = minimize(objective, x0, method='SLSQP', constraints=constraints)

    # Resultados
    print("Solución:")
    print(f"x (inversión en activo 1): {sol.x[0]:.4f}")
    print(f"y (inversión en activo 2): {sol.x[1]:.4f}")
    print(f"Rendimiento máximo: {-sol.fun:.4f}")
    print(f"Riesgo: {0.02*sol.x[0]**2 + 0.03*sol.x[1]**2:.4f}")

def problema_2():
    """Minimización de costos de producción con restricción de presupuesto"""
    # Función de costos
    def cost(x):
        return 5*x[0]**2 + 3*x[1]**2 + x[2]**2

    # Restricción
    constraint = {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 100}

    # Punto inicial
    x0 = np.array([33, 33, 34])

    # Resolver
    sol = minimize(cost, x0, method='SLSQP', constraints=constraint)

    # Resultados
    print("Solución:")
    print(f"x (producto A): {sol.x[0]:.2f}")
    print(f"y (producto B): {sol.x[1]:.2f}")
    print(f"z (producto C): {sol.x[2]:.2f}")
    print(f"Costo mínimo: {sol.fun:.2f}")

    # Gráfico
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Puntos para el gráfico
    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 100, 20)
    X, Y = np.meshgrid(x, y)
    Z = 100 - X - Y  # Por la restricción x + y + z = 100

    # Calcular costos
    C = 5*X**2 + 3*Y**2 + Z**2

    # Superficie
    surf = ax.plot_surface(X, Y, C, cmap='viridis', alpha=0.6)

    # Punto solución
    ax.scatter(sol.x[0], sol.x[1], sol.fun, color='red', s=100, label='Solución óptima')

    ax.set_xlabel('Producto A (x)')
    ax.set_ylabel('Producto B (y)')
    ax.set_zlabel('Costo')
    ax.set_title('Función de Costos con Restricción x + y + z = 100')
    plt.legend()
    plt.tight_layout()
    plt.show()

def problema_3():
    """Método del gradiente descendente para minimización"""
    # Función objetivo
    def f(x, y, z):
        return x**2 + y**2 + z**2 - 2*x*y + 3*z

    # Gradiente de la función
    def gradient(x, y, z):
        df_dx = 2*x - 2*y
        df_dy = 2*y - 2*x
        df_dz = 2*z + 3
        return np.array([df_dx, df_dy, df_dz])

    # Parámetros
    alpha = 0.1  # Tasa de aprendizaje
    max_iter = 15
    point = np.array([1.0, 1.0, 1.0])  # Punto inicial

    # Almacenar valores para gráfico
    f_values = []

    print("Iteración\tx\ty\tz\tf(x,y,z)")
    for i in range(max_iter):
        f_val = f(*point)
        f_values.append(f_val)
        
        print(f"{i+1}\t\t{point[0]:.4f}\t{point[1]:.4f}\t{point[2]:.4f}\t{f_val:.4f}")
        
        # Actualizar punto
        point = point - alpha * gradient(*point)

    # Gráfico de la evolución
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_iter+1), f_values, 'bo-')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de f(x,y,z)')
    plt.title('Evolución de la función objetivo en el gradiente descendente')
    plt.grid(True)
    plt.show()

def problema_4():
    """Minimización con restricciones de desigualdad"""
    # Función objetivo
    def f(x):
        return x[0]**2 + 4*x[0] + 5

    # Restricciones
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 2},  # x ≥ 2
        {'type': 'ineq', 'fun': lambda x: 5 - x[0]}    # x ≤ 5
    ]

    # Punto inicial
    x0 = np.array([2.5])  # Punto dentro del intervalo

    # Resolver
    sol = minimize(f, x0, method='SLSQP', constraints=constraints)

    # Resultados
    print("Solución:")
    print(f"x óptimo: {sol.x[0]:.4f}")
    print(f"Valor mínimo de f(x): {sol.fun:.4f}")

    # Gráfico
    x_vals = np.linspace(0, 6, 100)
    y_vals = x_vals**2 + 4*x_vals + 5

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label='f(x) = x² + 4x + 5')
    plt.axvline(x=2, color='r', linestyle='--', label='x ≥ 2')
    plt.axvline(x=5, color='g', linestyle='--', label='x ≤ 5')
    plt.scatter(sol.x, sol.fun, color='black', s=100, label='Mínimo')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Minimización con restricciones de desigualdad')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()