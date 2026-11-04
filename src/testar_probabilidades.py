from probabilidades import (
    calcular_matriz_placares,
    calcular_probabilidades_resultado
)

placares = calcular_matriz_placares(
    gols_mandante=1.6,
    gols_visitante=0.9
)

print("Matriz de placares:")
print((placares * 100).round(2))

resultados = calcular_probabilidades_resultado(placares)

print("\nProbabilidades:")
for resultado, probabilidade in resultados.items():
    print(f"{resultado}: {probabilidade:.2%}")