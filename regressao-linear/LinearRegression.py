import matplotlib

# Abra uma janela interativa, inclusive ao executar pelo PyCharm.
# Esta configuração precisa vir antes de importar pyplot.
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

xs = [1, 2, 3, 4, 5, 6, 7, 8]
ys = [2.9, 3.4, 4.9, 4.7, 6.2, 6.9, 7.3, 8.6]
w, b = 0.0, 0.0
lr = 0.01

# Guarde cada passo do gradiente descendente para exibi-lo na animação.
historico = [(w, b)]
for epoch in range(1000):
    dw, db = 0.0, 0.0
    for x, y in zip(xs, ys):
        pred = w * x + b
        err = pred - y
        dw += 2 * err * x / len(xs)
        db += 2 * err / len(ys)
    w -= lr * dw
    b -= lr * db
    historico.append((w, b))

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(xs, ys, color='blue', label='Dados observados')
reta, = ax.plot(xs, [0.0] * len(xs), color='red', label='Regressão linear')
ax.set(xlabel='x', ylabel='y', xlim=(0, 9), ylim=(-1, 10))
ax.grid(True, alpha=0.3)
ax.legend()
titulo = ax.set_title('Gradiente descendente')
fig.tight_layout()


def atualizar(epoca):
    peso, intercepto = historico[epoca]
    previstos = [peso * x + intercepto for x in xs]
    mse = sum((pred - y) ** 2 for pred, y in zip(previstos, ys)) / len(xs)
    reta.set_ydata(previstos)
    titulo.set_text(
        f'Época {epoca} | y = {peso:.2f}x + {intercepto:.2f} | MSE = {mse:.4f}'
    )
    return reta, titulo


# Mantenha a animação em uma variável para ela continuar ativa.
animacao = FuncAnimation(
    fig,
    atualizar,
    frames=range(len(historico)),
    init_func=lambda: atualizar(0),
    interval=30,  # Milissegundos entre passos; aumente para desacelerar.
    repeat=False,
    blit=False,
)
plt.show(block=True)
