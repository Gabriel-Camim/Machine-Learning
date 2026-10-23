# Regressão linear com gradiente descendente

Visualização animada do ajuste de uma reta `y = wx + b`, usando gradiente descendente implementado diretamente em Python. O gráfico mostra as épocas, os parâmetros e o erro quadrático médio.

## Executar

Requer Python 3.10+ com Tkinter. No Windows, Tkinter acompanha o instalador oficial do Python.

```sh
python -m pip install -r requirements.txt
python LinearRegression.py
```

São realizados 1.000 passos com taxa de aprendizado 0,01. A animação mostra cada passo a cada 30 ms.
