# Estudos de Machine Learning

Projetos de estudo em Python para visualizar regressão linear, gradiente descendente e mecanismos de atenção.

| Projeto | Conteúdo | Executar |
| --- | --- | --- |
| [Regressão linear](regressao-linear/) | Animação do ajuste de uma reta com gradiente descendente | `python LinearRegression.py` |
| [AttentionLab](attentionlab/) | Laboratório visual de self-attention causal multi-head, softmax e backpropagation em NumPy | `python main.py` |

## Preparar o ambiente

Requer Python 3.10+ com Tkinter. Execute na pasta do projeto escolhido:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Depois, execute o arquivo indicado na tabela. Consulte o README de cada projeto para detalhes.

## Testes do AttentionLab

Na pasta `attentionlab`:

```sh
python -m unittest -v test_attention_model.py
```

Os testes conferem gradientes numéricos, máscara causal, atualização SGD e redução da perda. O modelo de atenção é didático: usa embeddings fixos, conexão residual e tanh; não implementa um Transformer completo.
