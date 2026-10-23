# Laboratório de Attention

Interface de desktop em português para explorar self-attention causal e SGD.

## Executar

No PyCharm, execute `main.py`. Ou, no terminal:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Requer Python 3.10+ com Tkinter (incluído no instalador oficial para Windows).

1. Informe um texto com pelo menos três palavras. Cada palavra separada por espaço é um token; pontuação permanece junto da palavra.
2. Ajuste dimensões, janela, cabeças e camadas. Clique em **Aplicar configuração / Reiniciar**. Isso descarta o treinamento anterior.
3. Clique em **Iniciar treinamento** ou **Um passo de SGD**. Taxa e intervalo podem mudar durante o treinamento. O limite é o total de passos desde a reinicialização.
4. Selecione camada e cabeça. Explore as abas de fluxo, pesos/gradientes e curvas. Passe o mouse sobre as matrizes para consultar valores exatos. Role a área de texto inferior para consultar todas as previsões.

## Matemática e dimensões

Para cada camada e cabeça:

```text
X: [tokens, d_model]
Wq, Wk: [d_model, d_key]     Wv: [d_model, d_value]
Q = X @ Wq; K = X @ Wk; V = X @ Wv
S = Q @ K.T / sqrt(d_key)
S[i,j] = -infinito quando j > i
A = softmax(S, por linha)
Z = A @ V
Wo: [heads * d_value, d_model]
X_seguinte = tanh((X + concat(Z_heads) @ Wo) / sqrt(2))
logits = X_final @ W_saida
Loss = cross_entropy(logits, proximo_token)
W_novo = W_anterior - learning_rate * gradiente
```

Q e K têm a mesma dimensão para permitir o produto escalar. A dimensão V pode ser independente. Como as projeções por cabeça são explícitas e Wo reconecta a saída, `d_model` não precisa ser divisível pelo número de cabeças.

Embeddings aleatórios e posições senoidais ficam fixos. Wq, Wk, Wv, Wo de **todas** as camadas e W_saida são treinados com backpropagation analítica em NumPy. A aba de pesos exibe pesos atuais, gradientes **usados na última atualização** e deltas reais; antes do primeiro passo, os deltas são zero. As normas no resumo e nas curvas são dos gradientes calculados nos pesos atuais, para o próximo passo.

A janela usa os primeiros `context + 1` tokens disponíveis: até `context` entradas e seus respectivos próximos tokens como alvos. O treinamento repete essa janela fixa; a loss mede memorização desse exemplo, sem medir generalização. Este é um modelo didático com residual e tanh, sem MLP ou LayerNorm; não é um Transformer completo.

As cores dos pesos são reescaladas em cada quadro; a softmax usa sempre escala de 0 a 1. Cada atualização fica visível, sem passos de treinamento ocultos. O intervalo é uma espera mínima após cálculo e desenho; configurações maiores podem reduzir a velocidade.

## Verificação

```powershell
python -m unittest -v test_attention_model.py
```

Os testes comparam a backpropagation com diferenças finitas, verificam a máscara causal, a atualização SGD e a redução da loss.
