"""Self-attention causal multi-head com backpropagation explícita em NumPy."""
from dataclasses import dataclass
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Config:
    d_model: int = 16
    context: int = 12
    heads: int = 2
    layers: int = 2
    d_key: int = 8
    d_value: int = 8
    seed: int = 42


class AttentionModel:
    def __init__(self, text, config=None):
        self.config = c = config or Config()
        limits = {'d_model': (2, 64), 'context': (2, 64), 'heads': (1, 8),
                  'layers': (1, 6), 'd_key': (1, 32), 'd_value': (1, 32)}
        for name, (lo, hi) in limits.items():
            if not lo <= getattr(c, name) <= hi:
                raise ValueError(f'{name} deve estar entre {lo} e {hi}.')
        words = text.split()
        if len(words) < 3:
            raise ValueError('Informe pelo menos três tokens separados por espaços.')
        words = words[:c.context + 1]
        self.vocab = list(dict.fromkeys(words))
        self.tokens, self.targets = words[:-1], words[1:]
        self.ids = np.array([self.vocab.index(w) for w in self.tokens])
        self.target_ids = np.array([self.vocab.index(w) for w in self.targets])
        rng = np.random.default_rng(c.seed)
        embedding = rng.normal(0, 0.6, (len(self.vocab), c.d_model))
        pos = np.arange(len(self.tokens))[:, None]
        dims = np.arange(c.d_model)[None, :]
        angles = pos / 10000 ** (2 * (dims // 2) / c.d_model)
        positional = np.where(dims % 2 == 0, np.sin(angles), np.cos(angles))
        self.input = embedding[self.ids] + 0.3 * positional
        self.params = []
        for _ in range(c.layers):
            self.params.append({
                'Wq': rng.normal(0, 1 / np.sqrt(c.d_model), (c.heads, c.d_model, c.d_key)),
                'Wk': rng.normal(0, 1 / np.sqrt(c.d_model), (c.heads, c.d_model, c.d_key)),
                'Wv': rng.normal(0, 1 / np.sqrt(c.d_model), (c.heads, c.d_model, c.d_value)),
                'Wo': rng.normal(0, 1 / np.sqrt(c.heads * c.d_value),
                                 (c.heads * c.d_value, c.d_model)),
            })
        self.readout = rng.normal(0, 1 / np.sqrt(c.d_model), (c.d_model, len(self.vocab)))
        self.step_number = 0
        self.last_update = [{k: np.zeros_like(v) for k, v in p.items()} for p in self.params]
        self.refresh()

    def forward(self):
        c = self.config
        x = self.input
        self.cache = []
        n = len(x)
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        for p in self.params:
            q = np.einsum('nd,hdk->hnk', x, p['Wq'])
            k = np.einsum('nd,hdk->hnk', x, p['Wk'])
            v = np.einsum('nd,hdv->hnv', x, p['Wv'])
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(c.d_key)
            a = softmax(np.where(mask[None], -np.inf, scores))
            z = a @ v
            concat = z.transpose(1, 0, 2).reshape(n, -1)
            out = np.tanh((x + concat @ p['Wo']) / np.sqrt(2))
            self.cache.append(dict(x=x, q=q, k=k, v=v, scores=np.where(mask[None], -np.inf, scores),
                                   a=a, z=z, concat=concat, out=out))
            x = out
        logits = x @ self.readout
        self.probs = softmax(logits)
        max_logits = logits.max(axis=1)
        logsum = max_logits + np.log(np.exp(logits - max_logits[:, None]).sum(axis=1))
        return float(np.mean(logsum - logits[np.arange(n), self.target_ids]))

    def backward(self):
        c = self.config
        n = len(self.tokens)
        dl = self.probs.copy()
        dl[np.arange(n), self.target_ids] -= 1
        dl /= n
        self.readout_grad = self.cache[-1]['out'].T @ dl
        dx = dl @ self.readout.T
        self.grads = [None] * c.layers
        for i in reversed(range(c.layers)):
            b, p = self.cache[i], self.params[i]
            du = dx * (1 - b['out'] ** 2) / np.sqrt(2)
            dwo = b['concat'].T @ du
            dz = (du @ p['Wo'].T).reshape(n, c.heads, c.d_value).transpose(1, 0, 2)
            da = dz @ b['v'].transpose(0, 2, 1)
            dv = b['a'].transpose(0, 2, 1) @ dz
            # Produto do Jacobiano da softmax pelo gradiente de entrada.
            ds = b['a'] * (da - (da * b['a']).sum(axis=-1, keepdims=True))
            dq = ds @ b['k'] / np.sqrt(c.d_key)
            dk = ds.transpose(0, 2, 1) @ b['q'] / np.sqrt(c.d_key)
            self.grads[i] = {
                'Wq': np.einsum('nd,hnk->hdk', b['x'], dq),
                'Wk': np.einsum('nd,hnk->hdk', b['x'], dk),
                'Wv': np.einsum('nd,hnv->hdv', b['x'], dv), 'Wo': dwo,
            }
            dx = (du + np.einsum('hnk,hdk->nd', dq, p['Wq'])
                  + np.einsum('hnk,hdk->nd', dk, p['Wk'])
                  + np.einsum('hnv,hdv->nd', dv, p['Wv']))

    def refresh(self):
        self.loss = self.forward()
        self.backward()

    def train_step(self, learning_rate):
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= 2:
            raise ValueError('A taxa de aprendizado deve ser maior que 0 e no máximo 2.')
        self.applied_grads = [{k: v.copy() for k, v in g.items()} for g in self.grads]
        for i, p in enumerate(self.params):
            for name in p:
                update = -learning_rate * self.grads[i][name]
                p[name] += update
                self.last_update[i][name] = update
        self.readout -= learning_rate * self.readout_grad
        self.step_number += 1
        self.refresh()
        return self.loss
