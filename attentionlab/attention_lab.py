"""Interface Tkinter do laboratório de attention; sem dependências de ML."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from attention_model import AttentionModel, Config


class AttentionLab:
    def __init__(self, root):
        self.root = root
        root.title('Laboratório de Attention • Softmax e Gradiente Descendente')
        root.geometry('1400x940')
        root.minsize(1050, 740)
        self.running = False
        self.timer = None
        self.model = None
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=6)
        sidebar = ttk.Frame(root, padding=12, width=260)
        sidebar.pack(side='left', fill='y')
        ttk.Label(sidebar, text='ATTENTION LAB', font=('Segoe UI', 16, 'bold')).pack(anchor='w')
        ttk.Label(sidebar, text='Aprender vendo cada atualização', padding=(0, 0, 0, 14)).pack(anchor='w')
        ttk.Label(sidebar, text='Texto • tokens separados por espaços').pack(anchor='w')
        self.text = tk.Text(sidebar, width=31, height=5, wrap='word')
        self.text.insert('1.0', 'o gato observa o pássaro e o pássaro observa o gato na janela')
        self.text.pack(fill='x', pady=5)
        self.fields = {}
        specs = [('d_model', 'Dimensão do modelo', 16, 2, 64),
                 ('context', 'Janela máxima (tokens)', 12, 2, 64),
                 ('heads', 'Attention heads', 2, 1, 8),
                 ('layers', 'Camadas', 2, 1, 6),
                 ('d_key', 'Dimensão Q / K por head', 8, 1, 32),
                 ('d_value', 'Dimensão V por head', 8, 1, 32),
                 ('seed', 'Semente aleatória', 42, 0, 999999)]
        for name, label, value, lo, hi in specs:
            row = ttk.Frame(sidebar)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label).pack(side='left')
            var = tk.StringVar(value=str(value))
            self.fields[name] = var
            ttk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=6).pack(side='right')
        ttk.Button(sidebar, text='Aplicar configuração / Reiniciar', command=self.reset).pack(fill='x', pady=10)
        ttk.Separator(sidebar).pack(fill='x', pady=5)
        self.lr = tk.StringVar(value='0.1')
        self.delay = tk.StringVar(value='250')
        self.limit = tk.StringVar(value='1000')
        for label, var in [('Learning rate (SGD)', self.lr), ('Intervalo (ms, ≥ 50)', self.delay),
                           ('Limite total de passos', self.limit)]:
            row = ttk.Frame(sidebar)
            row.pack(fill='x', pady=4)
            ttk.Label(row, text=label).pack(side='left')
            ttk.Entry(row, textvariable=var, width=8).pack(side='right')
        self.start_button = ttk.Button(sidebar, text='▶ Iniciar treinamento', command=self.toggle)
        self.start_button.pack(fill='x', pady=(12, 4))
        ttk.Button(sidebar, text='Um passo de SGD', command=self.single_step).pack(fill='x')
        ttk.Label(sidebar, wraplength=250, justify='left', text=(
            'W ← W − η · ∂Loss/∂W\n\n'
            'Tarefa: prever o próximo token em uma janela fixa do texto. '
            'Embeddings e posições são fixos; Wq, Wk, Wv, Wo e a projeção de saída são treináveis.\n\n'
            'Modelo didático: atenção causal + conexão residual + tanh. Sem MLP ou LayerNorm. '
            'As dimensões Q/K e V são livres; Wo retorna à dimensão do modelo.'
        )).pack(fill='x', pady=18)
        body = ttk.Frame(root, padding=(0, 10, 10, 10))
        body.pack(side='left', fill='both', expand=True)
        controls = ttk.Frame(body)
        controls.pack(fill='x')
        self.layer = tk.StringVar(value='1')
        self.head = tk.StringVar(value='1')
        self.selectors = []
        for label, var in [('Camada', self.layer), ('Head', self.head)]:
            ttk.Label(controls, text=label).pack(side='left', padx=5)
            cb = ttk.Combobox(controls, textvariable=var, state='readonly', width=5)
            cb.pack(side='left')
            cb.bind('<<ComboboxSelected>>', lambda _: self.render())
            self.selectors.append(cb)
        self.status = ttk.Label(controls, font=('Segoe UI', 11, 'bold'))
        self.status.pack(side='right', padx=10)
        self.shapes = ttk.Label(body)
        self.shapes.pack(anchor='w', pady=7)
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill='both', expand=True)
        self.figures, self.canvases = [], []
        for title in ['1 · Fluxo de attention', '2 · Pesos e gradientes', '3 · Treinamento']:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            fig = Figure(figsize=(10, 6), dpi=90, layout='constrained')
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill='both', expand=True)
            canvas.mpl_connect('motion_notify_event', self.hover)
            self.figures.append(fig)
            self.canvases.append(canvas)
        self.notebook.bind('<<NotebookTabChanged>>', lambda _: self.render())
        self.detail = tk.Text(body, height=7, font=('Consolas', 10), wrap='word', state='disabled')
        self.detail.pack(fill='x', pady=(7, 0))
        self.hover_label = ttk.Label(body, text='Passe o mouse sobre uma célula para ler seu valor exato.')
        self.hover_label.pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.reset()

    def pause(self):
        self.running = False
        self.start_button.configure(text='▶ Iniciar / Continuar')
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def reset(self):
        self.pause()
        try:
            config = Config(**{k: int(v.get()) for k, v in self.fields.items()})
            model = AttentionModel(self.text.get('1.0', 'end'), config)
        except (ValueError, OverflowError) as exc:
            messagebox.showerror('Configuração inválida', str(exc))
            return
        self.model = model
        self.losses = [model.loss]
        self.grad_norms = [self.norms()]
        for selector, count in zip(self.selectors, [config.layers, config.heads]):
            selector.configure(values=list(range(1, count + 1)))
            selector.set('1')
        self.render()

    def norms(self):
        return [float(np.sqrt(sum(np.sum(g[k] ** 2) for k in g))) for g in self.model.grads]

    def training_settings(self):
        lr = float(self.lr.get().replace(',', '.'))
        delay, limit = int(self.delay.get()), int(self.limit.get())
        if not np.isfinite(lr) or not 0 < lr <= 2 or not 50 <= delay <= 60000 or not 1 <= limit <= 100000:
            raise ValueError('Use taxa em (0, 2], intervalo de 50 a 60000 ms e limite de 1 a 100000.')
        return lr, delay, limit

    def toggle(self):
        if self.running:
            self.pause()
        else:
            self.running = True
            self.start_button.configure(text='Ⅱ Pausar treinamento')
            self.tick()

    def single_step(self):
        self.pause()
        self.tick(single=True)

    def tick(self, single=False):
        self.timer = None
        try:
            lr, delay, limit = self.training_settings()
            if self.model.step_number >= limit:
                self.pause()
                self.status.configure(text=f'Limite atingido · passo {limit} · loss {self.model.loss:.5f}')
                return
            self.model.train_step(lr)
            self.losses.append(self.model.loss)
            self.grad_norms.append(self.norms())
            self.render()
            if self.running and not single:
                self.timer = self.root.after(delay, self.tick)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            self.pause()
            messagebox.showerror('Treinamento', str(exc))

    def heatmap(self, ax, matrix, title, probability=False, token_axes=False):
        matrix = np.asarray(matrix)
        finite = matrix[np.isfinite(matrix)]
        bound = max(float(np.max(np.abs(finite))) if finite.size else 1, 1e-10)
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', interpolation='nearest',
                       cmap='viridis' if probability else 'coolwarm',
                       vmin=0 if probability else -bound, vmax=1 if probability else bound)
        ax._matrix = matrix
        ax.set_title(f'{title}  {matrix.shape}', fontsize=10)
        if token_axes:
            n = len(self.model.tokens)
            stride = max(1, int(np.ceil(n / 12)))
            ticks = list(range(0, n, stride))
            labels = [f'{i}: {self.model.tokens[i][:9]}' for i in ticks]
            ax.set_xticks(ticks, labels, rotation=55, ha='right', fontsize=7)
            ax.set_yticks(ticks, labels, fontsize=7)
            ax.set_xlabel('Keys · tokens consultados', fontsize=8)
            ax.set_ylabel('Queries · tokens que consultam', fontsize=8)
        else:
            ax.tick_params(labelsize=7)
        if matrix.size <= 64:
            for (r, col), value in np.ndenumerate(matrix):
                ax.text(col, r, f'{value:.2f}' if np.isfinite(value) else '−∞',
                        ha='center', va='center', fontsize=6)
        ax.figure.colorbar(im, ax=ax, shrink=0.7, pad=0.02).ax.tick_params(labelsize=7)

    def render(self):
        if self.model is None:
            return
        m = self.model
        i, h = int(self.layer.get()) - 1, int(self.head.get()) - 1
        c, b, p = m.config, m.cache[i], m.params[i]
        accuracy = np.mean(m.probs.argmax(axis=1) == m.target_ids)
        self.status.configure(text=f'Passo {m.step_number} · loss {m.loss:.5f} · acerto {accuracy:.0%}')
        self.shapes.configure(text=(f'Tokens: {len(m.tokens)}/{c.context} · Wq/Wk por head: {c.d_model}×{c.d_key} · '
                                   f'Wv: {c.d_model}×{c.d_value} · Wo: {c.heads*c.d_value}×{c.d_model}'))
        tab = self.notebook.index(self.notebook.select())
        fig = self.figures[tab]
        fig.clear()
        if tab == 0:
            axes = fig.subplots(2, 3).flat
            items = [(b['q'][h], 'Q = X Wq', False, False),
                     (b['k'][h], 'K = X Wk', False, False),
                     (b['v'][h], 'V = X Wv', False, False),
                     (b['scores'][h], 'S = Q Kᵀ / √d_key + máscara', False, True),
                     (b['a'][h], 'A = softmax(S) por linha', True, True),
                     (b['z'][h], 'Saída da cabeça = A V', False, False)]
            for ax, (mat, title, prob, tokens) in zip(axes, items):
                self.heatmap(ax, mat, title, prob, tokens)
        elif tab == 1:
            axes = fig.subplots(2, 4).flat
            grads = m.applied_grads[i] if m.step_number else m.grads[i]
            items = [(p['Wq'][h], 'Wq atual'), (p['Wk'][h], 'Wk atual'),
                     (p['Wv'][h], 'Wv atual'), (p['Wo'], 'Wo atual (todas as heads)'),
                     (grads['Wq'][h], '∂Loss/∂Wq usado no passo'),
                     (grads['Wk'][h], '∂Loss/∂Wk usado no passo'),
                     (m.last_update[i]['Wq'][h], 'ΔWq do último passo'),
                     (m.last_update[i]['Wk'][h], 'ΔWk do último passo')]
            for ax, (mat, title) in zip(axes, items):
                self.heatmap(ax, mat, title)
        else:
            ax_loss, ax_grad = fig.subplots(2, 1)
            ax_loss.plot(self.losses, color='#2563eb')
            ax_loss.set(title='Cross-entropy · prever o próximo token', xlabel='Passo SGD', ylabel='Loss')
            norms = np.asarray(self.grad_norms)
            for layer in range(c.layers):
                ax_grad.plot(norms[:, layer], label=f'Camada {layer+1}')
            ax_grad.set(title='Norma dos gradientes atuais por camada', xlabel='Passo SGD', ylabel='Norma L2')
            ax_grad.legend(ncol=3, fontsize=8)
            for ax in [ax_loss, ax_grad]:
                ax.grid(alpha=0.2)
        fig.suptitle(f'Camada {i+1} / {c.layers} · Head {h+1} / {c.heads}', fontsize=12)
        self.canvases[tab].draw_idle()
        a = b['a'][h]
        lines = [f'Softmax: cada linha soma 1 (min={a.sum(axis=1).min():.6f}; max={a.sum(axis=1).max():.6f}). '
                 'Células futuras: mascaradas com −∞ antes da softmax → probabilidade 0.',
                 'Todas as camadas são atualizadas a cada passo. Gradientes atuais e variação do último passo:']
        for layer in range(c.layers):
            g, delta = m.grads[layer], m.last_update[layer]
            lines.append(f"  L{layer+1}: ||∇Wq||={np.linalg.norm(g['Wq']):.5f}  "
                         f"||∇Wk||={np.linalg.norm(g['Wk']):.5f}  "
                         f"||ΔWq||={np.linalg.norm(delta['Wq']):.5f}  ||ΔWk||={np.linalg.norm(delta['Wk']):.5f}")
        lines.append('Previsões (entrada → alvo / previsto / probabilidade):')
        for j, token in enumerate(m.tokens):
            pred = int(m.probs[j].argmax())
            lines.append(f'  {j}: {token} → {m.targets[j]} / {m.vocab[pred]} / {m.probs[j, pred]:.1%}')
        self.detail.configure(state='normal')
        self.detail.delete('1.0', 'end')
        self.detail.insert('1.0', '\n'.join(lines))
        self.detail.configure(state='disabled')

    def hover(self, event):
        ax = event.inaxes
        if ax is None or not hasattr(ax, '_matrix') or event.xdata is None or event.ydata is None:
            return
        r, col = int(np.floor(event.ydata + 0.5)), int(np.floor(event.xdata + 0.5))
        mat = ax._matrix
        if 0 <= r < mat.shape[0] and 0 <= col < mat.shape[1]:
            self.hover_label.configure(text=f'{ax.get_title()} · linha {r}, coluna {col} = {mat[r, col]:.9g}')

    def close(self):
        self.pause()
        self.root.destroy()


def main():
    root = tk.Tk()
    AttentionLab(root)
    root.mainloop()


if __name__ == '__main__':
    main()
