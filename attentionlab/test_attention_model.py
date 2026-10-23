import unittest
import numpy as np
from attention_model import AttentionModel, Config, softmax


class AttentionTests(unittest.TestCase):
    def model(self):
        return AttentionModel('o gato vê o pássaro na janela',
                              Config(d_model=4, context=6, heads=2, layers=2, d_key=3, d_value=2))

    def test_softmax_and_causal_mask(self):
        np.testing.assert_allclose(softmax(np.array([[10000., 10001.]] )).sum(axis=1), 1)
        m = self.model()
        for layer in m.cache:
            np.testing.assert_allclose(layer['a'].sum(axis=-1), 1, atol=1e-14)
            for head in layer['a']:
                np.testing.assert_array_equal(head[np.triu_indices(len(m.tokens), 1)], 0)

    def test_all_gradients_against_finite_differences(self):
        m = self.model()
        pairs = [(p[name], m.grads[i][name].copy())
                 for i, p in enumerate(m.params) for name in p]
        pairs.append((m.readout, m.readout_grad.copy()))
        epsilon = 1e-5
        for param, analytic in pairs:
            numeric = np.zeros_like(param)
            for index in np.ndindex(param.shape):
                old = param[index]
                param[index] = old + epsilon
                plus = m.forward()
                param[index] = old - epsilon
                minus = m.forward()
                param[index] = old
                numeric[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(analytic, numeric, atol=2e-9, rtol=2e-5)

    def test_sgd_updates_every_layer(self):
        m = self.model()
        before = [{k: v.copy() for k, v in p.items()} for p in m.params]
        grads = [{k: v.copy() for k, v in g.items()} for g in m.grads]
        readout, readout_grad = m.readout.copy(), m.readout_grad.copy()
        m.train_step(0.1)
        for i, p in enumerate(m.params):
            for k in p:
                np.testing.assert_allclose(p[k], before[i][k] - 0.1 * grads[i][k])
                np.testing.assert_allclose(m.last_update[i][k], -0.1 * grads[i][k])
                self.assertGreater(np.linalg.norm(m.last_update[i][k]), 0)
        np.testing.assert_allclose(m.readout, readout - 0.1 * readout_grad)
        self.assertEqual(m.step_number, 1)

    def test_training_reduces_loss(self):
        m = AttentionModel('o gato observa o pássaro e o pássaro observa o gato na janela')
        initial = m.loss
        for _ in range(300):
            m.train_step(0.1)
        self.assertLess(m.loss, initial * 0.5)

    def test_future_tokens_do_not_change_earlier_outputs(self):
        m = self.model()
        m.forward()
        before = m.probs.copy()
        m.input[-1] += 100
        m.forward()
        np.testing.assert_allclose(m.probs[:-1], before[:-1], atol=1e-14)

    def test_window_shapes_and_validation(self):
        m = AttentionModel('a b c d e', Config(context=2, d_model=7, heads=3, d_key=2, d_value=4))
        self.assertEqual(m.tokens, ['a', 'b'])
        self.assertEqual(m.targets, ['b', 'c'])
        self.assertEqual(m.params[0]['Wo'].shape, (12, 7))
        for text, cfg in [('a', Config()), ('a b c', Config(d_key=0))]:
            with self.assertRaises(ValueError):
                AttentionModel(text, cfg)
        for rate in [0, -1, float('nan'), 3]:
            with self.assertRaises(ValueError):
                m.train_step(rate)


if __name__ == '__main__':
    unittest.main()
