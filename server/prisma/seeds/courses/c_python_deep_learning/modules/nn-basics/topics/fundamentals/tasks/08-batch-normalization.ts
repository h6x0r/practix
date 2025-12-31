import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-batch-normalization',
	title: 'Batch Normalization',
	difficulty: 'hard',
	tags: ['numpy', 'batchnorm', 'neural-networks'],
	estimatedTime: '18m',
	isPremium: true,
	order: 8,
	description: `# Batch Normalization

Implement batch normalization for faster and more stable training.

## Task

Implement three functions:
1. \`batchnorm_forward(x, gamma, beta, eps)\` - Forward pass of batch norm
2. \`batchnorm_backward(dout, cache)\` - Backward pass for gradients
3. \`running_mean_update(running_mean, batch_mean, momentum)\` - Update running statistics

## Example

\`\`\`python
gamma = np.ones(hidden_size)  # Scale parameter
beta = np.zeros(hidden_size)  # Shift parameter

# Forward pass
out, cache = batchnorm_forward(x, gamma, beta, eps=1e-5)

# Backward pass
dx, dgamma, dbeta = batchnorm_backward(dout, cache)
\`\`\``,

	initialCode: `import numpy as np

def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      eps: float = 1e-5) -> tuple:
    """Batch normalization forward. Return (output, cache for backward)."""
    # Your code here
    pass

def batchnorm_backward(dout: np.ndarray, cache: dict) -> tuple:
    """Batch normalization backward. Return (dx, dgamma, dbeta)."""
    # Your code here
    pass

def running_mean_update(running_mean: np.ndarray, batch_mean: np.ndarray,
                        momentum: float = 0.9) -> np.ndarray:
    """Update running mean for inference. Return updated running_mean."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      eps: float = 1e-5) -> tuple:
    """Batch normalization forward. Return (output, cache for backward)."""
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = {
        'x': x, 'x_norm': x_norm, 'mean': mean, 'var': var,
        'gamma': gamma, 'eps': eps
    }
    return out, cache

def batchnorm_backward(dout: np.ndarray, cache: dict) -> tuple:
    """Batch normalization backward. Return (dx, dgamma, dbeta)."""
    x, x_norm, mean, var, gamma, eps = (
        cache['x'], cache['x_norm'], cache['mean'],
        cache['var'], cache['gamma'], cache['eps']
    )
    m = x.shape[0]

    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps) ** (-1.5), axis=0)
    dmean = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mean) / m + dmean / m

    return dx, dgamma, dbeta

def running_mean_update(running_mean: np.ndarray, batch_mean: np.ndarray,
                        momentum: float = 0.9) -> np.ndarray:
    """Update running mean for inference. Return updated running_mean."""
    return momentum * running_mean + (1 - momentum) * batch_mean
`,

	testCode: `import numpy as np
import unittest

class TestBatchNormalization(unittest.TestCase):
    def test_forward_shape(self):
        x = np.random.randn(32, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)
        out, cache = batchnorm_forward(x, gamma, beta)
        self.assertEqual(out.shape, x.shape)

    def test_forward_normalized(self):
        x = np.random.randn(32, 64) * 5 + 10
        gamma = np.ones(64)
        beta = np.zeros(64)
        out, cache = batchnorm_forward(x, gamma, beta)
        self.assertAlmostEqual(np.mean(out), 0, places=1)
        self.assertAlmostEqual(np.std(out), 1, places=1)

    def test_backward_shapes(self):
        x = np.random.randn(32, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)
        out, cache = batchnorm_forward(x, gamma, beta)
        dout = np.random.randn(32, 64)
        dx, dgamma, dbeta = batchnorm_backward(dout, cache)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(dgamma.shape, gamma.shape)

    def test_running_mean(self):
        running = np.zeros(64)
        batch = np.ones(64)
        updated = running_mean_update(running, batch, momentum=0.9)
        np.testing.assert_array_almost_equal(updated, np.ones(64) * 0.1)

    def test_forward_returns_tuple(self):
        x = np.random.randn(32, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)
        result = batchnorm_forward(x, gamma, beta)
        self.assertEqual(len(result), 2)

    def test_forward_cache_keys(self):
        x = np.random.randn(16, 32)
        gamma = np.ones(32)
        beta = np.zeros(32)
        out, cache = batchnorm_forward(x, gamma, beta)
        self.assertIn('x', cache)
        self.assertIn('gamma', cache)

    def test_backward_returns_three(self):
        x = np.random.randn(16, 32)
        gamma = np.ones(32)
        beta = np.zeros(32)
        out, cache = batchnorm_forward(x, gamma, beta)
        dout = np.random.randn(16, 32)
        result = batchnorm_backward(dout, cache)
        self.assertEqual(len(result), 3)

    def test_running_mean_shape(self):
        running = np.zeros(128)
        batch = np.random.randn(128)
        updated = running_mean_update(running, batch, momentum=0.9)
        self.assertEqual(updated.shape, running.shape)

    def test_dbeta_shape(self):
        x = np.random.randn(16, 32)
        gamma = np.ones(32)
        beta = np.zeros(32)
        out, cache = batchnorm_forward(x, gamma, beta)
        dout = np.random.randn(16, 32)
        dx, dgamma, dbeta = batchnorm_backward(dout, cache)
        self.assertEqual(dbeta.shape, beta.shape)
`,

	hint1: 'Normalize: x_norm = (x - mean) / sqrt(var + eps), then scale and shift',
	hint2: 'Running stats update: running = momentum * running + (1-momentum) * batch',

	whyItMatters: `Batch normalization revolutionized deep learning:

- **Faster training**: Allows higher learning rates
- **Regularization**: Acts as mild regularizer
- **Internal covariate shift**: Stabilizes layer inputs
- **Universal**: Used in almost all modern architectures

One of the most important techniques in deep learning.`,

	translations: {
		ru: {
			title: 'Пакетная нормализация',
			description: `# Пакетная нормализация

Реализуйте пакетную нормализацию для более быстрого и стабильного обучения.

## Задача

Реализуйте три функции:
1. \`batchnorm_forward(x, gamma, beta, eps)\` - Прямой проход batch norm
2. \`batchnorm_backward(dout, cache)\` - Обратный проход для градиентов
3. \`running_mean_update(running_mean, batch_mean, momentum)\` - Обновление статистик

## Пример

\`\`\`python
gamma = np.ones(hidden_size)  # Scale parameter
beta = np.zeros(hidden_size)  # Shift parameter

# Forward pass
out, cache = batchnorm_forward(x, gamma, beta, eps=1e-5)

# Backward pass
dx, dgamma, dbeta = batchnorm_backward(dout, cache)
\`\`\``,
			hint1: 'Нормализация: x_norm = (x - mean) / sqrt(var + eps), затем масштаб и сдвиг',
			hint2: 'Обновление статистик: running = momentum * running + (1-momentum) * batch',
			whyItMatters: `Пакетная нормализация изменила глубокое обучение:

- **Быстрое обучение**: Позволяет использовать высокие learning rate
- **Регуляризация**: Действует как мягкий регуляризатор
- **Внутренний ковариатный сдвиг**: Стабилизирует входы слоев`,
		},
		uz: {
			title: 'Batch normalizatsiya',
			description: `# Batch normalizatsiya

Tezroq va barqarorroq o'qitish uchun batch normalizatsiyani amalga oshiring.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`batchnorm_forward(x, gamma, beta, eps)\` - Batch norm oldinga o'tishi
2. \`batchnorm_backward(dout, cache)\` - Gradientlar uchun orqaga o'tish
3. \`running_mean_update(running_mean, batch_mean, momentum)\` - Ishlaydigan statistikalarni yangilash

## Misol

\`\`\`python
gamma = np.ones(hidden_size)  # Scale parameter
beta = np.zeros(hidden_size)  # Shift parameter

# Forward pass
out, cache = batchnorm_forward(x, gamma, beta, eps=1e-5)

# Backward pass
dx, dgamma, dbeta = batchnorm_backward(dout, cache)
\`\`\``,
			hint1: "Normallash: x_norm = (x - mean) / sqrt(var + eps), keyin masshtab va siljitish",
			hint2: "Ishlaydigan statistikalar yangilanishi: running = momentum * running + (1-momentum) * batch",
			whyItMatters: `Batch normalizatsiya chuqur o'rganishni inqilob qildi:

- **Tezroq o'qitish**: Yuqoriroq o'rganish tezliklariga ruxsat beradi
- **Regulyarizatsiya**: Yengil regulyarizator sifatida ishlaydi
- **Ichki kovariat siljishi**: Qatlam kirishlarini barqarorlashtiradi`,
		},
	},
};

export default task;
