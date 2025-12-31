import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-gradient-descent',
	title: 'Gradient Descent Variants',
	difficulty: 'medium',
	tags: ['numpy', 'optimization', 'neural-networks'],
	estimatedTime: '15m',
	isPremium: false,
	order: 6,
	description: `# Gradient Descent Variants

Implement different gradient descent optimizers.

## Task

Implement four functions:
1. \`sgd_update(params, grads, lr)\` - Basic SGD update
2. \`momentum_update(params, grads, velocity, lr, beta)\` - SGD with momentum
3. \`adam_update(params, grads, m, v, t, lr, beta1, beta2)\` - Adam optimizer
4. \`create_batches(X, y, batch_size)\` - Create mini-batches for SGD

## Example

\`\`\`python
# Basic SGD
params = sgd_update(params, grads, lr=0.01)

# With momentum
params, velocity = momentum_update(params, grads, velocity, lr=0.01, beta=0.9)

# Adam
params, m, v = adam_update(params, grads, m, v, t=1, lr=0.001)
\`\`\``,

	initialCode: `import numpy as np

def sgd_update(params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
    """Basic SGD: params = params - lr * grads. Return updated params."""
    # Your code here
    pass

def momentum_update(params: np.ndarray, grads: np.ndarray, velocity: np.ndarray,
                    lr: float, beta: float = 0.9) -> tuple:
    """SGD with momentum. Return (updated_params, new_velocity)."""
    # Your code here
    pass

def adam_update(params: np.ndarray, grads: np.ndarray, m: np.ndarray, v: np.ndarray,
                t: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999) -> tuple:
    """Adam optimizer. Return (updated_params, new_m, new_v)."""
    # Your code here
    pass

def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> list:
    """Create mini-batches. Return list of (X_batch, y_batch) tuples."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def sgd_update(params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
    """Basic SGD: params = params - lr * grads. Return updated params."""
    return params - lr * grads

def momentum_update(params: np.ndarray, grads: np.ndarray, velocity: np.ndarray,
                    lr: float, beta: float = 0.9) -> tuple:
    """SGD with momentum. Return (updated_params, new_velocity)."""
    velocity = beta * velocity + (1 - beta) * grads
    params = params - lr * velocity
    return params, velocity

def adam_update(params: np.ndarray, grads: np.ndarray, m: np.ndarray, v: np.ndarray,
                t: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999) -> tuple:
    """Adam optimizer. Return (updated_params, new_m, new_v)."""
    epsilon = 1e-8
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return params, m, v

def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> list:
    """Create mini-batches. Return list of (X_batch, y_batch) tuples."""
    n = len(X)
    indices = np.random.permutation(n)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    batches = []
    for i in range(0, n, batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        batches.append((X_batch, y_batch))
    return batches
`,

	testCode: `import numpy as np
import unittest

class TestGradientDescent(unittest.TestCase):
    def test_sgd_update(self):
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])
        new_params = sgd_update(params, grads, lr=0.1)
        expected = np.array([0.99, 1.98])
        np.testing.assert_array_almost_equal(new_params, expected)

    def test_momentum_update(self):
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])
        velocity = np.zeros(2)
        new_params, new_v = momentum_update(params, grads, velocity, lr=0.1, beta=0.9)
        self.assertEqual(len(new_v), 2)
        self.assertFalse(np.allclose(new_params, params))

    def test_adam_update(self):
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])
        m = np.zeros(2)
        v = np.zeros(2)
        new_params, new_m, new_v = adam_update(params, grads, m, v, t=1)
        self.assertFalse(np.allclose(new_params, params))

    def test_create_batches(self):
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        batches = create_batches(X, y, batch_size=32)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0][0].shape[0], 32)

    def test_sgd_returns_numpy(self):
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])
        result = sgd_update(params, grads, lr=0.1)
        self.assertIsInstance(result, np.ndarray)

    def test_momentum_returns_tuple(self):
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])
        velocity = np.zeros(2)
        result = momentum_update(params, grads, velocity, lr=0.1)
        self.assertEqual(len(result), 2)

    def test_adam_returns_tuple(self):
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])
        m = np.zeros(2)
        v = np.zeros(2)
        result = adam_update(params, grads, m, v, t=1)
        self.assertEqual(len(result), 3)

    def test_batches_cover_all_data(self):
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        batches = create_batches(X, y, batch_size=25)
        total = sum(b[0].shape[0] for b in batches)
        self.assertEqual(total, 100)

    def test_sgd_decreases_params(self):
        params = np.array([1.0, 1.0])
        grads = np.array([1.0, 1.0])
        new_params = sgd_update(params, grads, lr=0.1)
        self.assertTrue(np.all(new_params < params))

    def test_create_batches_shuffles(self):
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        batches = create_batches(X, y, batch_size=100)
        self.assertFalse(np.all(batches[0][0].flatten() == np.arange(100)))
`,

	hint1: 'SGD: params -= lr * grads. Momentum: velocity = beta * v + (1-beta) * grads',
	hint2: 'Adam: bias correction with m/(1-beta^t) and v/(1-beta^t)',

	whyItMatters: `Optimizers determine training success:

- **Convergence speed**: Adam converges faster than vanilla SGD
- **Stability**: Momentum smooths noisy gradients
- **Adaptive learning**: Adam adjusts per-parameter rates
- **Hyperparameter sensitivity**: Different optimizers need different tuning

Choosing the right optimizer is key to efficient training.`,

	translations: {
		ru: {
			title: 'Варианты градиентного спуска',
			description: `# Варианты градиентного спуска

Реализуйте различные оптимизаторы градиентного спуска.

## Задача

Реализуйте четыре функции:
1. \`sgd_update(params, grads, lr)\` - Базовый SGD
2. \`momentum_update(params, grads, velocity, lr, beta)\` - SGD с моментумом
3. \`adam_update(params, grads, m, v, t, lr, beta1, beta2)\` - Adam оптимизатор
4. \`create_batches(X, y, batch_size)\` - Создать мини-батчи

## Пример

\`\`\`python
# Basic SGD
params = sgd_update(params, grads, lr=0.01)

# With momentum
params, velocity = momentum_update(params, grads, velocity, lr=0.01, beta=0.9)

# Adam
params, m, v = adam_update(params, grads, m, v, t=1, lr=0.001)
\`\`\``,
			hint1: 'SGD: params -= lr * grads. Momentum: velocity = beta * v + (1-beta) * grads',
			hint2: 'Adam: коррекция смещения с m/(1-beta^t) и v/(1-beta^t)',
			whyItMatters: `Оптимизаторы определяют успех обучения:

- **Скорость сходимости**: Adam сходится быстрее базового SGD
- **Стабильность**: Momentum сглаживает шумные градиенты
- **Адаптивное обучение**: Adam настраивает скорость для каждого параметра`,
		},
		uz: {
			title: 'Gradient tushish variantlari',
			description: `# Gradient tushish variantlari

Turli gradient tushish optimallashtiruvchilarini amalga oshiring.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`sgd_update(params, grads, lr)\` - Asosiy SGD yangilanishi
2. \`momentum_update(params, grads, velocity, lr, beta)\` - Momentum bilan SGD
3. \`adam_update(params, grads, m, v, t, lr, beta1, beta2)\` - Adam optimallashtiruvchi
4. \`create_batches(X, y, batch_size)\` - Mini-batchlar yaratish

## Misol

\`\`\`python
# Basic SGD
params = sgd_update(params, grads, lr=0.01)

# With momentum
params, velocity = momentum_update(params, grads, velocity, lr=0.01, beta=0.9)

# Adam
params, m, v = adam_update(params, grads, m, v, t=1, lr=0.001)
\`\`\``,
			hint1: "SGD: params -= lr * grads. Momentum: velocity = beta * v + (1-beta) * grads",
			hint2: "Adam: m/(1-beta^t) va v/(1-beta^t) bilan siljish tuzatishi",
			whyItMatters: `Optimallashtiruvchilar o'qitish muvaffaqiyatini belgilaydi:

- **Yaqinlashish tezligi**: Adam oddiy SGD dan tezroq yaqinlashadi
- **Barqarorlik**: Momentum shovqinli gradientlarni tekislaydi
- **Moslashuvchan o'rganish**: Adam har bir parametr uchun tezlikni sozlaydi`,
		},
	},
};

export default task;
