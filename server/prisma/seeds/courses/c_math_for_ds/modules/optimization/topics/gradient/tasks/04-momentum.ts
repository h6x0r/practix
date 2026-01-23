import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-momentum',
	title: 'Momentum Optimization',
	difficulty: 'hard',
	tags: ['python', 'math', 'optimization', 'momentum', 'numpy'],
	estimatedTime: '25m',
	isPremium: true,
	order: 4,
	description: `# Gradient Descent with Momentum

Implement momentum optimization that accelerates convergence.

## Algorithm

\`\`\`
v = momentum * v + learning_rate * gradient
x = x - v
\`\`\`

Momentum helps escape local minima and smooths noisy gradients.
`,
	initialCode: `import numpy as np
from typing import Callable, Tuple

def gradient_descent_momentum(
    gradient_func: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    num_iterations: int = 100
) -> Tuple[np.ndarray, list]:
    """Gradient descent with momentum."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np
from typing import Callable, Tuple

def gradient_descent_momentum(
    gradient_func: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    num_iterations: int = 100
) -> Tuple[np.ndarray, list]:
    """Gradient descent with momentum."""
    x = x_init.copy()
    v = np.zeros_like(x)
    history = [x.copy()]

    for _ in range(num_iterations):
        grad = gradient_func(x)
        v = momentum * v + learning_rate * grad
        x = x - v
        history.append(x.copy())

    return x, history
`,
	testCode: `import unittest
import numpy as np

def assert_array_close(actual, expected, msg=""):
    """Helper for array comparison with clear error message"""
    if not np.allclose(actual, expected):
        raise AssertionError(f"Expected {expected.tolist()}, got {actual.tolist()}")

def assert_close(actual, expected, places=5, msg=""):
    """Helper for scalar comparison with clear error message"""
    if abs(actual - expected) > 10**(-places):
        raise AssertionError(f"Expected {expected}, got {actual}")

class TestMomentum(unittest.TestCase):
    def test_converges_quadratic(self):
        grad = lambda x: 2 * x
        x_init = np.array([5.0])
        x_final, _ = gradient_descent_momentum(grad, x_init, 0.1, 0.9, 100)
        assert x_final[0]  < 0.1

    def test_faster_than_vanilla(self):
        grad = lambda x: 2 * x
        x_init = np.array([10.0])
        x_mom, _ = gradient_descent_momentum(grad, x_init, 0.01, 0.9, 50)
        # Compare with vanilla-like behavior (momentum=0)
        x_vanilla, _ = gradient_descent_momentum(grad, x_init, 0.01, 0.0, 50)
        assert abs(x_mom[0]) < abs(x_vanilla[0])

    def test_returns_history(self):
        grad = lambda x: x
        x_init = np.array([1.0])
        _, history = gradient_descent_momentum(grad, x_init, 0.1, 0.9, 10)
        assert len(history) == 11, f"Expected 11, got {len(history)}"

    def test_2d_convergence(self):
        grad = lambda x: 2 * x
        x_init = np.array([3.0, 4.0])
        x_final, _ = gradient_descent_momentum(grad, x_init, 0.1, 0.9, 100)
        assert np.linalg.norm(x_final < 0.5)

    def test_zero_momentum(self):
        grad = lambda x: 2 * x
        x_init = np.array([1.0])
        x_final, _ = gradient_descent_momentum(grad, x_init, 0.1, 0.0, 50)
        assert x_final[0]  < 0.1

    def test_preserves_shape(self):
        grad = lambda x: x
        x_init = np.array([1, 2, 3, 4, 5.0])
        x_final, _ = gradient_descent_momentum(grad, x_init, 0.1, 0.9, 10)
        assert x_final.shape == x_init.shape, f"Expected x_init.shape, got {x_final.shape}"

    def test_returns_tuple(self):
        grad = lambda x: x
        result = gradient_descent_momentum(grad, np.array([1.0]), 0.1, 0.9, 1)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result).__name__}"

    def test_negative_gradient(self):
        grad = lambda x: -2 * x
        x_init = np.array([1.0])
        x_final, _ = gradient_descent_momentum(grad, x_init, 0.1, 0.9, 50)
        assert x_final[0] > 1.0

    def test_history_changes(self):
        grad = lambda x: 2 * x
        x_init = np.array([5.0])
        _, history = gradient_descent_momentum(grad, x_init, 0.1, 0.9, 10)
        self.assertFalse(np.allclose(history[0], history[-1]))

    def test_high_momentum(self):
        grad = lambda x: 2 * x
        x_init = np.array([5.0])
        x_final, _ = gradient_descent_momentum(grad, x_init, 0.01, 0.99, 200)
        assert x_final[0]  < 1.0

`,
	hint1: 'Initialize velocity v to zeros. Update: v = momentum * v + lr * grad, then x = x - v',
	hint2: 'Momentum accumulates past gradients, acting like a ball rolling downhill.',
	whyItMatters: `Momentum is in almost every neural network optimizer. It smooths gradient updates and helps escape saddle points. **Production Pattern:** Adam optimizer combines momentum with adaptive learning rates - it's the default in most frameworks.`,
	translations: {
		ru: {
			title: 'Оптимизация с моментом',
			description: `# Градиентный спуск с моментом

Реализуйте оптимизацию с моментом, ускоряющую сходимость.

## Алгоритм

\`\`\`
v = momentum * v + learning_rate * gradient
x = x - v
\`\`\`
`,
			hint1: 'Инициализируйте скорость v нулями. Обновление: v = momentum * v + lr * grad',
			hint2: 'Момент накапливает прошлые градиенты, как мяч, катящийся с горы.',
			whyItMatters: `Момент есть почти в каждом оптимизаторе нейросетей. **Production Pattern:** Adam сочетает момент с адаптивной скоростью обучения - это стандарт во всех фреймворках.`,
		},
		uz: {
			title: 'Momentum bilan optimizatsiya',
			description: `# Momentum bilan gradient tushishi

Yaqinlashishni tezlashtiradigan momentum optimizatsiyasini amalga oshiring.
`,
			hint1: 'Tezlik v ni nolga boshlang. Yangilash: v = momentum * v + lr * grad',
			hint2: 'Momentum o\'tgan gradientlarni to\'playdi, tepadan dumalab tushayotgan sharday.',
			whyItMatters: `Momentum deyarli har bir neyron tarmoq optimizatorida mavjud. Adam optimizatori momentum va adaptiv o'rganish tezligini birlashtiradi.`,
		},
	},
};

export default task;
