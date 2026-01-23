import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-lr-decay',
	title: 'Learning Rate Decay',
	difficulty: 'medium',
	tags: ['python', 'math', 'optimization', 'learning-rate', 'numpy'],
	estimatedTime: '20m',
	isPremium: true,
	order: 5,
	description: `# Learning Rate Decay

Implement gradient descent with decaying learning rate.

## Schedules

**Step decay**: lr = lr₀ × decay^(epoch / step_size)
**Exponential**: lr = lr₀ × e^(-decay × epoch)
**1/t decay**: lr = lr₀ / (1 + decay × epoch)

Start with large steps, then fine-tune with smaller steps!
`,
	initialCode: `import numpy as np
from typing import Callable, Tuple

def gradient_descent_with_decay(
    gradient_func: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    initial_lr: float = 0.1,
    decay: float = 0.01,
    num_iterations: int = 100
) -> Tuple[np.ndarray, list]:
    """Gradient descent with 1/t learning rate decay."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np
from typing import Callable, Tuple

def gradient_descent_with_decay(
    gradient_func: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    initial_lr: float = 0.1,
    decay: float = 0.01,
    num_iterations: int = 100
) -> Tuple[np.ndarray, list]:
    """Gradient descent with 1/t learning rate decay."""
    x = x_init.copy()
    history = [x.copy()]

    for i in range(num_iterations):
        lr = initial_lr / (1 + decay * i)
        grad = gradient_func(x)
        x = x - lr * grad
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

class TestLRDecay(unittest.TestCase):
    def test_converges(self):
        grad = lambda x: 2 * x
        x_init = np.array([5.0])
        x_final, _ = gradient_descent_with_decay(grad, x_init, 0.5, 0.1, 200)
        assert x_final[0]  < 0.5

    def test_returns_history(self):
        grad = lambda x: x
        x_init = np.array([1.0])
        _, history = gradient_descent_with_decay(grad, x_init, 0.1, 0.01, 10)
        assert len(history) == 11, f"Expected 11, got {len(history)}"

    def test_learning_rate_decreases(self):
        grad = lambda x: np.array([1.0])
        x_init = np.array([0.0])
        _, history = gradient_descent_with_decay(grad, x_init, 1.0, 1.0, 5)
        steps = [abs(history[i+1][0] - history[i][0]) for i in range(len(history)-1)]
        for i in range(1, len(steps)):
            assert steps[i] < steps[i-1] + 0.01

    def test_2d_convergence(self):
        grad = lambda x: 2 * x
        x_init = np.array([3.0, 4.0])
        x_final, _ = gradient_descent_with_decay(grad, x_init, 0.5, 0.05, 200)
        assert np.linalg.norm(x_final < 1.0)

    def test_zero_decay(self):
        grad = lambda x: 2 * x
        x_init = np.array([1.0])
        x_final, _ = gradient_descent_with_decay(grad, x_init, 0.1, 0.0, 50)
        assert x_final[0]  < 0.1

    def test_high_decay(self):
        grad = lambda x: 2 * x
        x_init = np.array([5.0])
        x_final, _ = gradient_descent_with_decay(grad, x_init, 0.5, 1.0, 100)
        self.assertIsNotNone(x_final)

    def test_preserves_shape(self):
        grad = lambda x: x
        x_init = np.array([1, 2, 3.0])
        x_final, _ = gradient_descent_with_decay(grad, x_init, 0.1, 0.01, 10)
        assert x_final.shape == x_init.shape, f"Expected x_init.shape, got {x_final.shape}"

    def test_returns_tuple(self):
        grad = lambda x: x
        result = gradient_descent_with_decay(grad, np.array([1.0]), 0.1, 0.01, 1)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result).__name__}"

    def test_initial_lr_used(self):
        grad = lambda x: np.array([1.0])
        x_init = np.array([0.0])
        _, history = gradient_descent_with_decay(grad, x_init, 1.0, 0.0, 1)
        self.assertAlmostEqual(history[1][0], -1.0)

    def test_stability(self):
        grad = lambda x: 2 * x
        x_init = np.array([10.0])
        x_final, _ = gradient_descent_with_decay(grad, x_init, 0.5, 0.01, 500)
        self.assertFalse(np.isnan(x_final[0]))
        self.assertFalse(np.isinf(x_final[0]))

`,
	hint1: 'Update learning rate each iteration: lr = initial_lr / (1 + decay * iteration)',
	hint2: 'Large initial lr takes big steps, decay reduces it for fine-tuning near minimum.',
	whyItMatters: `Learning rate scheduling is crucial for training. Too high LR = divergence, too low = slow. Decay helps get best of both. **Production Pattern:** Cosine annealing, warm restarts, and OneCycleLR are advanced schedules used in production.`,
	translations: {
		ru: {
			title: 'Затухание скорости обучения',
			description: `# Затухание скорости обучения

Реализуйте градиентный спуск с затухающей скоростью обучения.

## Расписания

**1/t затухание**: lr = lr₀ / (1 + decay × epoch)

Начните с больших шагов, затем точная настройка малыми!
`,
			hint1: 'Обновляйте LR каждую итерацию: lr = initial_lr / (1 + decay * i)',
			hint2: 'Большой начальный lr делает большие шаги, затухание уменьшает для точной настройки.',
			whyItMatters: `Планирование скорости обучения критично. Слишком высокий LR = расхождение, слишком низкий = медленно. **Production Pattern:** Cosine annealing используется в production.`,
		},
		uz: {
			title: 'O\'rganish tezligini pasaytirish',
			description: `# O'rganish tezligini pasaytirish

Pasayuvchi o'rganish tezligi bilan gradient tushishini amalga oshiring.
`,
			hint1: 'Har bir iteratsiyada LR ni yangilang: lr = initial_lr / (1 + decay * i)',
			hint2: 'Katta boshlang\'ich lr katta qadamlar qiladi, pasayish minimumga yaqinlashishda kamaytiradi.',
			whyItMatters: `O'rganish tezligini rejalashtirish muhim. Juda yuqori LR = ajralish, juda past = sekin.`,
		},
	},
};

export default task;
