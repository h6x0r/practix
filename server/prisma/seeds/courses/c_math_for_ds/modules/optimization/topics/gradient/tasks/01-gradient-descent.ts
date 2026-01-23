import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-gradient-descent',
	title: 'Gradient Descent',
	difficulty: 'medium',
	tags: ['python', 'math', 'optimization', 'gradient-descent', 'numpy'],
	estimatedTime: '25m',
	isPremium: false,
	order: 1,
	description: `# Gradient Descent

Implement the basic gradient descent optimization algorithm.

## Algorithm

\`\`\`
Repeat until convergence:
    x = x - learning_rate * gradient(f, x)
\`\`\`

## Parameters

- **learning_rate**: Step size (too big = diverge, too small = slow)
- **num_iterations**: How many steps to take
`,
	initialCode: `import numpy as np
from typing import Callable, Tuple

def gradient_descent(
    gradient_func: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    learning_rate: float = 0.01,
    num_iterations: int = 100
) -> Tuple[np.ndarray, list]:
    """
    Gradient descent optimization.

    Returns:
        Tuple of (final_x, history of x values)
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np
from typing import Callable, Tuple

def gradient_descent(
    gradient_func: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    learning_rate: float = 0.01,
    num_iterations: int = 100
) -> Tuple[np.ndarray, list]:
    """
    Gradient descent optimization.

    Returns:
        Tuple of (final_x, history of x values)
    """
    x = x_init.copy()
    history = [x.copy()]

    for _ in range(num_iterations):
        grad = gradient_func(x)
        x = x - learning_rate * grad
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

class TestGradientDescent(unittest.TestCase):
    def test_quadratic_1d(self):
        # f(x) = x^2, gradient = 2x, minimum at x=0
        grad = lambda x: 2 * x
        x_init = np.array([5.0])
        x_final, _ = gradient_descent(grad, x_init, 0.1, 100)
        assert_close(x_final[0], 0.0)

    def test_quadratic_2d(self):
        # f(x,y) = x^2 + y^2, gradient = [2x, 2y], minimum at (0,0)
        grad = lambda x: 2 * x
        x_init = np.array([3.0, 4.0])
        x_final, _ = gradient_descent(grad, x_init, 0.1, 100)
        assert_array_close(x_final, np.array([0.0, 0.0]))

    def test_returns_history(self):
        grad = lambda x: 2 * x
        x_init = np.array([1.0])
        _, history = gradient_descent(grad, x_init, 0.1, 10)
        assert len(history) == 11, f"Expected 11, got {len(history)}"

    def test_history_converges(self):
        grad = lambda x: 2 * x
        x_init = np.array([10.0])
        _, history = gradient_descent(grad, x_init, 0.1, 50)
        # Each step should get closer to 0
        for i in range(1, len(history)):
            assert abs(history[i][0]) <= abs(history[i-1][0]) + 0.001

    def test_learning_rate_effect(self):
        grad = lambda x: 2 * x
        x_init = np.array([5.0])
        x_slow, _ = gradient_descent(grad, x_init, 0.01, 50)
        x_fast, _ = gradient_descent(grad, x_init, 0.1, 50)
        # Faster learning rate should converge more
        assert abs(x_fast[0]) < abs(x_slow[0])

    def test_preserves_shape(self):
        grad = lambda x: x
        x_init = np.array([1.0, 2.0, 3.0])
        x_final, _ = gradient_descent(grad, x_init, 0.1, 10)
        assert x_final.shape == x_init.shape, f"Expected x_init.shape, got {x_final.shape}"

    def test_linear_gradient(self):
        # f(x) = 2x, gradient = 2, minimum at -inf (but should move left)
        grad = lambda x: np.array([2.0])
        x_init = np.array([0.0])
        x_final, _ = gradient_descent(grad, x_init, 0.1, 10)
        assert x_final[0] < 0

    def test_returns_tuple(self):
        grad = lambda x: x
        result = gradient_descent(grad, np.array([1.0]), 0.1, 1)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result).__name__}"
        assert len(result) == 2, f"Expected 2, got {len(result)}"

    def test_zero_gradient(self):
        grad = lambda x: np.zeros_like(x)
        x_init = np.array([5.0])
        x_final, _ = gradient_descent(grad, x_init, 0.1, 10)
        assert_close(x_final[0], 5.0)

    def test_high_dimensional(self):
        grad = lambda x: 2 * x
        x_init = np.random.randn(10)
        x_final, _ = gradient_descent(grad, x_init, 0.1, 200)
        assert_array_close(x_final, np.zeros(10))

`,
	hint1: 'Update rule: x = x - learning_rate * gradient(x)',
	hint2: 'Store each x in history list. Start with x_init.copy() to avoid mutating input.',
	whyItMatters: `Gradient descent is HOW neural networks learn. Every deep learning framework implements variants of gradient descent. Understanding it helps debug training issues. **Production Pattern:** Modern optimizers (Adam, RMSprop) build on basic gradient descent.`,
	translations: {
		ru: {
			title: 'Градиентный спуск',
			description: `# Градиентный спуск

Реализуйте базовый алгоритм оптимизации градиентным спуском.

## Алгоритм

\`\`\`
Повторять до сходимости:
    x = x - learning_rate * gradient(f, x)
\`\`\`
`,
			hint1: 'Правило обновления: x = x - learning_rate * gradient(x)',
			hint2: 'Сохраняйте каждый x в history. Начните с x_init.copy().',
			whyItMatters: `Градиентный спуск - ТО, КАК нейросети учатся. Понимание его помогает отлаживать обучение. **Production Pattern:** Современные оптимизаторы (Adam, RMSprop) построены на градиентном спуске.`,
		},
		uz: {
			title: 'Gradient tushishi',
			description: `# Gradient tushishi

Asosiy gradient tushish optimizatsiya algoritmini amalga oshiring.
`,
			hint1: 'Yangilash qoidasi: x = x - learning_rate * gradient(x)',
			hint2: 'Har bir x ni history ga saqlang.',
			whyItMatters: `Gradient tushishi neyron tarmoqlar QANDAY o'rganishini ko'rsatadi. Uni tushunish o'qitish muammolarini tuzatishga yordam beradi.`,
		},
	},
};

export default task;
