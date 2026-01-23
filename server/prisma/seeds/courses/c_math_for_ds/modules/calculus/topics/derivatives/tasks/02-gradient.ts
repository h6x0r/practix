import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-gradient',
	title: 'Gradient Computation',
	difficulty: 'medium',
	tags: ['python', 'math', 'calculus', 'gradient', 'numpy'],
	estimatedTime: '20m',
	isPremium: false,
	order: 2,
	description: `# Gradient Computation

Implement gradient (vector of partial derivatives) for a multivariate function.

## Background

For f(x₁, x₂, ..., xₙ), the gradient is:
\`\`\`
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
\`\`\`

The gradient points in the direction of steepest ascent.

## Example

\`\`\`python
f = lambda x: x[0]**2 + x[1]**2
gradient(f, [3, 4])  # [6, 8]
\`\`\`
`,
	initialCode: `import numpy as np
from typing import Callable

def compute_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute gradient of f at point x using numerical differentiation.

    Args:
        f: Multivariate function
        x: Point at which to compute gradient
        h: Step size

    Returns:
        Gradient vector
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np
from typing import Callable

def compute_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute gradient of f at point x using numerical differentiation.

    Args:
        f: Multivariate function
        x: Point at which to compute gradient
        h: Step size

    Returns:
        Gradient vector
    """
    x = np.array(x, dtype=float)
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        gradient[i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return gradient
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

class TestGradient(unittest.TestCase):
    def test_quadratic(self):
        f = lambda x: x[0]**2 + x[1]**2
        result = compute_gradient(f, np.array([3.0, 4.0]))
        assert_array_close(result, [6.0, 8.0])

    def test_linear(self):
        f = lambda x: 2*x[0] + 3*x[1]
        result = compute_gradient(f, np.array([1.0, 1.0]))
        assert_array_close(result, [2.0, 3.0])

    def test_constant(self):
        f = lambda x: 5.0
        result = compute_gradient(f, np.array([1.0, 2.0]))
        assert_array_close(result, [0.0, 0.0])

    def test_3d(self):
        f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
        result = compute_gradient(f, np.array([1.0, 2.0, 3.0]))
        assert_array_close(result, [2.0, 4.0, 6.0])

    def test_mixed_terms(self):
        f = lambda x: x[0] * x[1]
        result = compute_gradient(f, np.array([3.0, 4.0]))
        assert_array_close(result, [4.0, 3.0])

    def test_at_origin(self):
        f = lambda x: x[0]**2 + x[1]**2
        result = compute_gradient(f, np.array([0.0, 0.0]))
        assert_array_close(result, [0.0, 0.0])

    def test_negative_values(self):
        f = lambda x: x[0]**2 + x[1]**2
        result = compute_gradient(f, np.array([-2.0, -3.0]))
        assert_array_close(result, [-4.0, -6.0])

    def test_returns_numpy_array(self):
        f = lambda x: x[0]**2
        result = compute_gradient(f, np.array([1.0]))
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

    def test_gradient_length(self):
        f = lambda x: sum(x**2)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_gradient(f, x)
        assert len(result) == len(x, f"Expected len(x, got {len(result)}")

    def test_rosenbrock_partial(self):
        f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        result = compute_gradient(f, np.array([1.0, 1.0]))
        assert_array_close(result, [0.0, 0.0])

`,
	hint1: 'Loop through each dimension and compute partial derivative using central difference.',
	hint2: 'For each dimension i, perturb x[i] by ±h and apply central difference formula.',
	whyItMatters: `Gradient computation IS backpropagation. Neural networks learn by following the negative gradient of the loss function. **Production Pattern:** PyTorch and TensorFlow use automatic differentiation (autograd) to compute gradients efficiently.`,
	translations: {
		ru: {
			title: 'Вычисление градиента',
			description: `# Вычисление градиента

Реализуйте градиент (вектор частных производных) для многомерной функции.

## Теория

Для f(x₁, x₂, ..., xₙ) градиент:
\`\`\`
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
\`\`\`

Градиент указывает направление наискорейшего подъёма.
`,
			hint1: 'Пройдите по каждому измерению и вычислите частную производную.',
			hint2: 'Для каждого i изменяйте x[i] на ±h и применяйте формулу центральной разности.',
			whyItMatters: `Вычисление градиента - ЭТО обратное распространение. Нейросети обучаются, следуя отрицательному градиенту функции потерь. **Production Pattern:** PyTorch и TensorFlow используют автодифференцирование.`,
		},
		uz: {
			title: 'Gradientni hisoblash',
			description: `# Gradientni hisoblash

Ko'p o'zgaruvchili funksiya uchun gradient (qisman hosilalar vektori) ni amalga oshiring.
`,
			hint1: 'Har bir o\'lchov bo\'ylab aylanib, qisman hosila hisoblang.',
			hint2: 'Har bir i uchun x[i] ni ±h ga o\'zgartiring va markaziy farq formulasini qo\'llang.',
			whyItMatters: `Gradient hisoblash orqaga tarqalishdir. Neyron tarmoqlar yo'qotish funksiyasining manfiy gradientiga rioya qilib o'rganadi.`,
		},
	},
};

export default task;
