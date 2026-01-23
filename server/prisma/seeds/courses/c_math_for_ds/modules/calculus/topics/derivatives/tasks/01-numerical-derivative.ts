import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-numerical-derivative',
	title: 'Numerical Derivative',
	difficulty: 'easy',
	tags: ['python', 'math', 'calculus', 'derivatives', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Numerical Derivative

Implement numerical derivative using finite differences.

## Background

The derivative measures rate of change. Numerically:
\`\`\`
f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
\`\`\`

This is the central difference formula (more accurate than forward/backward).

## Example

\`\`\`python
f = lambda x: x**2
derivative(f, 3.0)  # ≈ 6.0 (exact: 2x = 6)
\`\`\`
`,
	initialCode: `import numpy as np
from typing import Callable

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Compute numerical derivative using central difference.

    Args:
        f: Function to differentiate
        x: Point at which to compute derivative
        h: Step size (default 1e-5)

    Returns:
        Approximate derivative f'(x)
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np
from typing import Callable

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Compute numerical derivative using central difference.

    Args:
        f: Function to differentiate
        x: Point at which to compute derivative
        h: Step size (default 1e-5)

    Returns:
        Approximate derivative f'(x)
    """
    return (f(x + h) - f(x - h)) / (2 * h)
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

class TestNumericalDerivative(unittest.TestCase):
    def test_x_squared(self):
        f = lambda x: x**2
        result = numerical_derivative(f, 3.0)
        assert_close(result, 6.0)

    def test_constant(self):
        f = lambda x: 5.0
        result = numerical_derivative(f, 2.0)
        assert_close(result, 0.0)

    def test_linear(self):
        f = lambda x: 3*x + 2
        result = numerical_derivative(f, 5.0)
        assert_close(result, 3.0)

    def test_cubic(self):
        f = lambda x: x**3
        result = numerical_derivative(f, 2.0)
        assert_close(result, 12.0)

    def test_sin(self):
        f = np.sin
        result = numerical_derivative(f, 0.0)
        assert_close(result, 1.0)

    def test_cos(self):
        f = np.cos
        result = numerical_derivative(f, 0.0)
        assert_close(result, 0.0)

    def test_exp(self):
        f = np.exp
        result = numerical_derivative(f, 0.0)
        assert_close(result, 1.0)

    def test_negative_x(self):
        f = lambda x: x**2
        result = numerical_derivative(f, -3.0)
        assert_close(result, -6.0)

    def test_sqrt(self):
        f = np.sqrt
        result = numerical_derivative(f, 4.0)
        assert_close(result, 0.25)

    def test_returns_float(self):
        f = lambda x: x**2
        result = numerical_derivative(f, 1.0)
        assert isinstance(result, float), f"Expected float, got {type(result).__name__}"

`,
	hint1: 'Central difference: (f(x+h) - f(x-h)) / (2*h)',
	hint2: 'This formula is more accurate than forward difference (f(x+h) - f(x)) / h',
	whyItMatters: `Numerical derivatives are used to verify analytical gradients. When implementing backprop, you can use numerical derivatives to check your gradient computation. **Production Pattern:** Gradient checking is a debugging technique - compare analytical gradients with numerical ones.`,
	translations: {
		ru: {
			title: 'Численная производная',
			description: `# Численная производная

Реализуйте численную производную методом конечных разностей.

## Теория

Производная измеряет скорость изменения. Численно:
\`\`\`
f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
\`\`\`
`,
			hint1: 'Центральная разность: (f(x+h) - f(x-h)) / (2*h)',
			hint2: 'Эта формула точнее, чем прямая разность',
			whyItMatters: `Численные производные используются для проверки аналитических градиентов. **Production Pattern:** Gradient checking - техника отладки градиентов.`,
		},
		uz: {
			title: 'Raqamli hosila',
			description: `# Raqamli hosila

Chekli farqlar usuli yordamida raqamli hosilani amalga oshiring.
`,
			hint1: 'Markaziy farq: (f(x+h) - f(x-h)) / (2*h)',
			hint2: 'Bu formula oldinga farqdan aniqroq',
			whyItMatters: `Raqamli hosilalar analitik gradientlarni tekshirish uchun ishlatiladi.`,
		},
	},
};

export default task;
