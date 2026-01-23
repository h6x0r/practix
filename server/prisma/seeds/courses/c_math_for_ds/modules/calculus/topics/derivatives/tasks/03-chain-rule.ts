import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-chain-rule',
	title: 'Chain Rule',
	difficulty: 'medium',
	tags: ['python', 'math', 'calculus', 'chain-rule', 'numpy'],
	estimatedTime: '20m',
	isPremium: true,
	order: 3,
	description: `# Chain Rule

Implement the chain rule for composite functions: d/dx[f(g(x))] = f'(g(x)) * g'(x)

## Background

The chain rule is fundamental to backpropagation:
- Forward: compute f(g(x))
- Backward: multiply gradients

## Example

\`\`\`python
# f(x) = (x²)³, let g(x) = x², f(u) = u³
# d/dx = 3(x²)² * 2x = 6x⁵
chain_rule_derivative(f, g, df, dg, x=2)  # 6 * 32 = 192
\`\`\`
`,
	initialCode: `import numpy as np
from typing import Callable

def chain_rule_derivative(
    outer_derivative: Callable[[float], float],
    inner_func: Callable[[float], float],
    inner_derivative: Callable[[float], float],
    x: float
) -> float:
    """
    Compute derivative using chain rule: f'(g(x)) * g'(x)

    Args:
        outer_derivative: f'(u)
        inner_func: g(x)
        inner_derivative: g'(x)
        x: Input value

    Returns:
        Derivative of f(g(x)) at x
    """
    # Your code here
    pass
`,
	solutionCode: `import numpy as np
from typing import Callable

def chain_rule_derivative(
    outer_derivative: Callable[[float], float],
    inner_func: Callable[[float], float],
    inner_derivative: Callable[[float], float],
    x: float
) -> float:
    """
    Compute derivative using chain rule: f'(g(x)) * g'(x)

    Args:
        outer_derivative: f'(u)
        inner_func: g(x)
        inner_derivative: g'(x)
        x: Input value

    Returns:
        Derivative of f(g(x)) at x
    """
    return outer_derivative(inner_func(x)) * inner_derivative(x)
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

class TestChainRule(unittest.TestCase):
    def test_x_squared_cubed(self):
        outer_deriv = lambda u: 3 * u**2  # d/du[u³] = 3u²
        inner = lambda x: x**2  # g(x) = x²
        inner_deriv = lambda x: 2*x  # g'(x) = 2x
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, 2.0)
        assert_close(result, 192.0)  # 3*(4)² * 4 = 192

    def test_sin_squared(self):
        outer_deriv = lambda u: 2*u  # d/du[u²]
        inner = np.sin
        inner_deriv = np.cos
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, 0.0)
        assert_close(result, 0.0)

    def test_exp_of_x_squared(self):
        outer_deriv = np.exp  # d/du[e^u] = e^u
        inner = lambda x: x**2
        inner_deriv = lambda x: 2*x
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, 1.0)
        assert_close(result, 2*np.e)

    def test_linear_composition(self):
        outer_deriv = lambda u: 2  # d/du[2u]
        inner = lambda x: 3*x  # 3x
        inner_deriv = lambda x: 3
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, 5.0)
        assert_close(result, 6.0)

    def test_sqrt_of_x_plus_1(self):
        outer_deriv = lambda u: 0.5 / np.sqrt(u)  # d/du[√u]
        inner = lambda x: x + 1
        inner_deriv = lambda x: 1
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, 3.0)
        assert_close(result, 0.25)

    def test_negative_x(self):
        outer_deriv = lambda u: 2*u
        inner = lambda x: x**2
        inner_deriv = lambda x: 2*x
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, -2.0)
        assert_close(result, -32.0)

    def test_at_zero(self):
        outer_deriv = lambda u: 1
        inner = lambda x: x
        inner_deriv = lambda x: 1
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, 0.0)
        assert_close(result, 1.0)

    def test_cube_root_composition(self):
        outer_deriv = lambda u: 3*u**2
        inner = lambda x: x**(1/3)
        inner_deriv = lambda x: (1/3)*x**(-2/3)
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, 8.0)
        assert_close(result, 1.0)

    def test_cosine_of_2x(self):
        outer_deriv = lambda u: -np.sin(u)  # d/du[cos(u)]
        inner = lambda x: 2*x
        inner_deriv = lambda x: 2
        result = chain_rule_derivative(outer_deriv, inner, inner_deriv, 0.0)
        assert_close(result, 0.0)

    def test_returns_float(self):
        result = chain_rule_derivative(lambda u: u, lambda x: x, lambda x: 1, 1.0)
        assert isinstance(result, float), f"Expected float, got {type(result).__name__}"

`,
	hint1: 'Chain rule: multiply outer derivative evaluated at inner function by inner derivative.',
	hint2: 'result = outer_derivative(inner_func(x)) * inner_derivative(x)',
	whyItMatters: `The chain rule IS backpropagation. When gradients flow backward through a neural network, each layer applies the chain rule. **Production Pattern:** Understanding chain rule helps debug gradient flow issues and vanishing/exploding gradients.`,
	translations: {
		ru: {
			title: 'Цепное правило',
			description: `# Цепное правило

Реализуйте цепное правило для композиции функций: d/dx[f(g(x))] = f'(g(x)) * g'(x)

## Теория

Цепное правило - основа обратного распространения:
- Прямой проход: вычислить f(g(x))
- Обратный проход: перемножить градиенты
`,
			hint1: 'Цепное правило: умножьте внешнюю производную от внутренней функции на внутреннюю производную.',
			hint2: 'result = outer_derivative(inner_func(x)) * inner_derivative(x)',
			whyItMatters: `Цепное правило - ЭТО обратное распространение. Когда градиенты текут обратно через нейросеть, каждый слой применяет цепное правило.`,
		},
		uz: {
			title: 'Zanjir qoidasi',
			description: `# Zanjir qoidasi

Kompozit funksiyalar uchun zanjir qoidasini amalga oshiring.
`,
			hint1: 'Zanjir qoidasi: ichki funksiyadan tashqi hosilani ichki hosilaga ko\'paytiring.',
			hint2: 'result = outer_derivative(inner_func(x)) * inner_derivative(x)',
			whyItMatters: `Zanjir qoidasi orqaga tarqalishdir. Gradientlar neyron tarmoq orqali orqaga oqayotganda har bir qatlam zanjir qoidasini qo'llaydi.`,
		},
	},
};

export default task;
