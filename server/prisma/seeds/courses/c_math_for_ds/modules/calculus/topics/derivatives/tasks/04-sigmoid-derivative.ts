import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-sigmoid-derivative',
	title: 'Sigmoid Derivative',
	difficulty: 'medium',
	tags: ['python', 'math', 'calculus', 'sigmoid', 'activation', 'numpy'],
	estimatedTime: '15m',
	isPremium: true,
	order: 4,
	description: `# Sigmoid Derivative

Implement the sigmoid function and its derivative.

## Background

Sigmoid: σ(x) = 1 / (1 + e^(-x))
Derivative: σ'(x) = σ(x) * (1 - σ(x))

The beautiful property: the derivative can be expressed in terms of the function itself!
`,
	initialCode: `import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    # Your code here
    pass

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)
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

class TestSigmoid(unittest.TestCase):
    def test_sigmoid_at_zero(self):
        result = sigmoid(np.array([0.0]))
        assert_close(result[0], 0.5)

    def test_sigmoid_large_positive(self):
        result = sigmoid(np.array([100.0]))
        assert_close(result[0], 1.0)

    def test_sigmoid_large_negative(self):
        result = sigmoid(np.array([-100.0]))
        assert_close(result[0], 0.0)

    def test_sigmoid_range(self):
        x = np.linspace(-10, 10, 100)
        result = sigmoid(x)
        assert np.all(result > 0 and np.all(result < 1))

    def test_derivative_at_zero(self):
        result = sigmoid_derivative(np.array([0.0]))
        assert_close(result[0], 0.25)

    def test_derivative_at_extremes(self):
        result = sigmoid_derivative(np.array([100.0, -100.0]))
        assert_array_close(result, [0.0, 0.0])

    def test_derivative_formula(self):
        x = np.array([1.0, 2.0, -1.0])
        s = sigmoid(x)
        expected = s * (1 - s)
        result = sigmoid_derivative(x)
        assert_array_close(result, expected)

    def test_derivative_symmetry(self):
        result_pos = sigmoid_derivative(np.array([2.0]))
        result_neg = sigmoid_derivative(np.array([-2.0]))
        assert_close(result_pos[0], result_neg[0])

    def test_returns_array(self):
        result = sigmoid(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

    def test_derivative_max_at_zero(self):
        x = np.linspace(-5, 5, 100)
        derivs = sigmoid_derivative(x)
        max_idx = np.argmax(derivs)
        assert 45 < max_idx < 55

`,
	hint1: 'Sigmoid: 1 / (1 + np.exp(-x))',
	hint2: 'Derivative uses the property: σ\'(x) = σ(x) * (1 - σ(x))',
	whyItMatters: `Sigmoid was THE activation function before ReLU. Understanding its derivative explains vanishing gradients: when σ(x) ≈ 0 or 1, σ'(x) ≈ 0. **Production Pattern:** Still used in binary classification output layers and LSTM gates.`,
	translations: {
		ru: {
			title: 'Производная сигмоиды',
			description: `# Производная сигмоиды

Реализуйте сигмоиду и её производную.

## Теория

Сигмоида: σ(x) = 1 / (1 + e^(-x))
Производная: σ'(x) = σ(x) * (1 - σ(x))
`,
			hint1: 'Сигмоида: 1 / (1 + np.exp(-x))',
			hint2: 'Производная использует свойство: σ\'(x) = σ(x) * (1 - σ(x))',
			whyItMatters: `Сигмоида была ГЛАВНОЙ функцией активации до ReLU. Её производная объясняет затухающие градиенты. **Production Pattern:** Используется в выходных слоях бинарной классификации и LSTM.`,
		},
		uz: {
			title: 'Sigmoid hosilasi',
			description: `# Sigmoid hosilasi

Sigmoid funksiyasi va uning hosilasini amalga oshiring.
`,
			hint1: 'Sigmoid: 1 / (1 + np.exp(-x))',
			hint2: 'Hosila xususiyatdan foydalanadi: σ\'(x) = σ(x) * (1 - σ(x))',
			whyItMatters: `Sigmoid ReLU dan oldin asosiy aktivatsiya funksiyasi edi. LSTM eshiklarida va ikkilik klassifikatsiya chiqish qatlamlarida ishlatiladi.`,
		},
	},
};

export default task;
