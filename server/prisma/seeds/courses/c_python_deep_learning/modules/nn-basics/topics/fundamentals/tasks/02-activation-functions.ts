import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-activation-functions',
	title: 'Activation Functions',
	difficulty: 'easy',
	tags: ['numpy', 'activation', 'neural-networks'],
	estimatedTime: '12m',
	isPremium: false,
	order: 2,
	description: `# Activation Functions

Implement common activation functions and their derivatives.

## Task

Implement six functions:
1. \`sigmoid(x)\` - Sigmoid activation
2. \`sigmoid_derivative(x)\` - Derivative of sigmoid
3. \`relu(x)\` - ReLU activation
4. \`relu_derivative(x)\` - Derivative of ReLU
5. \`tanh(x)\` - Hyperbolic tangent
6. \`softmax(x)\` - Softmax for multi-class output

## Example

\`\`\`python
x = np.array([-1, 0, 1, 2])

sigmoid(x)  # [0.27, 0.5, 0.73, 0.88]
relu(x)     # [0, 0, 1, 2]
\`\`\``,

	initialCode: `import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    # Your code here
    pass

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
    # Your code here
    pass

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)"""
    # Your code here
    pass

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0"""
    # Your code here
    pass

def tanh_activation(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation"""
    # Your code here
    pass

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax for multi-class. Handle numerical stability."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return (x > 0).astype(float)

def tanh_activation(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation"""
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax for multi-class. Handle numerical stability."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
`,

	testCode: `import numpy as np
import unittest

class TestActivationFunctions(unittest.TestCase):
    def test_sigmoid_range(self):
        x = np.array([-10, 0, 10])
        result = sigmoid(x)
        self.assertTrue(all(0 < r < 1 for r in result))

    def test_sigmoid_at_zero(self):
        result = sigmoid(np.array([0.0]))
        self.assertAlmostEqual(result[0], 0.5, places=5)

    def test_relu_negative(self):
        x = np.array([-5, -1, 0])
        result = relu(x)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_relu_positive(self):
        x = np.array([1, 2, 3])
        result = relu(x)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_tanh_range(self):
        x = np.array([-10, 0, 10])
        result = tanh_activation(x)
        self.assertTrue(all(-1 <= r <= 1 for r in result))

    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        self.assertAlmostEqual(result.sum(), 1.0, places=5)

    def test_sigmoid_derivative_at_zero(self):
        result = sigmoid_derivative(np.array([0.0]))
        self.assertAlmostEqual(result[0], 0.25, places=5)

    def test_relu_derivative_negative(self):
        x = np.array([-5, -1])
        result = relu_derivative(x)
        np.testing.assert_array_equal(result, [0, 0])

    def test_relu_derivative_positive(self):
        x = np.array([1, 2, 3])
        result = relu_derivative(x)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_tanh_at_zero(self):
        result = tanh_activation(np.array([0.0]))
        self.assertAlmostEqual(result[0], 0.0, places=5)
`,

	hint1: 'sigmoid = 1 / (1 + np.exp(-x)), use np.clip to avoid overflow',
	hint2: 'softmax: subtract max for stability before exp, then normalize',

	whyItMatters: `Activation functions are crucial because:

- **Non-linearity**: Enable learning complex patterns
- **Gradient flow**: Derivatives needed for backpropagation
- **Output normalization**: Softmax for probabilities
- **Architecture choice**: Different activations for different layers

Core building blocks of every neural network.`,

	translations: {
		ru: {
			title: 'Функции активации',
			description: `# Функции активации

Реализуйте популярные функции активации и их производные.

## Задача

Реализуйте шесть функций:
1. \`sigmoid(x)\` - Сигмоида
2. \`sigmoid_derivative(x)\` - Производная сигмоиды
3. \`relu(x)\` - ReLU
4. \`relu_derivative(x)\` - Производная ReLU
5. \`tanh(x)\` - Гиперболический тангенс
6. \`softmax(x)\` - Softmax для многоклассовой классификации

## Пример

\`\`\`python
x = np.array([-1, 0, 1, 2])

sigmoid(x)  # [0.27, 0.5, 0.73, 0.88]
relu(x)     # [0, 0, 1, 2]
\`\`\``,
			hint1: 'sigmoid = 1 / (1 + np.exp(-x)), используйте np.clip для избежания переполнения',
			hint2: 'softmax: вычтите max для стабильности перед exp, затем нормализуйте',
			whyItMatters: `Функции активации важны потому что:

- **Нелинейность**: Позволяют обучать сложные паттерны
- **Поток градиентов**: Производные нужны для обратного распространения
- **Нормализация выхода**: Softmax для вероятностей`,
		},
		uz: {
			title: 'Aktivatsiya funksiyalari',
			description: `# Aktivatsiya funksiyalari

Ommabop aktivatsiya funksiyalari va ularning hosilalarini amalga oshiring.

## Topshiriq

Oltita funksiyani amalga oshiring:
1. \`sigmoid(x)\` - Sigmoid aktivatsiyasi
2. \`sigmoid_derivative(x)\` - Sigmoid hosilasi
3. \`relu(x)\` - ReLU aktivatsiyasi
4. \`relu_derivative(x)\` - ReLU hosilasi
5. \`tanh(x)\` - Giperbolik tangens
6. \`softmax(x)\` - Ko'p sinfli klassifikatsiya uchun Softmax

## Misol

\`\`\`python
x = np.array([-1, 0, 1, 2])

sigmoid(x)  # [0.27, 0.5, 0.73, 0.88]
relu(x)     # [0, 0, 1, 2]
\`\`\``,
			hint1: "sigmoid = 1 / (1 + np.exp(-x)), overflow dan qochish uchun np.clip ishlating",
			hint2: "softmax: exp dan oldin max ni ayiring barqarorlik uchun, keyin normallang",
			whyItMatters: `Aktivatsiya funksiyalari muhim chunki:

- **Nochiziqlilik**: Murakkab naqshlarni o'rganishga imkon beradi
- **Gradient oqimi**: Orqaga tarqalish uchun hosilalar kerak
- **Chiqish normallash**: Ehtimolliklar uchun Softmax`,
		},
	},
};

export default task;
