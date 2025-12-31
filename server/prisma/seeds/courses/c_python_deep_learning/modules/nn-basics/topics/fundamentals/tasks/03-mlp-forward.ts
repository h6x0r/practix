import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-mlp-forward',
	title: 'MLP Forward Pass',
	difficulty: 'medium',
	tags: ['numpy', 'mlp', 'neural-networks'],
	estimatedTime: '15m',
	isPremium: false,
	order: 3,
	description: `# MLP Forward Pass

Implement the forward pass of a multi-layer perceptron.

## Task

Implement three functions:
1. \`init_weights(layer_sizes)\` - Initialize weights and biases for each layer
2. \`forward_layer(x, W, b, activation)\` - Single layer forward pass
3. \`mlp_forward(x, weights, biases)\` - Full MLP forward pass

## Example

\`\`\`python
# Network: 4 inputs -> 8 hidden -> 3 outputs
layer_sizes = [4, 8, 3]
weights, biases = init_weights(layer_sizes)

x = np.random.randn(1, 4)  # One sample with 4 features
output = mlp_forward(x, weights, biases)  # Shape: (1, 3)
\`\`\``,

	initialCode: `import numpy as np

def init_weights(layer_sizes: list) -> tuple:
    """Initialize weights and biases. Return (weights_list, biases_list)."""
    # Your code here
    pass

def forward_layer(x: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str = 'relu') -> np.ndarray:
    """Single layer: z = xW + b, then apply activation. Return activations."""
    # Your code here
    pass

def mlp_forward(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """Full forward pass through all layers. Return final output."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def init_weights(layer_sizes: list) -> tuple:
    """Initialize weights and biases. Return (weights_list, biases_list)."""
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
        b = np.zeros((1, layer_sizes[i+1]))
        weights.append(W)
        biases.append(b)
    return weights, biases

def forward_layer(x: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str = 'relu') -> np.ndarray:
    """Single layer: z = xW + b, then apply activation. Return activations."""
    z = np.dot(x, W) + b
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'softmax':
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    else:
        return z

def mlp_forward(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """Full forward pass through all layers. Return final output."""
    a = x
    for i in range(len(weights) - 1):
        a = forward_layer(a, weights[i], biases[i], 'relu')
    a = forward_layer(a, weights[-1], biases[-1], 'softmax')
    return a
`,

	testCode: `import numpy as np
import unittest

class TestMLPForward(unittest.TestCase):
    def test_init_weights_count(self):
        weights, biases = init_weights([4, 8, 3])
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(biases), 2)

    def test_init_weights_shapes(self):
        weights, biases = init_weights([4, 8, 3])
        self.assertEqual(weights[0].shape, (4, 8))
        self.assertEqual(weights[1].shape, (8, 3))

    def test_forward_layer_shape(self):
        x = np.random.randn(5, 4)
        W = np.random.randn(4, 8)
        b = np.zeros((1, 8))
        out = forward_layer(x, W, b, 'relu')
        self.assertEqual(out.shape, (5, 8))

    def test_mlp_forward_shape(self):
        weights, biases = init_weights([4, 8, 3])
        x = np.random.randn(10, 4)
        out = mlp_forward(x, weights, biases)
        self.assertEqual(out.shape, (10, 3))

    def test_mlp_output_sums_to_one(self):
        weights, biases = init_weights([4, 8, 3])
        x = np.random.randn(1, 4)
        out = mlp_forward(x, weights, biases)
        self.assertAlmostEqual(out.sum(), 1.0, places=5)

    def test_biases_shape(self):
        weights, biases = init_weights([4, 8, 3])
        self.assertEqual(biases[0].shape, (1, 8))
        self.assertEqual(biases[1].shape, (1, 3))

    def test_forward_layer_relu(self):
        x = np.array([[-1, 1, -2]])
        W = np.eye(3)
        b = np.zeros((1, 3))
        out = forward_layer(x, W, b, 'relu')
        np.testing.assert_array_equal(out, [[0, 1, 0]])

    def test_forward_layer_sigmoid(self):
        x = np.array([[0]])
        W = np.array([[1]])
        b = np.array([[0]])
        out = forward_layer(x, W, b, 'sigmoid')
        self.assertAlmostEqual(out[0, 0], 0.5, places=5)

    def test_deeper_network(self):
        weights, biases = init_weights([4, 16, 8, 3])
        self.assertEqual(len(weights), 3)

    def test_mlp_output_positive(self):
        weights, biases = init_weights([4, 8, 3])
        x = np.random.randn(1, 4)
        out = mlp_forward(x, weights, biases)
        self.assertTrue(np.all(out >= 0))
`,

	hint1: 'Initialize weights with small random values: np.random.randn() * 0.01',
	hint2: 'Forward: z = np.dot(x, W) + b, then apply activation function',

	whyItMatters: `The forward pass is essential because:

- **Prediction**: How neural networks make predictions
- **Layer composition**: Understanding information flow
- **Debugging**: Verify shapes and activations
- **Foundation for backprop**: Must understand forward to compute gradients

Core algorithm behind every neural network.`,

	translations: {
		ru: {
			title: 'Прямой проход MLP',
			description: `# Прямой проход MLP

Реализуйте прямой проход многослойного перцептрона.

## Задача

Реализуйте три функции:
1. \`init_weights(layer_sizes)\` - Инициализировать веса и смещения
2. \`forward_layer(x, W, b, activation)\` - Прямой проход одного слоя
3. \`mlp_forward(x, weights, biases)\` - Полный прямой проход MLP

## Пример

\`\`\`python
# Network: 4 inputs -> 8 hidden -> 3 outputs
layer_sizes = [4, 8, 3]
weights, biases = init_weights(layer_sizes)

x = np.random.randn(1, 4)  # One sample with 4 features
output = mlp_forward(x, weights, biases)  # Shape: (1, 3)
\`\`\``,
			hint1: 'Инициализируйте веса малыми случайными значениями: np.random.randn() * 0.01',
			hint2: 'Forward: z = np.dot(x, W) + b, затем примените функцию активации',
			whyItMatters: `Прямой проход важен потому что:

- **Предсказание**: Как нейросети делают предсказания
- **Композиция слоев**: Понимание потока информации
- **Отладка**: Проверка форм и активаций`,
		},
		uz: {
			title: 'MLP oldinga o\'tish',
			description: `# MLP oldinga o'tish

Ko'p qatlamli perceptronning oldinga o'tishini amalga oshiring.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`init_weights(layer_sizes)\` - Og'irliklar va siljishlarni ishga tushirish
2. \`forward_layer(x, W, b, activation)\` - Bitta qatlam oldinga o'tishi
3. \`mlp_forward(x, weights, biases)\` - To'liq MLP oldinga o'tishi

## Misol

\`\`\`python
# Network: 4 inputs -> 8 hidden -> 3 outputs
layer_sizes = [4, 8, 3]
weights, biases = init_weights(layer_sizes)

x = np.random.randn(1, 4)  # One sample with 4 features
output = mlp_forward(x, weights, biases)  # Shape: (1, 3)
\`\`\``,
			hint1: "Og'irliklarni kichik tasodifiy qiymatlar bilan ishga tushiring: np.random.randn() * 0.01",
			hint2: "Forward: z = np.dot(x, W) + b, keyin aktivatsiya funksiyasini qo'llang",
			whyItMatters: `Oldinga o'tish muhim chunki:

- **Bashorat**: Neyrosetka bashoratlarni qanday qiladi
- **Qatlam kompozitsiyasi**: Axborot oqimini tushunish
- **Nosozliklarni tuzatish**: Shakllar va aktivatsiyalarni tekshirish`,
		},
	},
};

export default task;
