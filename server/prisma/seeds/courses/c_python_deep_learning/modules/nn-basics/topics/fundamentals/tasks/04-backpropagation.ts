import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-backpropagation',
	title: 'Backpropagation',
	difficulty: 'hard',
	tags: ['numpy', 'backprop', 'neural-networks'],
	estimatedTime: '20m',
	isPremium: true,
	order: 4,
	description: `# Backpropagation

Implement backpropagation to compute gradients for a 2-layer network.

## Task

Implement three functions:
1. \`compute_loss(y_pred, y_true)\` - Cross-entropy loss
2. \`backward_pass(x, y, weights, biases, activations)\` - Compute gradients
3. \`update_weights(weights, biases, grads_w, grads_b, lr)\` - Apply gradient descent

## Example

\`\`\`python
# Forward pass (computed elsewhere)
activations = [a1, a2]  # Hidden and output activations

# Compute gradients
grads_w, grads_b = backward_pass(x, y, weights, biases, activations)

# Update weights
weights, biases = update_weights(weights, biases, grads_w, grads_b, lr=0.01)
\`\`\``,

	initialCode: `import numpy as np

def compute_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Cross-entropy loss. y_pred is softmax output, y_true is one-hot."""
    # Your code here
    pass

def backward_pass(x: np.ndarray, y: np.ndarray, weights: list, biases: list,
                  activations: list) -> tuple:
    """Compute gradients. Return (grads_w, grads_b) lists."""
    # Your code here
    pass

def update_weights(weights: list, biases: list, grads_w: list,
                   grads_b: list, lr: float) -> tuple:
    """Apply gradient descent update. Return (new_weights, new_biases)."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def compute_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Cross-entropy loss. y_pred is softmax output, y_true is one-hot."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def backward_pass(x: np.ndarray, y: np.ndarray, weights: list, biases: list,
                  activations: list) -> tuple:
    """Compute gradients. Return (grads_w, grads_b) lists."""
    m = x.shape[0]
    grads_w = []
    grads_b = []

    # Output layer gradient (softmax + cross-entropy)
    dz2 = activations[-1] - y
    dw2 = np.dot(activations[-2].T, dz2) / m
    db2 = np.mean(dz2, axis=0, keepdims=True)

    # Hidden layer gradient
    da1 = np.dot(dz2, weights[-1].T)
    dz1 = da1 * (activations[-2] > 0)  # ReLU derivative
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.mean(dz1, axis=0, keepdims=True)

    grads_w = [dw1, dw2]
    grads_b = [db1, db2]

    return grads_w, grads_b

def update_weights(weights: list, biases: list, grads_w: list,
                   grads_b: list, lr: float) -> tuple:
    """Apply gradient descent update. Return (new_weights, new_biases)."""
    new_weights = [w - lr * gw for w, gw in zip(weights, grads_w)]
    new_biases = [b - lr * gb for b, gb in zip(biases, grads_b)]
    return new_weights, new_biases
`,

	testCode: `import numpy as np
import unittest

class TestBackpropagation(unittest.TestCase):
    def test_loss_perfect_prediction(self):
        y_pred = np.array([[0.99, 0.01]])
        y_true = np.array([[1, 0]])
        loss = compute_loss(y_pred, y_true)
        self.assertLess(loss, 0.1)

    def test_loss_bad_prediction(self):
        y_pred = np.array([[0.01, 0.99]])
        y_true = np.array([[1, 0]])
        loss = compute_loss(y_pred, y_true)
        self.assertGreater(loss, 2)

    def test_backward_returns_lists(self):
        x = np.random.randn(5, 4)
        y = np.eye(3)[np.array([0, 1, 2, 0, 1])]
        weights = [np.random.randn(4, 8), np.random.randn(8, 3)]
        biases = [np.zeros((1, 8)), np.zeros((1, 3))]
        a1 = np.maximum(0, np.dot(x, weights[0]) + biases[0])
        a2 = np.exp(np.dot(a1, weights[1]) + biases[1])
        a2 = a2 / a2.sum(axis=1, keepdims=True)
        grads_w, grads_b = backward_pass(x, y, weights, biases, [a1, a2])
        self.assertEqual(len(grads_w), 2)
        self.assertEqual(len(grads_b), 2)

    def test_update_changes_weights(self):
        weights = [np.ones((4, 8)), np.ones((8, 3))]
        biases = [np.zeros((1, 8)), np.zeros((1, 3))]
        grads_w = [np.ones((4, 8)) * 0.1, np.ones((8, 3)) * 0.1]
        grads_b = [np.ones((1, 8)) * 0.1, np.ones((1, 3)) * 0.1]
        new_w, new_b = update_weights(weights, biases, grads_w, grads_b, lr=0.1)
        self.assertFalse(np.allclose(new_w[0], weights[0]))

    def test_loss_is_float(self):
        y_pred = np.array([[0.5, 0.5]])
        y_true = np.array([[1, 0]])
        loss = compute_loss(y_pred, y_true)
        self.assertIsInstance(loss, float)

    def test_loss_positive(self):
        y_pred = np.array([[0.5, 0.5]])
        y_true = np.array([[1, 0]])
        loss = compute_loss(y_pred, y_true)
        self.assertGreater(loss, 0)

    def test_backward_shapes(self):
        x = np.random.randn(5, 4)
        y = np.eye(3)[np.array([0, 1, 2, 0, 1])]
        weights = [np.random.randn(4, 8), np.random.randn(8, 3)]
        biases = [np.zeros((1, 8)), np.zeros((1, 3))]
        a1 = np.maximum(0, np.dot(x, weights[0]) + biases[0])
        a2 = np.exp(np.dot(a1, weights[1]) + biases[1])
        a2 = a2 / a2.sum(axis=1, keepdims=True)
        grads_w, grads_b = backward_pass(x, y, weights, biases, [a1, a2])
        self.assertEqual(grads_w[0].shape, (4, 8))
        self.assertEqual(grads_w[1].shape, (8, 3))

    def test_update_returns_tuple(self):
        weights = [np.ones((4, 8))]
        biases = [np.zeros((1, 8))]
        grads_w = [np.ones((4, 8)) * 0.1]
        grads_b = [np.ones((1, 8)) * 0.1]
        result = update_weights(weights, biases, grads_w, grads_b, lr=0.1)
        self.assertEqual(len(result), 2)

    def test_update_changes_biases(self):
        weights = [np.ones((4, 8))]
        biases = [np.zeros((1, 8))]
        grads_w = [np.ones((4, 8)) * 0.1]
        grads_b = [np.ones((1, 8)) * 0.1]
        new_w, new_b = update_weights(weights, biases, grads_w, grads_b, lr=0.1)
        self.assertFalse(np.allclose(new_b[0], biases[0]))
`,

	hint1: 'Cross-entropy: -mean(sum(y_true * log(y_pred)))',
	hint2: 'For softmax+cross-entropy, gradient simplifies to: dz = y_pred - y_true',

	whyItMatters: `Backpropagation is the heart of deep learning:

- **Training algorithm**: How neural networks learn
- **Chain rule**: Efficiently compute all gradients
- **Gradient flow**: Understanding vanishing/exploding gradients
- **Debugging**: Gradient checking for correctness

Every modern deep learning framework implements this.`,

	translations: {
		ru: {
			title: 'Обратное распространение',
			description: `# Обратное распространение

Реализуйте обратное распространение для вычисления градиентов 2-слойной сети.

## Задача

Реализуйте три функции:
1. \`compute_loss(y_pred, y_true)\` - Кросс-энтропийная функция потерь
2. \`backward_pass(x, y, weights, biases, activations)\` - Вычислить градиенты
3. \`update_weights(weights, biases, grads_w, grads_b, lr)\` - Применить градиентный спуск

## Пример

\`\`\`python
# Forward pass (computed elsewhere)
activations = [a1, a2]  # Hidden and output activations

# Compute gradients
grads_w, grads_b = backward_pass(x, y, weights, biases, activations)

# Update weights
weights, biases = update_weights(weights, biases, grads_w, grads_b, lr=0.01)
\`\`\``,
			hint1: 'Кросс-энтропия: -mean(sum(y_true * log(y_pred)))',
			hint2: 'Для softmax+cross-entropy градиент упрощается до: dz = y_pred - y_true',
			whyItMatters: `Обратное распространение - сердце глубокого обучения:

- **Алгоритм обучения**: Как нейросети обучаются
- **Правило цепочки**: Эффективное вычисление всех градиентов
- **Поток градиентов**: Понимание затухающих/взрывающихся градиентов`,
		},
		uz: {
			title: 'Orqaga tarqalish',
			description: `# Orqaga tarqalish

2 qatlamli tarmoq uchun gradientlarni hisoblash uchun orqaga tarqalishni amalga oshiring.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`compute_loss(y_pred, y_true)\` - Kross-entropiya yo'qotish
2. \`backward_pass(x, y, weights, biases, activations)\` - Gradientlarni hisoblash
3. \`update_weights(weights, biases, grads_w, grads_b, lr)\` - Gradient tushishni qo'llash

## Misol

\`\`\`python
# Forward pass (computed elsewhere)
activations = [a1, a2]  # Hidden and output activations

# Compute gradients
grads_w, grads_b = backward_pass(x, y, weights, biases, activations)

# Update weights
weights, biases = update_weights(weights, biases, grads_w, grads_b, lr=0.01)
\`\`\``,
			hint1: "Kross-entropiya: -mean(sum(y_true * log(y_pred)))",
			hint2: "softmax+cross-entropy uchun gradient soddalashadi: dz = y_pred - y_true",
			whyItMatters: `Orqaga tarqalish chuqur o'rganishning yuragi:

- **O'rganish algoritmi**: Neyrosetka qanday o'rganadi
- **Zanjir qoidasi**: Barcha gradientlarni samarali hisoblash
- **Gradient oqimi**: So'nuvchi/portlovchi gradientlarni tushunish`,
		},
	},
};

export default task;
