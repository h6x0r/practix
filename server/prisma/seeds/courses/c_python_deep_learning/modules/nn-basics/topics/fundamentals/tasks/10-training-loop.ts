import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-training-loop',
	title: 'Complete Training Loop',
	difficulty: 'hard',
	tags: ['numpy', 'training', 'neural-networks'],
	estimatedTime: '20m',
	isPremium: true,
	order: 10,
	description: `# Complete Training Loop

Put it all together: implement a complete training loop for an MLP.

## Task

Implement two functions:
1. \`train_epoch(model, X, y, lr, batch_size)\` - Train for one epoch
2. \`train_network(X_train, y_train, X_val, y_val, hidden_sizes, epochs, lr)\` - Full training

## Example

\`\`\`python
# Train a 2-layer MLP
history = train_network(
    X_train, y_train, X_val, y_val,
    hidden_sizes=[128, 64],
    epochs=100,
    lr=0.001
)

# history contains train_loss, val_loss, val_accuracy per epoch
\`\`\``,

	initialCode: `import numpy as np

def train_epoch(X: np.ndarray, y: np.ndarray, weights: list, biases: list,
                lr: float, batch_size: int) -> tuple:
    """Train for one epoch. Return (updated_weights, updated_biases, epoch_loss)."""
    # Your code here
    pass

def train_network(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  hidden_sizes: list, epochs: int, lr: float) -> dict:
    """Full training loop. Return history dict with losses and accuracy."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def train_epoch(X: np.ndarray, y: np.ndarray, weights: list, biases: list,
                lr: float, batch_size: int) -> tuple:
    """Train for one epoch. Return (updated_weights, updated_biases, epoch_loss)."""
    n = len(X)
    indices = np.random.permutation(n)
    total_loss = 0
    n_batches = 0

    for i in range(0, n, batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        m = len(X_batch)

        # Forward pass
        a1 = relu(np.dot(X_batch, weights[0]) + biases[0])
        a2 = softmax(np.dot(a1, weights[1]) + biases[1])

        total_loss += cross_entropy(a2, y_batch)
        n_batches += 1

        # Backward pass
        dz2 = a2 - y_batch
        dw2 = np.dot(a1.T, dz2) / m
        db2 = np.mean(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, weights[1].T)
        dz1 = da1 * (a1 > 0)
        dw1 = np.dot(X_batch.T, dz1) / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # Update weights
        weights[0] -= lr * dw1
        weights[1] -= lr * dw2
        biases[0] -= lr * db1
        biases[1] -= lr * db2

    return weights, biases, total_loss / n_batches

def train_network(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  hidden_sizes: list, epochs: int, lr: float) -> dict:
    """Full training loop. Return history dict with losses and accuracy."""
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]

    # Initialize weights
    layer_sizes = [n_features] + hidden_sizes + [n_classes]
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i+1]))
        weights.append(W)
        biases.append(b)

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        weights, biases, train_loss = train_epoch(
            X_train, y_train, weights, biases, lr, batch_size=32
        )

        # Validation
        a1 = relu(np.dot(X_val, weights[0]) + biases[0])
        a2 = softmax(np.dot(a1, weights[1]) + biases[1])
        val_loss = cross_entropy(a2, y_val)
        val_acc = np.mean(np.argmax(a2, axis=1) == np.argmax(y_val, axis=1))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

    return history
`,

	testCode: `import numpy as np
import unittest

class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = np.eye(3)[np.random.randint(0, 3, 100)]

        # Initialize simple network
        self.weights = [
            np.random.randn(10, 8) * 0.1,
            np.random.randn(8, 3) * 0.1
        ]
        self.biases = [np.zeros((1, 8)), np.zeros((1, 3))]

    def test_train_epoch_returns_tuple(self):
        result = train_epoch(self.X, self.y, self.weights, self.biases, 0.01, 32)
        self.assertEqual(len(result), 3)

    def test_train_epoch_loss_positive(self):
        w, b, loss = train_epoch(self.X, self.y, self.weights, self.biases, 0.01, 32)
        self.assertGreater(loss, 0)

    def test_train_network_returns_history(self):
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y[:80], self.y[80:]
        history = train_network(X_train, y_train, X_val, y_val, [8], 5, 0.01)
        self.assertIn('train_loss', history)
        self.assertIn('val_accuracy', history)

    def test_training_reduces_loss(self):
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y[:80], self.y[80:]
        history = train_network(X_train, y_train, X_val, y_val, [16], 20, 0.1)
        self.assertLess(history['train_loss'][-1], history['train_loss'][0])

    def test_train_epoch_weights_updated(self):
        w_before = [w.copy() for w in self.weights]
        w, b, loss = train_epoch(self.X, self.y, self.weights, self.biases, 0.01, 32)
        self.assertFalse(np.allclose(w[0], w_before[0]))

    def test_history_length_matches_epochs(self):
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y[:80], self.y[80:]
        history = train_network(X_train, y_train, X_val, y_val, [8], 10, 0.01)
        self.assertEqual(len(history['train_loss']), 10)

    def test_val_accuracy_in_range(self):
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y[:80], self.y[80:]
        history = train_network(X_train, y_train, X_val, y_val, [8], 5, 0.01)
        for acc in history['val_accuracy']:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_val_loss_tracked(self):
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y[:80], self.y[80:]
        history = train_network(X_train, y_train, X_val, y_val, [8], 5, 0.01)
        self.assertEqual(len(history['val_loss']), 5)

    def test_train_epoch_batch_sizes(self):
        w, b, loss = train_epoch(self.X, self.y, self.weights, self.biases, 0.01, 16)
        self.assertGreater(loss, 0)
`,

	hint1: 'Combine forward pass, backward pass, and weight updates in train_epoch',
	hint2: 'Track metrics per epoch: train_loss, val_loss, val_accuracy',

	whyItMatters: `A complete training loop ties everything together:

- **End-to-end understanding**: See how all components work together
- **Debugging skills**: Identify issues in the training pipeline
- **Foundation for PyTorch**: Same concepts, cleaner API
- **Experimentation**: Modify any part of the loop

Essential foundation before using deep learning frameworks.`,

	translations: {
		ru: {
			title: 'Полный цикл обучения',
			description: `# Полный цикл обучения

Соберите всё вместе: реализуйте полный цикл обучения для MLP.

## Задача

Реализуйте две функции:
1. \`train_epoch(model, X, y, lr, batch_size)\` - Обучить за одну эпоху
2. \`train_network(X_train, y_train, X_val, y_val, hidden_sizes, epochs, lr)\` - Полное обучение

## Пример

\`\`\`python
# Train a 2-layer MLP
history = train_network(
    X_train, y_train, X_val, y_val,
    hidden_sizes=[128, 64],
    epochs=100,
    lr=0.001
)

# history contains train_loss, val_loss, val_accuracy per epoch
\`\`\``,
			hint1: 'Объедините прямой проход, обратный проход и обновление весов в train_epoch',
			hint2: 'Отслеживайте метрики по эпохам: train_loss, val_loss, val_accuracy',
			whyItMatters: `Полный цикл обучения объединяет всё:

- **Полное понимание**: Видеть как все компоненты работают вместе
- **Навыки отладки**: Находить проблемы в пайплайне обучения
- **Фундамент для PyTorch**: Те же концепции, чище API`,
		},
		uz: {
			title: "To'liq o'qitish sikli",
			description: `# To'liq o'qitish sikli

Hammasini birlashtiring: MLP uchun to'liq o'qitish siklini amalga oshiring.

## Topshiriq

Ikkita funksiyani amalga oshiring:
1. \`train_epoch(model, X, y, lr, batch_size)\` - Bitta davrda o'qitish
2. \`train_network(X_train, y_train, X_val, y_val, hidden_sizes, epochs, lr)\` - To'liq o'qitish

## Misol

\`\`\`python
# Train a 2-layer MLP
history = train_network(
    X_train, y_train, X_val, y_val,
    hidden_sizes=[128, 64],
    epochs=100,
    lr=0.001
)

# history contains train_loss, val_loss, val_accuracy per epoch
\`\`\``,
			hint1: "Oldinga o'tish, orqaga o'tish va og'irliklarni yangilashni train_epoch da birlashtiring",
			hint2: "Davrlar bo'yicha ko'rsatkichlarni kuzating: train_loss, val_loss, val_accuracy",
			whyItMatters: `To'liq o'qitish sikli hammasini birlashtiradi:

- **Oxirigacha tushunish**: Barcha komponentlar qanday birga ishlashini ko'ring
- **Nosozliklarni tuzatish ko'nikmalari**: O'qitish pipelineidagi muammolarni aniqlash
- **PyTorch uchun asos**: Xuddi shu tushunchalar, tozaroq API`,
		},
	},
};

export default task;
