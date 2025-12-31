import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-perceptron',
	title: 'Perceptron from Scratch',
	difficulty: 'easy',
	tags: ['numpy', 'perceptron', 'neural-networks'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Perceptron from Scratch

Implement a single perceptron - the building block of neural networks.

## Task

Implement three functions:
1. \`perceptron_forward(x, weights, bias)\` - Compute weighted sum and apply step function
2. \`train_perceptron(X, y, lr, epochs)\` - Train perceptron using perceptron learning rule
3. \`predict(X, weights, bias)\` - Make predictions on new data

## Example

\`\`\`python
import numpy as np

# AND gate
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

weights, bias = train_perceptron(X, y, lr=0.1, epochs=100)
predictions = predict(X, weights, bias)  # [0, 0, 0, 1]
\`\`\``,

	initialCode: `import numpy as np

def perceptron_forward(x: np.ndarray, weights: np.ndarray, bias: float) -> int:
    """Compute weighted sum and apply step function. Return 0 or 1."""
    # Your code here
    pass

def train_perceptron(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 100):
    """Train perceptron. Return (weights, bias) tuple."""
    # Your code here
    pass

def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Predict for all samples. Return array of 0s and 1s."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def perceptron_forward(x: np.ndarray, weights: np.ndarray, bias: float) -> int:
    """Compute weighted sum and apply step function. Return 0 or 1."""
    z = np.dot(x, weights) + bias
    return 1 if z >= 0 else 0

def train_perceptron(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 100):
    """Train perceptron. Return (weights, bias) tuple."""
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        for xi, yi in zip(X, y):
            pred = perceptron_forward(xi, weights, bias)
            error = yi - pred
            weights += lr * error * xi
            bias += lr * error

    return weights, bias

def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Predict for all samples. Return array of 0s and 1s."""
    return np.array([perceptron_forward(x, weights, bias) for x in X])
`,

	testCode: `import numpy as np
import unittest

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        self.X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
        self.y_and = np.array([0, 0, 0, 1])

    def test_forward_returns_0_or_1(self):
        weights = np.array([0.5, 0.5])
        result = perceptron_forward(np.array([1, 1]), weights, -0.7)
        self.assertIn(result, [0, 1])

    def test_train_returns_tuple(self):
        result = train_perceptron(self.X_and, self.y_and, 0.1, 10)
        self.assertEqual(len(result), 2)

    def test_and_gate(self):
        weights, bias = train_perceptron(self.X_and, self.y_and, 0.1, 100)
        preds = predict(self.X_and, weights, bias)
        np.testing.assert_array_equal(preds, self.y_and)

    def test_predict_shape(self):
        weights, bias = train_perceptron(self.X_and, self.y_and)
        preds = predict(self.X_and, weights, bias)
        self.assertEqual(len(preds), 4)

    def test_weights_shape(self):
        weights, bias = train_perceptron(self.X_and, self.y_and)
        self.assertEqual(len(weights), 2)

    def test_bias_is_float(self):
        weights, bias = train_perceptron(self.X_and, self.y_and)
        self.assertIsInstance(bias, (int, float))

    def test_predict_returns_numpy(self):
        weights, bias = train_perceptron(self.X_and, self.y_and)
        preds = predict(self.X_and, weights, bias)
        self.assertIsInstance(preds, np.ndarray)

    def test_forward_with_zero_weights(self):
        weights = np.array([0.0, 0.0])
        result = perceptron_forward(np.array([1, 1]), weights, 0)
        self.assertEqual(result, 1)

    def test_or_gate(self):
        X_or = np.array([[0,0], [0,1], [1,0], [1,1]])
        y_or = np.array([0, 1, 1, 1])
        weights, bias = train_perceptron(X_or, y_or, 0.1, 100)
        preds = predict(X_or, weights, bias)
        np.testing.assert_array_equal(preds, y_or)
`,

	hint1: 'Weighted sum: z = np.dot(x, weights) + bias, then apply step: 1 if z >= 0 else 0',
	hint2: 'Update rule: weights += lr * (y - pred) * x, bias += lr * (y - pred)',

	whyItMatters: `The perceptron is foundational because:

- **Historical importance**: First neural network model (1958)
- **Building block**: Modern networks are stacked perceptrons
- **Linear classifier**: Learns linear decision boundaries
- **Gradient concepts**: Introduces weight updates and learning

Understanding perceptrons unlocks deep learning intuition.`,

	translations: {
		ru: {
			title: 'Перцептрон с нуля',
			description: `# Перцептрон с нуля

Реализуйте одиночный перцептрон - строительный блок нейросетей.

## Задача

Реализуйте три функции:
1. \`perceptron_forward(x, weights, bias)\` - Вычислить взвешенную сумму и применить ступенчатую функцию
2. \`train_perceptron(X, y, lr, epochs)\` - Обучить перцептрон
3. \`predict(X, weights, bias)\` - Сделать предсказания

## Пример

\`\`\`python
import numpy as np

# AND gate
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

weights, bias = train_perceptron(X, y, lr=0.1, epochs=100)
predictions = predict(X, weights, bias)  # [0, 0, 0, 1]
\`\`\``,
			hint1: 'Взвешенная сумма: z = np.dot(x, weights) + bias, затем step: 1 if z >= 0 else 0',
			hint2: 'Правило обновления: weights += lr * (y - pred) * x',
			whyItMatters: `Перцептрон - это основа потому что:

- **Историческое значение**: Первая модель нейросети (1958)
- **Строительный блок**: Современные сети - это стеки перцептронов
- **Линейный классификатор**: Обучает линейные границы решений`,
		},
		uz: {
			title: 'Perceptron noldan',
			description: `# Perceptron noldan

Yagona perceptronni amalga oshiring - neyrosetka qurilish bloki.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`perceptron_forward(x, weights, bias)\` - Og'irlikli yig'indini hisoblash va pog'ona funksiyasini qo'llash
2. \`train_perceptron(X, y, lr, epochs)\` - Perceptronni o'rgatish
3. \`predict(X, weights, bias)\` - Bashoratlar qilish

## Misol

\`\`\`python
import numpy as np

# AND gate
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

weights, bias = train_perceptron(X, y, lr=0.1, epochs=100)
predictions = predict(X, weights, bias)  # [0, 0, 0, 1]
\`\`\``,
			hint1: "Og'irlikli yig'indi: z = np.dot(x, weights) + bias, keyin step: 1 if z >= 0 else 0",
			hint2: "Yangilash qoidasi: weights += lr * (y - pred) * x",
			whyItMatters: `Perceptron asosiy chunki:

- **Tarixiy ahamiyat**: Birinchi neyrosetka modeli (1958)
- **Qurilish bloki**: Zamonaviy tarmoqlar - bu perceptron steklari
- **Chiziqli klassifikator**: Chiziqli qaror chegaralarini o'rganadi`,
		},
	},
};

export default task;
