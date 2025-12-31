import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'pydl-nn-loss-functions',
	title: 'Loss Functions',
	difficulty: 'medium',
	tags: ['numpy', 'loss', 'neural-networks'],
	estimatedTime: '12m',
	isPremium: false,
	order: 5,
	description: `# Loss Functions

Implement common loss functions for neural network training.

## Task

Implement four functions:
1. \`mse_loss(y_pred, y_true)\` - Mean Squared Error for regression
2. \`binary_cross_entropy(y_pred, y_true)\` - BCE for binary classification
3. \`categorical_cross_entropy(y_pred, y_true)\` - CCE for multi-class
4. \`huber_loss(y_pred, y_true, delta)\` - Robust loss for regression

## Example

\`\`\`python
y_pred = np.array([0.9, 0.1, 0.8])
y_true = np.array([1.0, 0.0, 1.0])

mse = mse_loss(y_pred, y_true)  # 0.0067
bce = binary_cross_entropy(y_pred, y_true)  # 0.164
\`\`\``,

	initialCode: `import numpy as np

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error loss."""
    # Your code here
    pass

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary Cross-Entropy loss. Clip predictions for numerical stability."""
    # Your code here
    pass

def categorical_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Categorical Cross-Entropy loss. y_pred is softmax output."""
    # Your code here
    pass

def huber_loss(y_pred: np.ndarray, y_true: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss - combines MSE and MAE. Robust to outliers."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error loss."""
    return np.mean((y_pred - y_true) ** 2)

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary Cross-Entropy loss. Clip predictions for numerical stability."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Categorical Cross-Entropy loss. y_pred is softmax output."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def huber_loss(y_pred: np.ndarray, y_true: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss - combines MSE and MAE. Robust to outliers."""
    error = y_pred - y_true
    is_small = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
    return np.mean(np.where(is_small, squared_loss, linear_loss))
`,

	testCode: `import numpy as np
import unittest

class TestLossFunctions(unittest.TestCase):
    def test_mse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        loss = mse_loss(y, y)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_mse_positive(self):
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.5, 2.5, 3.5])
        loss = mse_loss(y_pred, y_true)
        self.assertGreater(loss, 0)

    def test_bce_range(self):
        y_pred = np.array([0.9, 0.1])
        y_true = np.array([1.0, 0.0])
        loss = binary_cross_entropy(y_pred, y_true)
        self.assertGreater(loss, 0)
        self.assertLess(loss, 1)

    def test_cce_one_hot(self):
        y_pred = np.array([[0.9, 0.05, 0.05]])
        y_true = np.array([[1, 0, 0]])
        loss = categorical_cross_entropy(y_pred, y_true)
        self.assertLess(loss, 0.2)

    def test_huber_small_error(self):
        y_pred = np.array([1.0])
        y_true = np.array([1.1])
        loss = huber_loss(y_pred, y_true, delta=1.0)
        self.assertAlmostEqual(loss, 0.5 * 0.1 ** 2, places=5)

    def test_mse_returns_float(self):
        y_pred = np.array([1.0, 2.0])
        y_true = np.array([1.5, 2.5])
        loss = mse_loss(y_pred, y_true)
        self.assertIsInstance(loss, (float, np.floating))

    def test_bce_perfect_prediction(self):
        y_pred = np.array([0.99, 0.01])
        y_true = np.array([1.0, 0.0])
        loss = binary_cross_entropy(y_pred, y_true)
        self.assertLess(loss, 0.1)

    def test_cce_bad_prediction(self):
        y_pred = np.array([[0.1, 0.8, 0.1]])
        y_true = np.array([[1, 0, 0]])
        loss = categorical_cross_entropy(y_pred, y_true)
        self.assertGreater(loss, 1.0)

    def test_huber_large_error(self):
        y_pred = np.array([0.0])
        y_true = np.array([10.0])
        loss = huber_loss(y_pred, y_true, delta=1.0)
        expected = 1.0 * 10.0 - 0.5 * 1.0 ** 2
        self.assertAlmostEqual(loss, expected, places=3)

    def test_mse_multiple_samples(self):
        y_pred = np.random.randn(100)
        y_true = np.random.randn(100)
        loss = mse_loss(y_pred, y_true)
        self.assertGreater(loss, 0)
`,

	hint1: 'MSE = mean((y_pred - y_true)^2), BCE needs log clipping',
	hint2: 'Huber: use MSE when |error| <= delta, else use linear loss',

	whyItMatters: `Loss functions guide neural network learning:

- **Optimization target**: What the network minimizes
- **Task-specific**: Different losses for regression vs classification
- **Gradient properties**: Affects training stability
- **Robustness**: Huber loss handles outliers better

Choosing the right loss is crucial for model performance.`,

	translations: {
		ru: {
			title: 'Функции потерь',
			description: `# Функции потерь

Реализуйте популярные функции потерь для обучения нейросетей.

## Задача

Реализуйте четыре функции:
1. \`mse_loss(y_pred, y_true)\` - MSE для регрессии
2. \`binary_cross_entropy(y_pred, y_true)\` - BCE для бинарной классификации
3. \`categorical_cross_entropy(y_pred, y_true)\` - CCE для многоклассовой
4. \`huber_loss(y_pred, y_true, delta)\` - Робастная функция для регрессии

## Пример

\`\`\`python
y_pred = np.array([0.9, 0.1, 0.8])
y_true = np.array([1.0, 0.0, 1.0])

mse = mse_loss(y_pred, y_true)  # 0.0067
bce = binary_cross_entropy(y_pred, y_true)  # 0.164
\`\`\``,
			hint1: 'MSE = mean((y_pred - y_true)^2), BCE требует обрезки log',
			hint2: 'Huber: используйте MSE когда |error| <= delta, иначе линейную потерю',
			whyItMatters: `Функции потерь направляют обучение нейросети:

- **Цель оптимизации**: Что сеть минимизирует
- **Специфика задачи**: Разные потери для регрессии и классификации
- **Свойства градиента**: Влияет на стабильность обучения`,
		},
		uz: {
			title: "Yo'qotish funksiyalari",
			description: `# Yo'qotish funksiyalari

Neyrosetka o'qitish uchun ommabop yo'qotish funksiyalarini amalga oshiring.

## Topshiriq

To'rtta funksiyani amalga oshiring:
1. \`mse_loss(y_pred, y_true)\` - Regressiya uchun MSE
2. \`binary_cross_entropy(y_pred, y_true)\` - Ikkilik klassifikatsiya uchun BCE
3. \`categorical_cross_entropy(y_pred, y_true)\` - Ko'p sinfli CCE
4. \`huber_loss(y_pred, y_true, delta)\` - Regressiya uchun robust yo'qotish

## Misol

\`\`\`python
y_pred = np.array([0.9, 0.1, 0.8])
y_true = np.array([1.0, 0.0, 1.0])

mse = mse_loss(y_pred, y_true)  # 0.0067
bce = binary_cross_entropy(y_pred, y_true)  # 0.164
\`\`\``,
			hint1: "MSE = mean((y_pred - y_true)^2), BCE log kesish talab qiladi",
			hint2: "Huber: |error| <= delta bo'lganda MSE ishlating, aks holda chiziqli yo'qotish",
			whyItMatters: `Yo'qotish funksiyalari neyrosetka o'rganishini yo'naltiradi:

- **Optimallashtirish maqsadi**: Tarmoq nimani minimallashtiradi
- **Topshiriqqa xos**: Regressiya va klassifikatsiya uchun turli yo'qotishlar
- **Gradient xususiyatlari**: O'qitish barqarorligiga ta'sir qiladi`,
		},
	},
};

export default task;
