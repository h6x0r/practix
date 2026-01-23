import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-cross-entropy',
	title: 'Cross-Entropy Loss',
	difficulty: 'hard',
	tags: ['python', 'math', 'statistics', 'cross-entropy', 'loss', 'numpy'],
	estimatedTime: '20m',
	isPremium: true,
	order: 5,
	description: `# Cross-Entropy Loss

Implement binary and categorical cross-entropy loss functions.

## Binary Cross-Entropy

L = -[y×log(p) + (1-y)×log(1-p)]

## Categorical Cross-Entropy

L = -Σᵢ yᵢ×log(pᵢ)

THE loss function for classification!
`,
	initialCode: `import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy loss."""
    # Your code here
    pass

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Categorical cross-entropy loss (y_true is one-hot)."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy loss."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Categorical cross-entropy loss (y_true is one-hot)."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
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

class TestCrossEntropy(unittest.TestCase):
    def test_binary_perfect_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.99, 0.01, 0.99, 0.01])
        result = binary_cross_entropy(y_true, y_pred)
        assert result < 0.05

    def test_binary_wrong_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.01, 0.99, 0.01, 0.99])
        result = binary_cross_entropy(y_true, y_pred)
        assert result > 2.0

    def test_binary_uncertain(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0.5, 0.5])
        result = binary_cross_entropy(y_true, y_pred)
        assert_close(result, np.log(2), places=4)

    def test_categorical_perfect(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.99, 0.005, 0.005], [0.01, 0.98, 0.01]])
        result = categorical_cross_entropy(y_true, y_pred)
        assert result < 0.05

    def test_categorical_wrong(self):
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.01, 0.01, 0.98]])
        result = categorical_cross_entropy(y_true, y_pred)
        assert result > 2.0

    def test_binary_returns_float(self):
        result = binary_cross_entropy(np.array([1]), np.array([0.5]))
        self.assertIsInstance(float(result), float)

    def test_categorical_returns_float(self):
        result = categorical_cross_entropy(np.array([[1, 0]]), np.array([[0.5, 0.5]]))
        self.assertIsInstance(float(result), float)

    def test_binary_non_negative(self):
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0.1, 0.9, 100)
        result = binary_cross_entropy(y_true, y_pred)
        assert result >= 0

    def test_categorical_non_negative(self):
        y_true = np.eye(3)
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        result = categorical_cross_entropy(y_true, y_pred)
        assert result >= 0

    def test_handles_edge_values(self):
        y_true = np.array([1])
        y_pred = np.array([0.999999])
        result = binary_cross_entropy(y_true, y_pred)
        self.assertFalse(np.isnan(result))

`,
	hint1: 'Clip predictions to avoid log(0): np.clip(y_pred, eps, 1-eps)',
	hint2: 'Binary: -mean(y*log(p) + (1-y)*log(1-p)). Categorical: -mean(sum(y*log(p), axis=-1))',
	whyItMatters: `Cross-entropy is THE loss function for classification. Neural network classifiers minimize cross-entropy. It measures how well predicted probabilities match true labels. **Production Pattern:** Always use cross-entropy with softmax output for multi-class classification.`,
	translations: {
		ru: {
			title: 'Кросс-энтропия',
			description: `# Кросс-энтропийная функция потерь

Реализуйте бинарную и категориальную кросс-энтропию.

## Бинарная кросс-энтропия

L = -[y×log(p) + (1-y)×log(1-p)]

## Категориальная кросс-энтропия

L = -Σᵢ yᵢ×log(pᵢ)

ГЛАВНАЯ функция потерь для классификации!
`,
			hint1: 'Обрежьте предсказания: np.clip(y_pred, eps, 1-eps)',
			hint2: 'Бинарная: -mean(y*log(p) + (1-y)*log(1-p))',
			whyItMatters: `Кросс-энтропия - ГЛАВНАЯ функция потерь для классификации. **Production Pattern:** Всегда используйте кросс-энтропию с softmax для многоклассовой классификации.`,
		},
		uz: {
			title: 'Kross-entropiya',
			description: `# Kross-entropiya yo'qotish funksiyasi

Ikkilik va kategorik kross-entropiyani amalga oshiring.

Klassifikatsiya uchun ASOSIY yo'qotish funksiyasi!
`,
			hint1: 'Bashoratlarni kesib tashlang: np.clip(y_pred, eps, 1-eps)',
			hint2: 'Ikkilik: -mean(y*log(p) + (1-y)*log(1-p))',
			whyItMatters: `Kross-entropiya klassifikatsiya uchun ASOSIY yo'qotish funksiyasi. Ko'p sinfli klassifikatsiya uchun har doim softmax bilan kross-entropiyadan foydalaning.`,
		},
	},
};

export default task;
