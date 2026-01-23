import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-standardization',
	title: 'Data Standardization',
	difficulty: 'easy',
	tags: ['python', 'math', 'statistics', 'standardization', 'preprocessing', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 4,
	description: `# Data Standardization

Implement z-score standardization: transform data to have mean=0 and std=1.

## Formula

z = (x - μ) / σ

This is essential preprocessing for many ML algorithms!
`,
	initialCode: `import numpy as np

def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize data to zero mean and unit variance."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize data to zero mean and unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data - mean
    return (data - mean) / std
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

class TestStandardize(unittest.TestCase):
    def test_mean_zero(self):
        data = np.array([1, 2, 3, 4, 5])
        result = standardize(data)
        self.assertAlmostEqual(np.mean(result), 0.0, places=10)

    def test_std_one(self):
        data = np.array([1, 2, 3, 4, 5])
        result = standardize(data)
        self.assertAlmostEqual(np.std(result), 1.0, places=10)

    def test_preserves_shape(self):
        data = np.array([1, 2, 3, 4, 5])
        result = standardize(data)
        assert result.shape == data.shape, f"Expected data.shape, got {result.shape}"

    def test_negative_values(self):
        data = np.array([-5, -3, -1, 1, 3, 5])
        result = standardize(data)
        self.assertAlmostEqual(np.mean(result), 0.0, places=10)

    def test_already_standardized(self):
        data = np.array([-1.5, -0.5, 0.5, 1.5])
        result = standardize(data)
        self.assertAlmostEqual(np.mean(result), 0.0, places=10)

    def test_large_values(self):
        data = np.array([1000, 2000, 3000])
        result = standardize(data)
        self.assertAlmostEqual(np.mean(result), 0.0, places=10)

    def test_returns_array(self):
        result = standardize(np.array([1, 2, 3]))
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

    def test_ordering_preserved(self):
        data = np.array([1, 2, 3, 4, 5])
        result = standardize(data)
        for i in range(len(result) - 1):
            assert result[i] < result[i+1]

    def test_floats(self):
        data = np.array([1.5, 2.5, 3.5, 4.5])
        result = standardize(data)
        self.assertAlmostEqual(np.mean(result), 0.0, places=10)

    def test_symmetric_data(self):
        data = np.array([-2, -1, 0, 1, 2])
        result = standardize(data)
        assert_close(result[2], 0.0)

`,
	hint1: 'z = (x - mean) / std',
	hint2: 'Use np.mean() and np.std() to compute statistics, then apply the formula.',
	whyItMatters: `Standardization is essential preprocessing. Gradient descent converges faster with standardized features. SVM and neural networks require it. **Production Pattern:** sklearn.preprocessing.StandardScaler fits on training data and transforms both train and test.`,
	translations: {
		ru: {
			title: 'Стандартизация данных',
			description: `# Стандартизация данных

Реализуйте z-score стандартизацию: преобразование данных к среднему=0 и std=1.

## Формула

z = (x - μ) / σ

Это важнейшая предобработка для многих алгоритмов ML!
`,
			hint1: 'z = (x - mean) / std',
			hint2: 'Используйте np.mean() и np.std(), затем примените формулу.',
			whyItMatters: `Стандартизация - важнейшая предобработка. Градиентный спуск сходится быстрее на стандартизованных данных. **Production Pattern:** StandardScaler обучается на тренировочных данных и применяется к обоим.`,
		},
		uz: {
			title: 'Ma\'lumotlarni standartlashtirish',
			description: `# Ma'lumotlarni standartlashtirish

z-score standartlashtirishni amalga oshiring: ma'lumotlarni o'rtacha=0 va std=1 ga aylantiring.
`,
			hint1: 'z = (x - mean) / std',
			hint2: 'np.mean() va np.std() dan foydalaning, keyin formulani qo\'llang.',
			whyItMatters: `Standartlashtirish muhim oldindan ishlov berish. Gradient tushishi standartlashtirilgan xususiyatlar bilan tezroq yaqinlashadi.`,
		},
	},
};

export default task;
