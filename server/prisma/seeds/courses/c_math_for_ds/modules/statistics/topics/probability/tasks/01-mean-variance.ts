import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-mean-variance',
	title: 'Mean & Variance',
	difficulty: 'easy',
	tags: ['python', 'math', 'statistics', 'mean', 'variance', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Mean & Variance

Implement functions to compute mean and variance of a dataset.

## Formulas

Mean: μ = (1/n) Σxᵢ
Variance: σ² = (1/n) Σ(xᵢ - μ)²

## Example

\`\`\`python
data = [2, 4, 4, 4, 5, 5, 7, 9]
mean(data)      # 5.0
variance(data)  # 4.0
\`\`\`
`,
	initialCode: `import numpy as np

def compute_mean(data: np.ndarray) -> float:
    """Compute arithmetic mean."""
    # Your code here
    pass

def compute_variance(data: np.ndarray) -> float:
    """Compute population variance."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def compute_mean(data: np.ndarray) -> float:
    """Compute arithmetic mean."""
    return np.mean(data)

def compute_variance(data: np.ndarray) -> float:
    """Compute population variance."""
    return np.var(data)
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

class TestMeanVariance(unittest.TestCase):
    def test_mean_simple(self):
        data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
        result = compute_mean(data)
        assert_close(result, 5.0)

    def test_mean_single(self):
        result = compute_mean(np.array([5.0]))
        assert_close(result, 5.0)

    def test_variance_simple(self):
        data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
        result = compute_variance(data)
        assert_close(result, 4.0)

    def test_variance_zero(self):
        data = np.array([5, 5, 5, 5])
        result = compute_variance(data)
        assert_close(result, 0.0)

    def test_mean_negative(self):
        data = np.array([-2, -1, 0, 1, 2])
        result = compute_mean(data)
        assert_close(result, 0.0)

    def test_variance_symmetric(self):
        data = np.array([-2, -1, 0, 1, 2])
        result = compute_variance(data)
        assert_close(result, 2.0)

    def test_mean_floats(self):
        data = np.array([1.5, 2.5, 3.5])
        result = compute_mean(data)
        assert_close(result, 2.5)

    def test_variance_floats(self):
        data = np.array([1.0, 2.0, 3.0])
        result = compute_variance(data)
        assert_close(result, 2/3)

    def test_mean_returns_float(self):
        result = compute_mean(np.array([1, 2, 3]))
        self.assertIsInstance(float(result), float)

    def test_variance_returns_float(self):
        result = compute_variance(np.array([1, 2, 3]))
        self.assertIsInstance(float(result), float)

`,
	hint1: 'Use np.mean() and np.var() for efficient computation.',
	hint2: 'Variance is the mean of squared deviations from the mean.',
	whyItMatters: `Mean and variance are fundamental to understanding data. Batch normalization uses running mean/variance. Feature scaling (standardization) requires these statistics. **Production Pattern:** Always compute statistics on training data only, then apply to test data.`,
	translations: {
		ru: {
			title: 'Среднее и дисперсия',
			description: `# Среднее и дисперсия

Реализуйте функции вычисления среднего и дисперсии набора данных.
`,
			hint1: 'Используйте np.mean() и np.var() для эффективного вычисления.',
			hint2: 'Дисперсия - среднее квадратов отклонений от среднего.',
			whyItMatters: `Среднее и дисперсия фундаментальны для понимания данных. Batch normalization использует скользящее среднее/дисперсию.`,
		},
		uz: {
			title: 'O\'rtacha va dispersiya',
			description: `# O'rtacha va dispersiya

Ma'lumotlar to'plamining o'rtacha va dispersiyasini hisoblaydigan funksiyalarni amalga oshiring.
`,
			hint1: 'Samarali hisoblash uchun np.mean() va np.var() dan foydalaning.',
			hint2: 'Dispersiya - o\'rtachadan chetlanishlar kvadratlarining o\'rtachasi.',
			whyItMatters: `O'rtacha va dispersiya ma'lumotlarni tushunish uchun asosiy. Batch normalization harakatlanuvchi o'rtacha/dispersiyadan foydalanadi.`,
		},
	},
};

export default task;
