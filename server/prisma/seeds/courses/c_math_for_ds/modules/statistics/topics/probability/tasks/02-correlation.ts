import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-correlation',
	title: 'Correlation Coefficient',
	difficulty: 'medium',
	tags: ['python', 'math', 'statistics', 'correlation', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Correlation Coefficient

Implement Pearson correlation coefficient between two variables.

## Formula

r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]

Range: [-1, 1], where 1 = perfect positive, -1 = perfect negative, 0 = no linear relationship
`,
	initialCode: `import numpy as np

def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    return np.corrcoef(x, y)[0, 1]
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

class TestCorrelation(unittest.TestCase):
    def test_perfect_positive(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = correlation(x, y)
        assert_close(result, 1.0)

    def test_perfect_negative(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        result = correlation(x, y)
        assert_close(result, -1.0)

    def test_no_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, -1, 1, -1, 1])
        result = correlation(x, y)
        assert -0.5 < result < 0.5

    def test_identical_arrays(self):
        x = np.array([1, 2, 3, 4, 5])
        result = correlation(x, x)
        assert_close(result, 1.0)

    def test_range_bounds(self):
        x = np.random.randn(100)
        y = np.random.randn(100)
        result = correlation(x, y)
        assert -1.0 <= result <= 1.0

    def test_linear_relationship(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 3 + np.array([0.1, -0.1, 0.1, -0.1, 0.1])
        result = correlation(x, y)
        assert result > 0.99

    def test_negative_values(self):
        x = np.array([-2, -1, 0, 1, 2])
        y = np.array([-4, -2, 0, 2, 4])
        result = correlation(x, y)
        assert_close(result, 1.0)

    def test_floats(self):
        x = np.array([1.5, 2.5, 3.5, 4.5])
        y = np.array([3.0, 5.0, 7.0, 9.0])
        result = correlation(x, y)
        assert_close(result, 1.0)

    def test_returns_scalar(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = correlation(x, y)
        assert np.isscalar(result or result.shape == ())

    def test_symmetric(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([4, 3, 2, 1])
        r1 = correlation(x, y)
        r2 = correlation(y, x)
        assert_close(r1, r2)

`,
	hint1: 'Use np.corrcoef(x, y)[0, 1] to get the correlation coefficient.',
	hint2: 'Alternatively, compute (covariance) / (std_x * std_y)',
	whyItMatters: `Correlation helps identify relationships between features. High correlation between features (multicollinearity) can hurt regression models. **Production Pattern:** Feature selection often removes highly correlated features to reduce redundancy.`,
	translations: {
		ru: {
			title: 'Коэффициент корреляции',
			description: `# Коэффициент корреляции

Реализуйте коэффициент корреляции Пирсона между двумя переменными.

Диапазон: [-1, 1], где 1 = идеальная положительная, -1 = идеальная отрицательная, 0 = нет линейной связи
`,
			hint1: 'Используйте np.corrcoef(x, y)[0, 1]',
			hint2: 'Альтернативно: ковариация / (std_x * std_y)',
			whyItMatters: `Корреляция помогает выявить связи между признаками. **Production Pattern:** Отбор признаков часто удаляет высоко коррелированные признаки.`,
		},
		uz: {
			title: 'Korrelyatsiya koeffitsienti',
			description: `# Korrelyatsiya koeffitsienti

Ikki o'zgaruvchi orasidagi Pirson korrelyatsiya koeffitsientini amalga oshiring.
`,
			hint1: 'np.corrcoef(x, y)[0, 1] dan foydalaning',
			hint2: 'Muqobil: kovariatsiya / (std_x * std_y)',
			whyItMatters: `Korrelyatsiya xususiyatlar orasidagi aloqalarni aniqlashga yordam beradi.`,
		},
	},
};

export default task;
