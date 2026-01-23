import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-mse-loss',
	title: 'MSE Loss Function',
	difficulty: 'easy',
	tags: ['python', 'math', 'optimization', 'loss', 'mse', 'numpy'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Mean Squared Error Loss

Implement MSE loss and its gradient for regression tasks.

## Formula

MSE = (1/n) Σ(yᵢ - ŷᵢ)²
Gradient w.r.t. ŷ = (2/n)(ŷ - y)
`,
	initialCode: `import numpy as np

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    # Your code here
    pass

def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute gradient of MSE w.r.t. predictions."""
    # Your code here
    pass
`,
	solutionCode: `import numpy as np

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute gradient of MSE w.r.t. predictions."""
    n = len(y_true)
    return (2 / n) * (y_pred - y_true)
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

class TestMSE(unittest.TestCase):
    def test_perfect_prediction(self):
        y = np.array([1, 2, 3])
        result = mse_loss(y, y)
        assert_close(result, 0.0)

    def test_simple_mse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        result = mse_loss(y_true, y_pred)
        assert_close(result, 1.0)

    def test_gradient_direction(self):
        y_true = np.array([0.0])
        y_pred = np.array([1.0])
        grad = mse_gradient(y_true, y_pred)
        assert grad[0] > 0

    def test_gradient_zero_at_correct(self):
        y = np.array([1.0, 2.0, 3.0])
        grad = mse_gradient(y, y)
        assert_array_close(grad, [0, 0, 0])

    def test_mse_positive(self):
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        result = mse_loss(y_true, y_pred)
        assert result >= 0

    def test_mse_symmetric(self):
        y_true = np.array([1, 2])
        y_pred = np.array([3, 4])
        mse1 = mse_loss(y_true, y_pred)
        mse2 = mse_loss(y_pred, y_true)
        assert_close(mse1, mse2)

    def test_gradient_shape(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 3, 4, 5])
        grad = mse_gradient(y_true, y_pred)
        assert grad.shape == y_true.shape, f"Expected y_true.shape, got {grad.shape}"

    def test_mse_returns_float(self):
        result = mse_loss(np.array([1]), np.array([2]))
        self.assertIsInstance(float(result), float)

    def test_gradient_magnitude(self):
        y_true = np.array([0.0])
        y_pred = np.array([2.0])
        grad = mse_gradient(y_true, y_pred)
        assert_close(grad[0], 4.0)

    def test_larger_error_larger_mse(self):
        y_true = np.array([0, 0])
        y_pred_small = np.array([1, 1])
        y_pred_large = np.array([2, 2])
        assert mse_loss(y_true, y_pred_large > mse_loss(y_true, y_pred_small))

`,
	hint1: 'MSE: np.mean((y_true - y_pred) ** 2)',
	hint2: 'Gradient: (2/n) * (y_pred - y_true)',
	whyItMatters: `MSE is the standard loss for regression. Linear regression minimizes MSE. Understanding the gradient helps understand how models learn. **Production Pattern:** Many regression models use MSE or variants like RMSE, MAE.`,
	translations: {
		ru: {
			title: 'Функция потерь MSE',
			description: `# Среднеквадратичная ошибка

Реализуйте MSE loss и её градиент для задач регрессии.

## Формула

MSE = (1/n) Σ(yᵢ - ŷᵢ)²
`,
			hint1: 'MSE: np.mean((y_true - y_pred) ** 2)',
			hint2: 'Градиент: (2/n) * (y_pred - y_true)',
			whyItMatters: `MSE - стандартная функция потерь для регрессии. Понимание градиента помогает понять, как модели учатся.`,
		},
		uz: {
			title: 'MSE yo\'qotish funksiyasi',
			description: `# O'rtacha kvadratik xato

Regressiya vazifalari uchun MSE yo'qotish va uning gradientini amalga oshiring.
`,
			hint1: 'MSE: np.mean((y_true - y_pred) ** 2)',
			hint2: 'Gradient: (2/n) * (y_pred - y_true)',
			whyItMatters: `MSE regressiya uchun standart yo'qotish funksiyasi. Gradientni tushunish modellar qanday o'rganishini tushunishga yordam beradi.`,
		},
	},
};

export default task;
