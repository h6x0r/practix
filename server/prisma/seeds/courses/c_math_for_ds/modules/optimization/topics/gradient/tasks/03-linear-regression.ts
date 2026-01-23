import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'math-linear-regression',
	title: 'Linear Regression',
	difficulty: 'medium',
	tags: ['python', 'math', 'optimization', 'linear-regression', 'numpy'],
	estimatedTime: '25m',
	isPremium: true,
	order: 3,
	description: `# Linear Regression with Gradient Descent

Implement linear regression using gradient descent optimization.

## Model

y = X @ w + b

## Gradients

∂L/∂w = (2/n) × Xᵀ @ (predictions - y)
∂L/∂b = (2/n) × sum(predictions - y)
`,
	initialCode: `import numpy as np

def assert_array_close(actual, expected, msg=""):
    """Helper for array comparison with clear error message"""
    if not np.allclose(actual, expected):
        raise AssertionError(f"Expected {expected.tolist()}, got {actual.tolist()}")

def assert_close(actual, expected, places=5, msg=""):
    """Helper for scalar comparison with clear error message"""
    if abs(actual - expected) > 10**(-places):
        raise AssertionError(f"Expected {expected}, got {actual}")

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, iterations: int = 100) -> None:
        """Train the model."""
        # Your code here
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        # Your code here
        pass
`,
	solutionCode: `import numpy as np

def assert_array_close(actual, expected, msg=""):
    """Helper for array comparison with clear error message"""
    if not np.allclose(actual, expected):
        raise AssertionError(f"Expected {expected.tolist()}, got {actual.tolist()}")

def assert_close(actual, expected, places=5, msg=""):
    """Helper for scalar comparison with clear error message"""
    if abs(actual - expected) > 10**(-places):
        raise AssertionError(f"Expected {expected}, got {actual}")

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, iterations: int = 100) -> None:
        """Train the model."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(iterations):
            predictions = X @ self.weights + self.bias
            error = predictions - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return X @ self.weights + self.bias
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

class TestLinearRegression(unittest.TestCase):
    def test_perfect_linear(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = 2 * X.flatten() + 1
        model = LinearRegression(learning_rate=0.1)
        model.fit(X, y, iterations=1000)
        preds = model.predict(X)
        assert_array_close(preds, y)

    def test_weights_shape(self):
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        model = LinearRegression()
        model.fit(X, y, iterations=10)
        assert model.weights.shape == (3,, f"Expected (3,, got {model.weights.shape}")

    def test_prediction_shape(self):
        X = np.random.randn(50, 4)
        y = np.random.randn(50)
        model = LinearRegression()
        model.fit(X, y, iterations=10)
        preds = model.predict(X)
        assert preds.shape == (50,, f"Expected (50,, got {preds.shape}")

    def test_bias_updated(self):
        X = np.ones((10, 1))
        y = np.full(10, 5.0)
        model = LinearRegression(learning_rate=0.1)
        model.fit(X, y, iterations=100)
        assert model.bias - 5.0  < 1.0 or abs(model.weights[0] - 5.0 < 1.0)

    def test_multivariate(self):
        X = np.random.randn(100, 5)
        true_w = np.array([1, 2, 3, 4, 5])
        y = X @ true_w + np.random.randn(100) * 0.1
        model = LinearRegression(learning_rate=0.01)
        model.fit(X, y, iterations=500)
        preds = model.predict(X)
        mse = np.mean((preds - y) ** 2)
        assert mse < 1.0

    def test_no_nan(self):
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        model = LinearRegression()
        model.fit(X, y, iterations=100)
        preds = model.predict(X)
        self.assertFalse(np.any(np.isnan(preds)))

    def test_zero_intercept(self):
        X = np.array([[1], [2], [3]])
        y = 2 * X.flatten()
        model = LinearRegression(learning_rate=0.1)
        model.fit(X, y, iterations=500)
        assert model.bias  < 0.5

    def test_negative_slope(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([4, 3, 2, 1])
        model = LinearRegression(learning_rate=0.1)
        model.fit(X, y, iterations=500)
        assert model.weights[0] < 0

    def test_predict_returns_array(self):
        X = np.array([[1], [2]])
        y = np.array([1, 2])
        model = LinearRegression()
        model.fit(X, y)
        result = model.predict(X)
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result).__name__}"

    def test_learning_reduces_error(self):
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        model = LinearRegression(learning_rate=0.01)
        model.fit(X, y, iterations=1)
        preds1 = model.predict(X)
        mse1 = np.mean((preds1 - y) ** 2)
        model.fit(X, y, iterations=100)
        preds2 = model.predict(X)
        mse2 = np.mean((preds2 - y) ** 2)
        assert mse2 <= mse1

`,
	hint1: 'Initialize weights to zeros, update using gradient: w -= lr * dw',
	hint2: 'Gradients: dw = (2/n) * X.T @ (pred - y), db = (2/n) * sum(pred - y)',
	whyItMatters: `Linear regression is the foundation of deep learning. Neural networks are just many linear regressions with activations. **Production Pattern:** sklearn LinearRegression uses closed-form solution, but gradient descent scales to billions of samples.`,
	translations: {
		ru: {
			title: 'Линейная регрессия',
			description: `# Линейная регрессия с градиентным спуском

Реализуйте линейную регрессию с оптимизацией градиентным спуском.

## Модель

y = X @ w + b
`,
			hint1: 'Инициализируйте веса нулями, обновляйте: w -= lr * dw',
			hint2: 'Градиенты: dw = (2/n) * X.T @ (pred - y), db = (2/n) * sum(pred - y)',
			whyItMatters: `Линейная регрессия - фундамент глубокого обучения. Нейросети - это много линейных регрессий с активациями. **Production Pattern:** Градиентный спуск масштабируется на миллиарды примеров.`,
		},
		uz: {
			title: 'Chiziqli regressiya',
			description: `# Gradient tushishi bilan chiziqli regressiya

Gradient tushish optimizatsiyasi bilan chiziqli regressiyani amalga oshiring.
`,
			hint1: 'Og\'irliklarni nolga boshlang, yangilang: w -= lr * dw',
			hint2: 'Gradientlar: dw = (2/n) * X.T @ (pred - y)',
			whyItMatters: `Chiziqli regressiya chuqur o'rganishning asosi. Neyron tarmoqlar - bu aktivatsiyalar bilan ko'p chiziqli regressiyalar.`,
		},
	},
};

export default task;
