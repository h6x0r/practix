import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-polynomial-regression',
	title: 'Polynomial Regression',
	difficulty: 'medium',
	tags: ['sklearn', 'regression', 'polynomial'],
	estimatedTime: '15m',
	isPremium: false,
	order: 2,
	description: `# Polynomial Regression

Polynomial regression captures non-linear relationships using polynomial features.

## Task

Implement three functions:
1. \`create_polynomial_features(X, degree)\` - Transform features to polynomial
2. \`train_polynomial_regression(X, y, degree)\` - Train polynomial model
3. \`compare_degrees(X, y, degrees)\` - Compare R² for different degrees

## Example

\`\`\`python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x²

X_poly = create_polynomial_features(X, degree=2)
model = train_polynomial_regression(X, y, degree=2)
scores = compare_degrees(X, y, [1, 2, 3])  # {1: 0.95, 2: 1.0, 3: 1.0}
\`\`\``,

	initialCode: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Transform features to polynomial. Return transformed X."""
    # Your code here
    pass

def train_polynomial_regression(X: np.ndarray, y: np.ndarray, degree: int):
    """Train polynomial regression. Return (model, poly_transformer)."""
    # Your code here
    pass

def compare_degrees(X: np.ndarray, y: np.ndarray, degrees: list) -> dict:
    """Compare R² scores for different degrees. Return {degree: score}."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Transform features to polynomial. Return transformed X."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

def train_polynomial_regression(X: np.ndarray, y: np.ndarray, degree: int):
    """Train polynomial regression. Return (model, poly_transformer)."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly

def compare_degrees(X: np.ndarray, y: np.ndarray, degrees: list) -> dict:
    """Compare R² scores for different degrees. Return {degree: score}."""
    scores = {}
    for degree in degrees:
        model, poly = train_polynomial_regression(X, y, degree)
        X_poly = poly.transform(X)
        scores[degree] = model.score(X_poly, y)
    return scores
`,

	testCode: `import numpy as np
import unittest

class TestPolynomialRegression(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])

    def test_create_polynomial_features_shape(self):
        X_poly = create_polynomial_features(self.X, 2)
        self.assertEqual(X_poly.shape[1], 2)

    def test_create_polynomial_features_degree3(self):
        X_poly = create_polynomial_features(self.X, 3)
        self.assertEqual(X_poly.shape[1], 3)

    def test_train_returns_tuple(self):
        result = train_polynomial_regression(self.X, self.y, 2)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_model_has_coef(self):
        model, _ = train_polynomial_regression(self.X, self.y, 2)
        self.assertTrue(hasattr(model, 'coef_'))

    def test_degree2_perfect_fit(self):
        model, poly = train_polynomial_regression(self.X, self.y, 2)
        X_poly = poly.transform(self.X)
        score = model.score(X_poly, self.y)
        self.assertGreater(score, 0.99)

    def test_compare_degrees_returns_dict(self):
        scores = compare_degrees(self.X, self.y, [1, 2])
        self.assertIsInstance(scores, dict)
        self.assertIn(1, scores)
        self.assertIn(2, scores)

    def test_higher_degree_better_fit(self):
        scores = compare_degrees(self.X, self.y, [1, 2])
        self.assertGreater(scores[2], scores[1])

    def test_poly_features_returns_numpy(self):
        X_poly = create_polynomial_features(self.X, 2)
        self.assertIsInstance(X_poly, np.ndarray)

    def test_compare_degrees_scores_in_range(self):
        scores = compare_degrees(self.X, self.y, [1, 2, 3])
        for score in scores.values():
            self.assertTrue(0 <= score <= 1)

    def test_polynomial_features_samples_preserved(self):
        X_poly = create_polynomial_features(self.X, 3)
        self.assertEqual(X_poly.shape[0], self.X.shape[0])
`,

	hint1: 'Use PolynomialFeatures(degree=n).fit_transform(X) to create polynomial features',
	hint2: 'Fit LinearRegression on transformed polynomial features',

	whyItMatters: `Polynomial regression is important for:

- **Non-linear relationships**: Capture curves and bends
- **Feature engineering**: Create interaction terms
- **Flexibility**: Balance between linear and complex models
- **Overfitting awareness**: Learn about bias-variance tradeoff

Foundation for understanding model complexity.`,

	translations: {
		ru: {
			title: 'Полиномиальная регрессия',
			description: `# Полиномиальная регрессия

Полиномиальная регрессия улавливает нелинейные связи с помощью полиномиальных признаков.

## Задача

Реализуйте три функции:
1. \`create_polynomial_features(X, degree)\` - Преобразовать признаки в полиномиальные
2. \`train_polynomial_regression(X, y, degree)\` - Обучить полиномиальную модель
3. \`compare_degrees(X, y, degrees)\` - Сравнить R² для разных степеней

## Пример

\`\`\`python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x²

X_poly = create_polynomial_features(X, degree=2)
model = train_polynomial_regression(X, y, degree=2)
scores = compare_degrees(X, y, [1, 2, 3])  # {1: 0.95, 2: 1.0, 3: 1.0}
\`\`\``,
			hint1: 'Используйте PolynomialFeatures(degree=n).fit_transform(X)',
			hint2: 'Обучите LinearRegression на преобразованных полиномиальных признаках',
			whyItMatters: `Полиномиальная регрессия важна для:

- **Нелинейные связи**: Улавливание кривых и изгибов
- **Feature engineering**: Создание взаимодействий признаков
- **Гибкость**: Баланс между простыми и сложными моделями`,
		},
		uz: {
			title: 'Polinomial regressiya',
			description: `# Polinomial regressiya

Polinomial regressiya polinomial xususiyatlar yordamida nochiziqli munosabatlarni ushlaydi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`create_polynomial_features(X, degree)\` - Xususiyatlarni polinomialga aylantirish
2. \`train_polynomial_regression(X, y, degree)\` - Polinomial modelni o'rgatish
3. \`compare_degrees(X, y, degrees)\` - Turli darajalar uchun R² ni taqqoslash

## Misol

\`\`\`python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x²

X_poly = create_polynomial_features(X, degree=2)
model = train_polynomial_regression(X, y, degree=2)
scores = compare_degrees(X, y, [1, 2, 3])  # {1: 0.95, 2: 1.0, 3: 1.0}
\`\`\``,
			hint1: "PolynomialFeatures(degree=n).fit_transform(X) dan foydalaning",
			hint2: "Aylantrilgan polinomial xususiyatlarda LinearRegression ni o'rgating",
			whyItMatters: `Polinomial regressiya quyidagilar uchun muhim:

- **Nochiziqli munosabatlar**: Egri chiziqlar va burilishlarni ushlash
- **Feature engineering**: O'zaro ta'sir terminlarini yaratish
- **Moslashuvchanlik**: Oddiy va murakkab modellar o'rtasidagi muvozanat`,
		},
	},
};

export default task;
