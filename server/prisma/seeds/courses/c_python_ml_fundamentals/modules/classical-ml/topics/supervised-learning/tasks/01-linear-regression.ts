import { Task } from '../../../../../../../types';

const task: Task = {
	slug: 'sklearn-linear-regression',
	title: 'Linear Regression',
	difficulty: 'easy',
	tags: ['sklearn', 'regression', 'linear'],
	estimatedTime: '15m',
	isPremium: false,
	order: 1,
	description: `# Linear Regression

Linear regression models the relationship between features and a continuous target.

## Task

Implement three functions:
1. \`train_linear_regression(X, y)\` - Train model and return it
2. \`predict_and_score(model, X, y)\` - Get predictions and R² score
3. \`get_coefficients(model)\` - Return coefficients and intercept

## Example

\`\`\`python
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = train_linear_regression(X, y)
preds, score = predict_and_score(model, X, y)
coef, intercept = get_coefficients(model)
\`\`\``,

	initialCode: `import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_regression(X: np.ndarray, y: np.ndarray):
    """Train linear regression model. Return fitted model."""
    # Your code here
    pass

def predict_and_score(model, X: np.ndarray, y: np.ndarray) -> tuple:
    """Get predictions and R² score. Return (predictions, score)."""
    # Your code here
    pass

def get_coefficients(model) -> tuple:
    """Return (coefficients, intercept)."""
    # Your code here
    pass
`,

	solutionCode: `import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_regression(X: np.ndarray, y: np.ndarray):
    """Train linear regression model. Return fitted model."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_and_score(model, X: np.ndarray, y: np.ndarray) -> tuple:
    """Get predictions and R² score. Return (predictions, score)."""
    predictions = model.predict(X)
    score = model.score(X, y)
    return predictions, score

def get_coefficients(model) -> tuple:
    """Return (coefficients, intercept)."""
    return model.coef_, model.intercept_
`,

	testCode: `import numpy as np
import unittest

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    def test_train_returns_model(self):
        model = train_linear_regression(self.X, self.y)
        self.assertIsNotNone(model)

    def test_model_has_coef(self):
        model = train_linear_regression(self.X, self.y)
        self.assertTrue(hasattr(model, 'coef_'))

    def test_predict_and_score_returns_tuple(self):
        model = train_linear_regression(self.X, self.y)
        result = predict_and_score(model, self.X, self.y)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_predictions_shape(self):
        model = train_linear_regression(self.X, self.y)
        preds, _ = predict_and_score(model, self.X, self.y)
        self.assertEqual(len(preds), len(self.y))

    def test_perfect_fit_score(self):
        model = train_linear_regression(self.X, self.y)
        _, score = predict_and_score(model, self.X, self.y)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_get_coefficients_returns_tuple(self):
        model = train_linear_regression(self.X, self.y)
        result = get_coefficients(model)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_coefficient_value(self):
        model = train_linear_regression(self.X, self.y)
        coef, intercept = get_coefficients(model)
        self.assertAlmostEqual(coef[0], 2.0, places=5)
        self.assertAlmostEqual(intercept, 0.0, places=5)

    def test_predictions_are_numpy(self):
        model = train_linear_regression(self.X, self.y)
        preds, _ = predict_and_score(model, self.X, self.y)
        self.assertIsInstance(preds, np.ndarray)

    def test_model_can_predict_new_data(self):
        model = train_linear_regression(self.X, self.y)
        X_new = np.array([[6], [7]])
        preds = model.predict(X_new)
        self.assertAlmostEqual(preds[0], 12.0, places=5)
        self.assertAlmostEqual(preds[1], 14.0, places=5)

    def test_score_in_valid_range(self):
        model = train_linear_regression(self.X, self.y)
        _, score = predict_and_score(model, self.X, self.y)
        self.assertTrue(0 <= score <= 1)
`,

	hint1: 'Use LinearRegression().fit(X, y) to train, model.predict(X) to predict',
	hint2: 'Use model.score(X, y) for R², model.coef_ and model.intercept_ for parameters',

	whyItMatters: `Linear regression is foundational because:

- **Interpretability**: Coefficients explain feature impact
- **Baseline model**: Compare complex models against it
- **Feature importance**: Identify key predictors
- **Simplicity**: Fast training and inference

The starting point for understanding ML.`,

	translations: {
		ru: {
			title: 'Линейная регрессия',
			description: `# Линейная регрессия

Линейная регрессия моделирует связь между признаками и непрерывной целевой переменной.

## Задача

Реализуйте три функции:
1. \`train_linear_regression(X, y)\` - Обучить модель и вернуть её
2. \`predict_and_score(model, X, y)\` - Получить предсказания и R² score
3. \`get_coefficients(model)\` - Вернуть коэффициенты и свободный член

## Пример

\`\`\`python
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = train_linear_regression(X, y)
preds, score = predict_and_score(model, X, y)
coef, intercept = get_coefficients(model)
\`\`\``,
			hint1: 'Используйте LinearRegression().fit(X, y) для обучения',
			hint2: 'Используйте model.score(X, y) для R², model.coef_ для коэффициентов',
			whyItMatters: `Линейная регрессия фундаментальна потому что:

- **Интерпретируемость**: Коэффициенты объясняют влияние признаков
- **Базовая модель**: Сравнивайте сложные модели с ней
- **Важность признаков**: Определяйте ключевые предикторы`,
		},
		uz: {
			title: 'Chiziqli regressiya',
			description: `# Chiziqli regressiya

Chiziqli regressiya xususiyatlar va uzluksiz maqsad o'rtasidagi munosabatni modellashtiradi.

## Topshiriq

Uchta funksiyani amalga oshiring:
1. \`train_linear_regression(X, y)\` - Modelni o'rgatish va qaytarish
2. \`predict_and_score(model, X, y)\` - Bashoratlar va R² ball olish
3. \`get_coefficients(model)\` - Koeffitsientlar va interceptni qaytarish

## Misol

\`\`\`python
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = train_linear_regression(X, y)
preds, score = predict_and_score(model, X, y)
coef, intercept = get_coefficients(model)
\`\`\``,
			hint1: "O'rgatish uchun LinearRegression().fit(X, y) dan foydalaning",
			hint2: "R² uchun model.score(X, y), koeffitsientlar uchun model.coef_ dan foydalaning",
			whyItMatters: `Chiziqli regressiya asosiy chunki:

- **Interpretatsiya**: Koeffitsientlar xususiyat ta'sirini tushuntiradi
- **Bazaviy model**: Murakkab modellarni u bilan taqqoslang
- **Xususiyat ahamiyati**: Asosiy prediktorlarni aniqlang`,
		},
	},
};

export default task;
